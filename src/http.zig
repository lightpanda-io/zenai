const std = @import("std");
const json = @import("json.zig");
const retry = @import("retry.zig");

/// Error set returned by `fetchJsonWithRetry`. Each provider's `ApiError`
/// is a superset, adding its own preflight errors (`MissingApiKey` for the
/// keyed providers; keenable, which supports keyless calls, adds
/// `MissingAppTitle` instead).
pub const FetchError = error{
    ApiError,
    EmptyResponse,
} || std.http.Client.FetchError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

/// Non-2xx detail captured via a client's `setErrorDetail` for later
/// inspection. `message` is the one-line message of a provider JSON error
/// (see `extractErrorMessage`), otherwise the raw body; null when the body
/// was empty.
/// Owns `message`; `set` frees the previous message, `deinit` the last.
pub const ErrorDetail = struct {
    status: ?u10 = null,
    message: ?[]u8 = null,

    pub fn set(self: *ErrorDetail, allocator: std.mem.Allocator, status: u10, body: []const u8) void {
        self.setLogged(allocator, status, body, null);
    }

    pub fn setLogged(self: *ErrorDetail, allocator: std.mem.Allocator, status: u10, body: []const u8, log_tag: ?[]const u8) void {
        self.status = status;
        if (self.message) |old| allocator.free(old);
        self.message = null;
        if (body.len > 0) {
            if (log_tag) |tag| std.log.err("{s} API error (HTTP {d}): {s}", .{ tag, status, body });
            self.message = extractErrorMessage(allocator, body) orelse (allocator.dupe(u8, body) catch null);
        }
    }

    pub fn deinit(self: *ErrorDetail, allocator: std.mem.Allocator) void {
        if (self.message) |b| allocator.free(b);
        self.* = .{};
    }

    /// A copy owning its own `message`, to keep past the client's next call
    /// or `deinit`.
    pub fn clone(self: ErrorDetail, allocator: std.mem.Allocator) std.mem.Allocator.Error!ErrorDetail {
        return .{
            .status = self.status,
            .message = if (self.message) |m| try allocator.dupe(u8, m) else null,
        };
    }
};

/// Cross-thread trigger for aborting an in-flight HTTP request. A request path
/// arms it with the active connection's stream around the blocking read (see
/// `armInterrupt`); another thread (the SIGINT handler) calls `fire` to
/// `shutdown` that socket, which unblocks the read so the request returns an
/// error instead of waiting for the server. The errored connection is dropped
/// from the pool by `Request.deinit`.
///
/// `fire` is sticky: one landing before the socket is armed (during connect/TLS)
/// is honored by `arm` rather than lost. `reset` clears it per turn.
pub const Interrupt = struct {
    /// The armed connection's stream reader, which carries both the stream
    /// and the `Io` needed to shut it down.
    reader: std.atomic.Value(?*const std.Io.net.Stream.Reader) = .init(null),
    fired: std.atomic.Value(bool) = .init(false),

    fn arm(self: *Interrupt, reader: *const std.Io.net.Stream.Reader) void {
        self.reader.store(reader, .release);
        if (self.fired.load(.acquire)) reader.stream.shutdown(reader.io, .both) catch {};
    }

    fn disarm(self: *Interrupt) void {
        self.reader.store(null, .release);
    }

    pub fn fire(self: *Interrupt) void {
        self.fired.store(true, .release);
        if (self.reader.load(.acquire)) |r| r.stream.shutdown(r.io, .both) catch {};
    }

    pub fn isFired(self: *const Interrupt) bool {
        return self.fired.load(.acquire);
    }

    /// Clear armed + fired state so the interrupt is reusable next turn.
    pub fn reset(self: *Interrupt) void {
        self.fired.store(false, .release);
        self.reader.store(null, .release);
    }
};

/// Arms an optional `Interrupt` with `req`'s connection stream so a cross-thread
/// `Interrupt.fire` can abort a blocking read on that connection. The guard
/// must outlive the read but be torn down before `req.deinit` (LIFO) so a late
/// `fire` can't `shutdown` a socket the pool has already recycled:
///
///     var guard = http.armInterrupt(interrupt, &req);
///     defer guard.deinit();
///     errdefer guard.poison();   // also poison on non-error abort paths
pub const InterruptGuard = struct {
    interrupt: ?*Interrupt,
    req: *std.http.Client.Request,

    /// Mark the connection unusable so `req.deinit` destroys it rather than
    /// returning a socket with unknown framing (after an abort or read error)
    /// to the pool. `receiveHead` only auto-marks `closing` once it succeeds,
    /// so an abort during the header or body read must poison explicitly.
    pub fn poison(self: InterruptGuard) void {
        if (self.req.connection) |conn| conn.closing = true;
    }

    pub fn deinit(self: InterruptGuard) void {
        if (self.interrupt) |it| it.disarm();
    }
};

pub fn armInterrupt(interrupt: ?*Interrupt, req: *std.http.Client.Request) InterruptGuard {
    if (interrupt) |it| {
        if (req.connection) |conn| it.arm(&conn.stream_reader);
    }
    return .{ .interrupt = interrupt, .req = req };
}

/// Bounds one request's wall-clock time from an established connection to
/// the end of the response body. A private `Interrupt` shuts the socket
/// down at the deadline, exactly as a Ctrl-C would, so the blocking read
/// fails instead of waiting on the server. The connect phase is out of
/// reach: std's `ConnectTcpOptions.timeout` is declared but unimplemented.
///
/// A raw `std.Thread` rather than `io.concurrent`: consumers run a
/// single-threaded `Io.Threaded`, where `concurrent` is unavailable and
/// cancellation is a no-op. One fixed deadline for the whole exchange, so
/// not suitable for long-lived streams without a resettable deadline.
const Watchdog = struct {
    interrupt: Interrupt = .{},
    done: std.Io.Event = .unset,
    thread: ?std.Thread = null,

    fn start(self: *Watchdog, io: std.Io, req: *std.http.Client.Request, timeout_ms: u32) error{SystemResources}!void {
        _ = armInterrupt(&self.interrupt, req);
        const deadline: std.Io.Clock.Timestamp = .fromNow(io, .{ .raw = .fromMilliseconds(timeout_ms), .clock = .awake });
        // Default stack: under libc it also holds the host's static TLS, which
        // can outgrow a small fixed size and fail the spawn.
        self.thread = std.Thread.spawn(.{}, run, .{ self, io, deadline }) catch return error.SystemResources;
    }

    fn run(self: *Watchdog, io: std.Io, deadline: std.Io.Clock.Timestamp) void {
        // Spurious wakeups (e.g. EINTR) also report Timeout; only the deadline counts.
        while (true) {
            if (self.done.waitTimeout(io, .{ .deadline = deadline })) |_| return else |_| {}
            if (deadline.compare(.lte, .now(io, .awake))) break;
        }
        self.interrupt.fire();
    }

    /// Must run before `req.deinit`: joins the thread so a late `fire` can't
    /// hit a recycled socket, and poisons a connection the deadline shut down
    /// just after the body had already completed.
    fn stop(self: *Watchdog, io: std.Io, guard: InterruptGuard) void {
        const thread = self.thread orelse return;
        self.done.set(io);
        thread.join();
        if (self.interrupt.isFired()) guard.poison();
    }
};

/// Common HTTP fetch + retry + JSON parse pipeline shared by all provider
/// clients. On a non-retryable HTTP error response, calls
/// `error_handler.setErrorDetail(status, body)` so the caller can record
/// provider-specific error detail before this function returns
/// `error.ApiError`. `timeout_ms` bounds each attempt from an established
/// connection to the end of the body (see `Watchdog`); null waits
/// indefinitely.
pub fn fetchJsonWithRetry(
    allocator: std.mem.Allocator,
    http_client: *std.http.Client,
    policy: retry.RetryPolicy,
    timeout_ms: ?u32,
    options: std.http.Client.FetchOptions,
    comptime T: type,
    error_handler: anytype,
) FetchError!Response(T) {
    // Provider clients carry an optional `interrupt` so a SIGINT can abort an
    // in-flight read; clients without the field (e.g. tavily) opt out at comptime.
    const interrupt: ?*Interrupt = if (@hasField(@TypeOf(error_handler.*), "interrupt"))
        error_handler.interrupt
    else
        null;
    var attempt: u8 = 0;
    while (true) : (attempt += 1) {
        var response_buf: std.Io.Writer.Allocating = .init(allocator);
        var keep_buf = false;
        defer if (!keep_buf) response_buf.deinit();

        var retry_after_ms: ?u32 = null;
        const status = fetchCapturingRetryAfter(allocator, http_client, options, &response_buf.writer, interrupt, timeout_ms, &retry_after_ms) catch |err| {
            // Don't retry a request the user cancelled.
            if (interrupt) |it| if (it.isFired()) return err;
            if (retry.isRetryableFetchError(err) and attempt + 1 < policy.max_attempts) {
                retry.sleepBackoff(http_client.io, attempt, policy);
                continue;
            }
            return err;
        };

        const body = response_buf.written();
        const status_code: u10 = @intFromEnum(status);
        if (status_code >= 200 and status_code < 300) {
            if (body.len == 0) return error.EmptyResponse;
            const parsed = try std.json.parseFromSlice(T, allocator, body, .{ .ignore_unknown_fields = true });
            keep_buf = true;
            return .{ .value = parsed.value, .json_buf = response_buf, .parsed = parsed };
        }

        if (retry.isRetryableStatus(status_code) and attempt + 1 < policy.max_attempts) {
            retry.sleepBackoffHinted(http_client.io, attempt, policy, retry_after_ms);
            continue;
        }
        error_handler.setErrorDetail(status_code, body);
        return error.ApiError;
    }
}

/// A single HTTP round-trip into `response_writer`, mirroring
/// `std.http.Client.fetch` (redirects, content-encoding) but arming
/// `interrupt` with the connection's stream around the blocking read so a
/// SIGINT on another thread can abort it. Returns the response status.
pub fn fetchInterruptible(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    options: std.http.Client.FetchOptions,
    response_writer: *std.Io.Writer,
    interrupt: ?*Interrupt,
) std.http.Client.FetchError!std.http.Status {
    var retry_after_ms: ?u32 = null;
    return fetchCapturingRetryAfter(allocator, client, options, response_writer, interrupt, null, &retry_after_ms);
}

/// JSON-encode a request body the way every provider expects it: optional
/// fields set to null are omitted rather than sent as `null`.
fn encodeBody(allocator: std.mem.Allocator, body: anytype) std.mem.Allocator.Error![]u8 {
    return json.stringifyAlloc(allocator, body, .{ .emit_null_optional_fields = false });
}

/// `fetchJsonWithRetry` for a JSON POST: encodes `body` with `encodeBody`
/// and sets the content type.
pub fn postJsonWithRetry(
    allocator: std.mem.Allocator,
    http_client: *std.http.Client,
    policy: retry.RetryPolicy,
    timeout_ms: ?u32,
    url: []const u8,
    extra_headers: []const std.http.Header,
    body: anytype,
    comptime T: type,
    error_handler: anytype,
) FetchError!Response(T) {
    const payload = try encodeBody(allocator, body);
    defer allocator.free(payload);

    return fetchJsonWithRetry(allocator, http_client, policy, timeout_ms, .{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload,
        .extra_headers = extra_headers,
        .headers = .{ .content_type = .{ .override = "application/json" } },
    }, T, error_handler);
}

/// Server-provided retry hint from a response head: `Retry-After-Ms`
/// (milliseconds, Anthropic) wins over the standard `Retry-After` (seconds).
/// Parsed and clamped by `retry.parseRetryAfter`; null when absent or
/// unparseable (e.g. the HTTP-date form).
fn retryAfterFromHead(head: std.http.Client.Response.Head) ?u32 {
    var seconds_hint: ?u32 = null;
    var it = head.iterateHeaders();
    while (it.next()) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, "retry-after-ms")) {
            if (retry.parseRetryAfter(h.value, .milliseconds)) |ms| return ms;
        } else if (seconds_hint == null and std.ascii.eqlIgnoreCase(h.name, "retry-after")) {
            seconds_hint = retry.parseRetryAfter(h.value, .seconds);
        }
    }
    return seconds_hint;
}

/// `fetchInterruptible` plus capture of the response's Retry-After hint into
/// `retry_after_ms` (read from the head before the body is streamed, while
/// the header bytes are still valid) so `fetchJsonWithRetry` can honor it,
/// and an optional `timeout_ms` after which the exchange fails with
/// `error.Timeout` (see `Watchdog`).
fn fetchCapturingRetryAfter(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    options: std.http.Client.FetchOptions,
    response_writer: *std.Io.Writer,
    interrupt: ?*Interrupt,
    timeout_ms: ?u32,
    retry_after_ms: *?u32,
) std.http.Client.FetchError!std.http.Status {
    return reconnectOnStale(std.http.Client.FetchError!std.http.Status, interrupt, fetchOnce, .{ allocator, client, options, response_writer, interrupt, timeout_ms, retry_after_ms });
}

fn fetchOnce(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    options: std.http.Client.FetchOptions,
    response_writer: *std.Io.Writer,
    interrupt: ?*Interrupt,
    timeout_ms: ?u32,
    retry_after_ms: *?u32,
) std.http.Client.FetchError!std.http.Status {
    const uri = switch (options.location) {
        .url => |u| try std.Uri.parse(u),
        .uri => |u| u,
    };
    const method: std.http.Method = options.method orelse
        if (options.payload != null) .POST else .GET;
    // RedirectBehavior's integer value is the max redirect count: GET follows up
    // to 3 (matching std.http.Client.fetch), payload requests leave it unhandled.
    const redirect_behavior: std.http.Client.Request.RedirectBehavior = options.redirect_behavior orelse
        if (options.payload == null) @enumFromInt(3) else .unhandled;

    var req = try client.request(method, uri, .{
        .redirect_behavior = redirect_behavior,
        .headers = options.headers,
        .extra_headers = options.extra_headers,
        .privileged_headers = options.privileged_headers,
        .keep_alive = options.keep_alive,
    });
    defer req.deinit();

    // Arm the interrupt with this connection's stream around the send/receive, and
    // poison the connection on any failed exchange (including an interrupt-
    // induced read error) so `req.deinit` drops the socket instead of pooling
    // one with unknown framing.
    var guard = armInterrupt(interrupt, &req);
    defer guard.deinit();
    errdefer guard.poison();

    var watchdog: Watchdog = .{};
    if (timeout_ms) |ms| try watchdog.start(client.io, &req, ms);
    defer watchdog.stop(client.io, guard);

    return exchange(allocator, &req, redirect_behavior, options, response_writer, retry_after_ms) catch |err| {
        if (watchdog.interrupt.isFired()) return error.Timeout;
        return err;
    };
}

/// The send/receive half of `fetchOnce`, split out so its
/// caller can map any failure that lands while the watchdog has fired to
/// `error.Timeout`.
fn exchange(
    allocator: std.mem.Allocator,
    req: *std.http.Client.Request,
    redirect_behavior: std.http.Client.Request.RedirectBehavior,
    options: std.http.Client.FetchOptions,
    response_writer: *std.Io.Writer,
    retry_after_ms: *?u32,
) std.http.Client.FetchError!std.http.Status {
    const own_redirect_buffer = redirect_behavior != .unhandled and options.redirect_buffer == null;
    const redirect_buffer: []u8 = if (redirect_behavior == .unhandled) &.{} else options.redirect_buffer orelse try allocator.alloc(u8, 8 * 1024);
    defer if (own_redirect_buffer) allocator.free(redirect_buffer);

    var response = try sendReceiveHead(req, options.payload, redirect_buffer);
    retry_after_ms.* = retryAfterFromHead(response.head);

    const decompress_buffer: []u8 = switch (response.head.content_encoding) {
        .identity => &.{},
        .zstd => options.decompress_buffer orelse try allocator.alloc(u8, std.compress.zstd.default_window_len),
        .deflate, .gzip => options.decompress_buffer orelse try allocator.alloc(u8, std.compress.flate.max_window_len),
        .compress => return error.UnsupportedCompressionMethod,
    };
    defer if (options.decompress_buffer == null and decompress_buffer.len > 0) allocator.free(decompress_buffer);

    var transfer_buffer: [4096]u8 = undefined;
    var decompress: std.http.Decompress = undefined;
    const reader = response.readerDecompressing(&transfer_buffer, &decompress, decompress_buffer);

    _ = reader.streamRemaining(response_writer) catch |err| switch (err) {
        // A transport failure mid-body leaves `bodyErr` null: std records only
        // framing errors there. It stays generic rather than going through
        // `readError`, because a reset here is not a stale socket -- the server
        // has the request -- and must not be resent by `reconnectOnStale`.
        error.ReadFailed => return response.bodyErr() orelse error.ReadFailed,
        else => |e| return e,
    };

    return response.head.status;
}

/// A request attempt's failure on a pooled keep-alive socket the server closed
/// while idle: the peer closed or reset the connection before any response
/// head (`sendReceiveHead` produces exactly these then). The request never
/// reached the server and nothing has been delivered.
fn isStaleConnection(err: anyerror) bool {
    return err == error.HttpConnectionClosing or err == error.ConnectionResetByPeer;
}

/// Run one request attempt, `@call(.auto, attempt, args)` returning `R`, and on a stale
/// pooled socket run it once more on a fresh connection, immediately and
/// whatever the caller's `RetryPolicy`. A stale socket says nothing about the
/// server's health: without this, a long-lived client with
/// `RetryPolicy.disabled` fails the first call after every idle gap, and one
/// with backoff sleeps a second to save a handshake. Once only, because a
/// fresh connection failing the same way is the server, and surfaces. Not
/// after a user interrupt, whose aborted read looks the same.
fn reconnectOnStale(
    comptime R: type,
    interrupt: ?*Interrupt,
    comptime attempt: anytype,
    args: anytype,
) R {
    return @call(.auto, attempt, args) catch |err| {
        if (!isStaleConnection(err)) return err;
        if (interrupt) |it| if (it.isFired()) return err;
        return @call(.auto, attempt, args);
    };
}

const SendReceiveHeadError = std.Io.Writer.Error || std.http.Client.Request.ReceiveHeadError || error{ConnectionResetByPeer};

/// Send `req` with `payload` (bodiless when null) and wait for the response
/// head. std.http reports a failed send or head read only as `WriteFailed` /
/// `ReadFailed`, leaving the cause on the connection; when the cause is the
/// peer having closed or reset the socket it surfaces here as
/// `ConnectionResetByPeer` or `HttpConnectionClosing`, so `isStaleConnection`
/// can see it. Anything else keeps the generic error.
fn sendReceiveHead(
    req: *std.http.Client.Request,
    payload: ?[]const u8,
    redirect_buffer: []u8,
) SendReceiveHeadError!std.http.Client.Response {
    send(req, payload) catch return writeError(req.connection.?);
    return req.receiveHead(redirect_buffer) catch |err| switch (err) {
        error.ReadFailed => readError(req.connection.?),
        error.WriteFailed => writeError(req.connection.?),
        else => |e| e,
    };
}

fn send(req: *std.http.Client.Request, payload: ?[]const u8) std.Io.Writer.Error!void {
    const p = payload orelse return req.sendBodiless();
    req.transfer_encoding = .{ .content_length = p.len };
    var body = try req.sendBodyUnflushed(&.{});
    try body.writer.writeAll(p);
    try body.end();
    try req.connection.?.flush();
}

fn writeError(connection: *std.http.Client.Connection) SendReceiveHeadError {
    return switch (connection.stream_writer.err orelse return error.WriteFailed) {
        // EPIPE: the peer already closed its end.
        error.ConnectionResetByPeer, error.SocketUnconnected => error.ConnectionResetByPeer,
        else => error.WriteFailed,
    };
}

fn readError(connection: *std.http.Client.Connection) SendReceiveHeadError {
    return switch (connection.getReadError() orelse return error.ReadFailed) {
        error.ConnectionResetByPeer => error.ConnectionResetByPeer,
        // A TLS peer that closed without close_notify, which std cannot tell
        // apart from a head cut short; plain TCP reports the former as
        // HttpConnectionClosing, the shape of an idle keep-alive close.
        error.TlsConnectionTruncated => error.HttpConnectionClosing,
        else => error.ReadFailed,
    };
}

/// Error set returned by `streamSse`. Each streaming provider's
/// `StreamError` is a superset (it adds `MissingApiKey`, which callers
/// check before streaming).
pub const SseError = error{
    ApiError,
    InvalidSseData,
} || std.http.Client.RequestError || std.http.Client.Request.ReceiveHeadError ||
    std.Io.Writer.Error || std.Io.Reader.DelimiterError ||
    std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

/// What a `streamLines` framer decides to do with one transport line.
const LineFrame = union(enum) {
    skip,
    stop,
    parse: []const u8,
};

fn frameSse(line: []const u8) LineFrame {
    if (line.len == 0) return .skip;
    // `event:` lines carry only the type, which the data JSON repeats.
    if (std.mem.startsWith(u8, line, "event: ")) return .skip;
    if (!std.mem.startsWith(u8, line, "data: ")) return .skip;
    const json_data = line["data: ".len..];
    // OpenAI terminates with a `[DONE]` sentinel; others just EOF.
    if (std.mem.eql(u8, json_data, "[DONE]")) return .stop;
    return .{ .parse = json_data };
}

fn frameNdjson(line: []const u8) LineFrame {
    if (line.len == 0) return .skip;
    return .{ .parse = line };
}

/// Shared transport for line-delimited streaming POST responses: opens the
/// request, arms the interrupt around the blocking read, checks status, then
/// runs each `\n`-delimited, `\r`-trimmed line through `frame` — skipping,
/// stopping, or parsing it as `EventT` and handing the value to `callback` —
/// until the body ends. `streamSse`/`streamNdjson` differ only in `frame`.
///
/// Requests `identity` encoding: the line reader does not decompress, so a
/// gzip'd body would arrive as unparseable bytes. `error_handler` is the
/// provider client; its `interrupt` field (if present) is armed around the
/// blocking read so a cross-thread `Interrupt.fire` can abort it, and
/// `setErrorDetail(status, "")` records a non-2xx status before returning
/// `error.ApiError` — mirroring `fetchJsonWithRetry`.
fn streamLines(
    allocator: std.mem.Allocator,
    http_client: *std.http.Client,
    url: []const u8,
    extra_headers: []const std.http.Header,
    payload: []const u8,
    comptime EventT: type,
    error_handler: anytype,
    context: anytype,
    callback: *const fn (@TypeOf(context), EventT) void,
    comptime frame: fn ([]const u8) LineFrame,
) SseError!void {
    const interrupt: ?*Interrupt = if (@hasField(@TypeOf(error_handler.*), "interrupt"))
        error_handler.interrupt
    else
        null;
    // A stale socket fails before the head, so no event has reached
    // `callback` yet and the stream can start over.
    return reconnectOnStale(SseError!void, interrupt, streamLinesOnce, .{
        allocator, http_client, url, extra_headers, payload, EventT, interrupt, error_handler, context, callback, frame,
    });
}

fn streamLinesOnce(
    allocator: std.mem.Allocator,
    http_client: *std.http.Client,
    url: []const u8,
    extra_headers: []const std.http.Header,
    payload: []const u8,
    comptime EventT: type,
    interrupt: ?*Interrupt,
    error_handler: anytype,
    context: anytype,
    callback: *const fn (@TypeOf(context), EventT) void,
    comptime frame: fn ([]const u8) LineFrame,
) SseError!void {
    const uri = try std.Uri.parse(url);
    var req = try http_client.request(.POST, uri, .{
        .extra_headers = extra_headers,
        .headers = .{
            .content_type = .{ .override = "application/json" },
            .accept_encoding = .{ .override = "identity" },
        },
        .redirect_behavior = .init(5),
    });
    defer req.deinit();

    // Let a SIGINT abort the blocking read; poison the connection on any
    // failed/aborted exchange so it isn't pooled with unknown framing.
    var guard = armInterrupt(interrupt, &req);
    defer guard.deinit();
    errdefer guard.poison();

    var redirect_buf: [0]u8 = undefined;
    var response = try sendReceiveHead(&req, payload, &redirect_buf);

    const transfer_buf = try allocator.alloc(u8, 256 * 1024);
    defer allocator.free(transfer_buf);
    const reader = response.reader(transfer_buf);

    const status_code: u10 = @intFromEnum(response.head.status);
    if (status_code < 200 or status_code >= 300) {
        // Read the error body so the failure carries a message, not just a status.
        var body: std.Io.Writer.Allocating = .init(allocator);
        defer body.deinit();
        _ = reader.streamRemaining(&body.writer) catch {};
        error_handler.setErrorDetail(status_code, body.written());
        return error.ApiError;
    }

    while (true) {
        const line = reader.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => return error.InvalidSseData,
            error.ReadFailed => {
                guard.poison();
                return;
            },
        } orelse return;

        const json_data = switch (frame(std.mem.trimEnd(u8, line, "\r"))) {
            .skip => continue,
            .stop => return,
            .parse => |d| d,
        };

        const parsed = std.json.parseFromSlice(EventT, allocator, json_data, .{ .ignore_unknown_fields = true }) catch |err| {
            std.log.err("stream: failed to parse chunk: {}", .{err});
            return error.InvalidSseData;
        };
        defer parsed.deinit();
        callback(context, parsed.value);
    }
}

/// POST `payload` and stream the Server-Sent Events response, invoking
/// `callback` with each parsed `data:` event of type `EventT` until the stream
/// ends (an `event:` line or `[DONE]` sentinel).
pub fn streamSse(
    allocator: std.mem.Allocator,
    http_client: *std.http.Client,
    url: []const u8,
    extra_headers: []const std.http.Header,
    payload: []const u8,
    comptime EventT: type,
    error_handler: anytype,
    context: anytype,
    callback: *const fn (@TypeOf(context), EventT) void,
) SseError!void {
    return streamLines(allocator, http_client, url, extra_headers, payload, EventT, error_handler, context, callback, frameSse);
}

/// POST `payload` and stream a newline-delimited JSON (NDJSON) response — one
/// JSON object per line, no `data:` framing or `[DONE]` sentinel, as Ollama's
/// native `/api/chat` emits — invoking `callback` with each parsed line.
pub fn streamNdjson(
    allocator: std.mem.Allocator,
    http_client: *std.http.Client,
    url: []const u8,
    extra_headers: []const std.http.Header,
    payload: []const u8,
    comptime EventT: type,
    error_handler: anytype,
    context: anytype,
    callback: *const fn (@TypeOf(context), EventT) void,
) SseError!void {
    return streamLines(allocator, http_client, url, extra_headers, payload, EventT, error_handler, context, callback, frameNdjson);
}

/// `streamSse` with `body` JSON-encoded by `encodeBody`.
pub fn streamSseValue(
    allocator: std.mem.Allocator,
    http_client: *std.http.Client,
    url: []const u8,
    extra_headers: []const std.http.Header,
    body: anytype,
    comptime EventT: type,
    error_handler: anytype,
    context: anytype,
    callback: *const fn (@TypeOf(context), EventT) void,
) SseError!void {
    const payload = try encodeBody(allocator, body);
    defer allocator.free(payload);
    return streamSse(allocator, http_client, url, extra_headers, payload, EventT, error_handler, context, callback);
}

/// `streamNdjson` with `body` JSON-encoded by `encodeBody`.
pub fn streamNdjsonValue(
    allocator: std.mem.Allocator,
    http_client: *std.http.Client,
    url: []const u8,
    extra_headers: []const std.http.Header,
    body: anytype,
    comptime EventT: type,
    error_handler: anytype,
    context: anytype,
    callback: *const fn (@TypeOf(context), EventT) void,
) SseError!void {
    const payload = try encodeBody(allocator, body);
    defer allocator.free(payload);
    return streamNdjson(allocator, http_client, url, extra_headers, payload, EventT, error_handler, context, callback);
}

/// Extract an owned copy of `error.message` from a provider JSON error body, or
/// null if absent or unparseable. Caller frees the result with `allocator`.
///
/// Parses only `message`: sibling fields vary by provider (e.g. llama.cpp types
/// `code` as an int where OpenAI uses a string), so a full typed parse would
/// fail and cost us the message.
/// The message in a JSON error body: `error.message` (OpenAI, Anthropic,
/// Gemini), or FastAPI's `detail` as a string, as an object's `message`
/// (TypeSafe), or as a validation list's first `msg`.
pub fn extractErrorMessage(allocator: std.mem.Allocator, body: []const u8) ?[]u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const root = parsed.value.object;
    const msg = if (root.get("error")) |err|
        stringField(err, "message")
    else if (root.get("detail")) |detail| switch (detail) {
        .string => |s| s,
        .object => stringField(detail, "message"),
        .array => |list| if (list.items.len > 0) stringField(list.items[0], "msg") else null,
        else => null,
    } else null;
    return allocator.dupe(u8, msg orelse return null) catch null;
}

fn stringField(value: std.json.Value, name: []const u8) ?[]const u8 {
    if (value != .object) return null;
    const field = value.object.get(name) orelse return null;
    return if (field == .string) field.string else null;
}

test "extractErrorMessage reads each provider's error shape" {
    const cases = [_]struct { body: []const u8, want: ?[]const u8 }{
        .{ .body = "{\"error\":{\"message\":\"bad key\",\"type\":\"auth\"}}", .want = "bad key" },
        .{ .body = "{\"detail\":{\"error_type\":\"authentication_error\",\"message\":\"Cannot authenticate\"}}", .want = "Cannot authenticate" },
        .{ .body = "{\"detail\":[{\"type\":\"missing\",\"loc\":[\"body\",\"model\"],\"msg\":\"Field required\"}]}", .want = "Field required" },
        .{ .body = "{\"detail\":\"Not Found\"}", .want = "Not Found" },
        .{ .body = "{\"detail\":[]}", .want = null },
        .{ .body = "{\"error\":\"plain\"}", .want = null },
        .{ .body = "<html>502</html>", .want = null },
    };
    for (cases) |case| {
        const got = extractErrorMessage(std.testing.allocator, case.body);
        defer if (got) |g| std.testing.allocator.free(g);
        if (case.want) |want| {
            try std.testing.expectEqualStrings(want, got orelse return error.TestExpectedMessage);
        } else {
            try std.testing.expect(got == null);
        }
    }
}

test "ErrorDetail.clone outlives the original" {
    var detail: ErrorDetail = .{};
    detail.set(std.testing.allocator, 401, "{\"detail\":{\"message\":\"nope\"}}");
    const copy = try detail.clone(std.testing.allocator);
    defer std.testing.allocator.free(copy.message.?);
    detail.deinit(std.testing.allocator);
    try std.testing.expectEqual(401, copy.status);
    try std.testing.expectEqualStrings("nope", copy.message.?);
}

/// Owns the parsed response and its backing memory.
/// Call `deinit()` when done to free all resources.
pub fn Response(comptime T: type) type {
    return struct {
        value: T,
        json_buf: std.Io.Writer.Allocating,
        parsed: std.json.Parsed(T),

        pub fn deinit(self: *@This()) void {
            self.parsed.deinit();
            self.json_buf.deinit();
        }
    };
}

/// Pagination options for list operations.
pub const ListOptions = struct {
    /// Maximum number of items to return per page.
    pageSize: ?i32 = null,
    /// Token from a previous response's `nextPageToken` to fetch the next page.
    pageToken: ?[]const u8 = null,
};

pub fn appendListParams(allocator: std.mem.Allocator, base_url: []const u8, options: ListOptions) ![]u8 {
    if (options.pageSize) |ps| {
        if (options.pageToken) |pt| {
            return std.fmt.allocPrint(allocator, "{s}?pageSize={d}&pageToken={s}", .{ base_url, ps, pt });
        }
        return std.fmt.allocPrint(allocator, "{s}?pageSize={d}", .{ base_url, ps });
    }
    if (options.pageToken) |pt| {
        return std.fmt.allocPrint(allocator, "{s}?pageToken={s}", .{ base_url, pt });
    }
    return allocator.dupe(u8, base_url);
}

/// A loopback HTTP server socket, its URL, and a client to reach it.
const Loopback = struct {
    server: std.Io.net.Server,
    url: []u8,
    client: std.http.Client,

    fn init() !Loopback {
        const loopback: std.Io.net.IpAddress = try .parse("127.0.0.1", 0);
        var server = try loopback.listen(std.testing.io, .{});
        errdefer server.deinit(std.testing.io);
        return .{
            .server = server,
            .url = try std.fmt.allocPrint(std.testing.allocator, "http://127.0.0.1:{d}/", .{server.socket.address.getPort()}),
            .client = .{ .allocator = std.testing.allocator, .io = std.testing.io },
        };
    }

    fn deinit(self: *Loopback) void {
        self.client.deinit();
        std.testing.allocator.free(self.url);
        self.server.deinit(std.testing.io);
    }

    const Ok = struct { ok: bool };
    const NoDetail = struct {
        fn setErrorDetail(_: *NoDetail, _: u10, _: []const u8) void {}
    };

    /// Serve one connection with `serveOnce(serve)` on a thread while `run`
    /// makes one call. A failed call may never have reached `accept`, so on
    /// error connect once to let the thread return: the join cannot hang.
    fn served(self: *Loopback, serve: Serve, comptime run: anytype, args: anytype) @typeInfo(@TypeOf(run)).@"fn".return_type.? {
        const thread = std.Thread.spawn(.{}, serveOnce, .{ &self.server, serve }) catch unreachable;
        defer thread.join();
        errdefer if (self.server.socket.address.connect(std.testing.io, .{ .mode = .stream })) |s| s.close(std.testing.io) else |_| {};
        return @call(.auto, run, args);
    }

    /// One JSON fetch with retries disabled. The timeout turns a reconnect
    /// that wrongly waits on a connection nobody accepts into an error instead
    /// of a hang.
    fn fetch(self: *Loopback, serve: Serve) FetchError!Response(Ok) {
        return self.served(serve, fetchOk, .{self});
    }

    fn fetchOk(self: *Loopback) FetchError!Response(Ok) {
        var detail: NoDetail = .{};
        return fetchJsonWithRetry(std.testing.allocator, &self.client, .disabled, 2000, .{
            .location = .{ .url = self.url },
        }, Ok, &detail);
    }

    /// One NDJSON stream; returns how many events reached the callback.
    fn stream(self: *Loopback, serve: Serve) SseError!u32 {
        return self.served(serve, streamOk, .{self});
    }

    fn streamOk(self: *Loopback) SseError!u32 {
        var detail: NoDetail = .{};
        var events: u32 = 0;
        try streamNdjson(std.testing.allocator, &self.client, self.url, &.{}, "{}", Ok, &detail, &events, countEvent);
        return events;
    }

    fn countEvent(events: *u32, _: Ok) void {
        events.* += 1;
    }
};

/// What `serveOnce` does with the one connection it accepts.
const Serve = enum {
    /// Answer one request with keep-alive, then close: the client is left
    /// holding a pooled socket the server has already closed.
    respond,
    /// Close without answering.
    hang_up,
    /// Send a head and part of the body, then close with the request still
    /// unread, which makes the kernel reset the connection mid-body.
    reset_mid_body,
};

/// Test server half for the loopback tests: accept one connection and handle
/// it as `serve` says.
fn serveOnce(server: *std.Io.net.Server, serve: Serve) void {
    const io = std.testing.io;
    const stream = server.accept(io) catch return;
    defer stream.close(io);
    var write_buf: [256]u8 = undefined;
    var writer = stream.writer(io, &write_buf);
    switch (serve) {
        .hang_up => {},
        .reset_mid_body => {
            writer.interface.writeAll("HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{") catch return;
            writer.interface.flush() catch return;
            // Let the client take the head before the reset discards it.
            io.sleep(.fromMilliseconds(50), .awake) catch {};
        },
        .respond => {
            var read_buf: [4096]u8 = undefined;
            var reader = stream.reader(io, &read_buf);
            var http_server: std.http.Server = .init(&reader.interface, &writer.interface);
            var request = http_server.receiveHead() catch return;
            request.respond("{\"ok\":true}\n", .{ .keep_alive = true }) catch return;
        },
    }
}

test "watchdog turns a stalled response into error.Timeout" {
    var t: Loopback = try .init();
    defer t.deinit();

    // Never accept: the kernel completes the handshake into the backlog and
    // buffers the request, so the client blocks in receiveHead until the
    // watchdog shuts the socket down.

    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    var retry_after_ms: ?u32 = null;
    try std.testing.expectError(
        error.Timeout,
        fetchCapturingRetryAfter(std.testing.allocator, &t.client, .{ .location = .{ .url = t.url } }, &out.writer, null, 50, &retry_after_ms),
    );
}

test "a pooled socket the server closed while idle is reconnected despite RetryPolicy.disabled" {
    var t: Loopback = try .init();
    defer t.deinit();

    // The first call pools its socket; the server then closes it. The second
    // call picks the dead socket from the pool, and must reconnect to the next
    // `serveOnce` instead of failing with HttpConnectionClosing.
    var first = try t.fetch(.respond);
    first.deinit();
    var second = try t.fetch(.respond);
    defer second.deinit();
    try std.testing.expect(second.value.ok);
}

test "the stale-socket reconnect happens once, not again on a fresh connection" {
    var t: Loopback = try .init();
    defer t.deinit();

    var first = try t.fetch(.respond);
    first.deinit();

    // The pooled socket is dead and the reconnect's fresh connection is closed
    // unanswered too: that is the server, not a stale pool, so it surfaces.
    // A second reconnect would sit in the listen backlog until the timeout.
    try std.testing.expectError(error.HttpConnectionClosing, t.fetch(.hang_up));
}

test "a stream on a pooled socket the server closed while idle is reconnected" {
    var t: Loopback = try .init();
    defer t.deinit();

    try std.testing.expectEqual(1, try t.stream(.respond));
    try std.testing.expectEqual(1, try t.stream(.respond));
}

test "a connection reset mid-body fails the call instead of panicking, and is not resent" {
    var t: Loopback = try .init();
    defer t.deinit();

    // A reconnect would wait on a connection nobody accepts until the timeout.
    try std.testing.expectError(error.ReadFailed, t.fetch(.reset_mid_body));
}
