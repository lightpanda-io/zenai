const std = @import("std");
const retry = @import("retry.zig");

/// Error set returned by `fetchJsonWithRetry`. Each provider's `ApiError`
/// is a superset (it adds `MissingApiKey`).
pub const FetchError = error{
    ApiError,
    EmptyResponse,
} || std.http.Client.FetchError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

/// Cross-thread trigger for aborting an in-flight HTTP request. A request path
/// arms it with the active socket fd around the blocking read (see `armInterrupt`);
/// another thread (the SIGINT handler) calls `fire` to `shutdown` that socket,
/// which unblocks the read so the request returns an error instead of waiting
/// for the server. The errored connection is dropped from the pool by `Request.deinit`.
///
/// `fire` is sticky: one landing before the socket is armed (during connect/TLS)
/// is honored by `arm` rather than lost. `reset` clears it per turn.
pub const Interrupt = struct {
    fd: std.atomic.Value(std.posix.socket_t) = .init(-1),
    fired: std.atomic.Value(bool) = .init(false),

    fn arm(self: *Interrupt, fd: std.posix.socket_t) void {
        self.fd.store(fd, .release);
        if (self.fired.load(.acquire)) std.posix.shutdown(fd, .both) catch {};
    }

    fn disarm(self: *Interrupt) void {
        self.fd.store(-1, .release);
    }

    pub fn fire(self: *Interrupt) void {
        self.fired.store(true, .release);
        const fd = self.fd.load(.acquire);
        if (fd >= 0) std.posix.shutdown(fd, .both) catch {};
    }

    pub fn isFired(self: *const Interrupt) bool {
        return self.fired.load(.acquire);
    }

    /// Clear armed + fired state so the interrupt is reusable next turn.
    pub fn reset(self: *Interrupt) void {
        self.fired.store(false, .release);
        self.fd.store(-1, .release);
    }
};

/// Arms an optional `Interrupt` with `req`'s socket fd so a cross-thread
/// `Interrupt.fire` can abort a blocking read on that connection. The guard
/// must outlive the read but be torn down before `req.deinit` (LIFO) so a late
/// `fire` can't `shutdown` a fd the pool has already recycled:
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
        if (req.connection) |conn| it.arm(conn.stream_reader.getStream().handle);
    }
    return .{ .interrupt = interrupt, .req = req };
}

/// Common HTTP fetch + retry + JSON parse pipeline shared by all provider
/// clients. On a non-retryable HTTP error response, calls
/// `error_handler.setErrorDetail(status, body)` so the caller can record
/// provider-specific error detail before this function returns
/// `error.ApiError`.
pub fn fetchJsonWithRetry(
    allocator: std.mem.Allocator,
    http_client: *std.http.Client,
    policy: retry.RetryPolicy,
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

        const status = fetchInterruptible(allocator, http_client, options, &response_buf.writer, interrupt) catch |err| {
            // Don't retry a request the user cancelled.
            if (interrupt) |it| if (it.isFired()) return err;
            if (retry.isRetryableFetchError(err) and attempt + 1 < policy.max_attempts) {
                retry.sleepMs(retry.backoffMs(attempt, policy));
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
            retry.sleepMs(retry.backoffMs(attempt, policy));
            continue;
        }
        error_handler.setErrorDetail(status_code, body);
        return error.ApiError;
    }
}

/// A single HTTP round-trip into `response_writer`, mirroring
/// `std.http.Client.fetch` (redirects, content-encoding) but arming
/// `interrupt` with the connection's socket fd around the blocking read so a
/// SIGINT on another thread can abort it. Returns the response status.
fn fetchInterruptible(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    options: std.http.Client.FetchOptions,
    response_writer: *std.Io.Writer,
    interrupt: ?*Interrupt,
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

    // Arm the interrupt with this connection's fd around the send/receive, and
    // poison the connection on any failed exchange (including an interrupt-
    // induced read error) so `req.deinit` drops the socket instead of pooling
    // one with unknown framing.
    var guard = armInterrupt(interrupt, &req);
    defer guard.deinit();
    errdefer guard.poison();

    if (options.payload) |payload| {
        req.transfer_encoding = .{ .content_length = payload.len };
        var body = try req.sendBodyUnflushed(&.{});
        try body.writer.writeAll(payload);
        try body.end();
        try req.connection.?.flush();
    } else {
        try req.sendBodiless();
    }

    const own_redirect_buffer = redirect_behavior != .unhandled and options.redirect_buffer == null;
    const redirect_buffer: []u8 = if (redirect_behavior == .unhandled) &.{} else options.redirect_buffer orelse try allocator.alloc(u8, 8 * 1024);
    defer if (own_redirect_buffer) allocator.free(redirect_buffer);

    var response = try req.receiveHead(redirect_buffer);

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
        error.ReadFailed => return response.bodyErr().?,
        else => |e| return e,
    };

    return response.head.status;
}

/// Error set returned by `streamSse`. Each provider's `StreamError` is a
/// superset (it adds `MissingApiKey`, which callers check before streaming).
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

    req.transfer_encoding = .{ .content_length = payload.len };
    var bw = try req.sendBodyUnflushed(&.{});
    try bw.writer.writeAll(payload);
    try bw.end();
    try req.connection.?.flush();

    var redirect_buf: [0]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);

    const status_code: u10 = @intFromEnum(response.head.status);
    if (status_code < 200 or status_code >= 300) {
        error_handler.setErrorDetail(status_code, "");
        return error.ApiError;
    }

    const transfer_buf = try allocator.alloc(u8, 256 * 1024);
    defer allocator.free(transfer_buf);
    const reader = response.reader(transfer_buf);

    while (true) {
        const line = reader.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => return error.InvalidSseData,
            error.ReadFailed => {
                guard.poison();
                return;
            },
        } orelse return;

        const json_data = switch (frame(std.mem.trimRight(u8, line, "\r"))) {
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

/// Extract an owned copy of `error.message` from a provider JSON error body, or
/// null if absent or unparseable. Caller frees the result with `allocator`.
///
/// Parses only `message`: sibling fields vary by provider (e.g. llama.cpp types
/// `code` as an int where OpenAI uses a string), so a full typed parse would
/// fail and cost us the message.
pub fn extractErrorMessage(allocator: std.mem.Allocator, body: []const u8) ?[]u8 {
    const Body = struct { @"error": ?struct { message: ?[]const u8 = null } = null };
    const parsed = std.json.parseFromSlice(Body, allocator, body, .{ .ignore_unknown_fields = true }) catch return null;
    defer parsed.deinit();
    const err = parsed.value.@"error" orelse return null;
    const msg = err.message orelse return null;
    return allocator.dupe(u8, msg) catch null;
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
    if (options.pageSize == null and options.pageToken == null) {
        return allocator.dupe(u8, base_url);
    }
    if (options.pageSize != null and options.pageToken != null) {
        return std.fmt.allocPrint(allocator, "{s}?pageSize={d}&pageToken={s}", .{ base_url, options.pageSize.?, options.pageToken.? });
    }
    if (options.pageSize) |ps| {
        return std.fmt.allocPrint(allocator, "{s}?pageSize={d}", .{ base_url, ps });
    }
    return std.fmt.allocPrint(allocator, "{s}?pageToken={s}", .{ base_url, options.pageToken.? });
}
