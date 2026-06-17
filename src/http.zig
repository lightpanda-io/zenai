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
pub const Interrupt = struct {
    fd: std.atomic.Value(std.posix.socket_t) = .init(-1),

    fn arm(self: *Interrupt, fd: std.posix.socket_t) void {
        self.fd.store(fd, .release);
    }

    fn disarm(self: *Interrupt) void {
        self.fd.store(-1, .release);
    }

    pub fn fire(self: *Interrupt) void {
        const fd = self.fd.load(.acquire);
        if (fd >= 0) std.posix.shutdown(fd, .both) catch {};
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
