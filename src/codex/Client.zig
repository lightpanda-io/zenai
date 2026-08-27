const std = @import("std");
const types = @import("../openai/types.zig");
const openai_mod = @import("../openai/Client.zig");
const http = @import("../http.zig");
const retry = @import("../retry.zig");

pub const RetryPolicy = retry.RetryPolicy;

const ResponsesRequest = types.ResponsesRequest;
const ResponseStreamEvent = types.ResponseStreamEvent;

/// Client for the OpenAI "Codex" backend — the ChatGPT-subscription endpoint at
/// `chatgpt.com/backend-api/codex`. It speaks the same Responses API as the
/// `openai` client (shared request/response types), but authenticates with an
/// OAuth subscription token plus a ChatGPT account id rather than an API key.
const Client = @This();

allocator: std.mem.Allocator,
/// OAuth access token (subscription bearer), not an API key.
access_token: []const u8,
base_url: []const u8,
/// ChatGPT account id from the OAuth JWT; sent as `ChatGPT-Account-Id`.
/// Owned copy — a borrowed pointer would dangle once token refreshes free it.
account_id: ?[]const u8,
/// Client identifier sent as `originator`.
originator: []const u8,
user_agent: []const u8,
/// Sent as `session-id`; null omits it.
session_id: ?[]const u8,
http_client: std.http.Client,
last_error_message: ?[]u8 = null,
last_error_status: ?u10 = null,
interrupt: ?*http.Interrupt = null,
/// Cached "Bearer {token}" header value, built on first request.
authorization: ?[]const u8 = null,

pub const InitOptions = struct {
    base_url: []const u8 = "https://chatgpt.com/backend-api/codex",
    account_id: ?[]const u8 = null,
    session_id: ?[]const u8 = null,
    originator: []const u8 = "lightpanda",
    user_agent: []const u8 = "lightpanda",
    /// Accepted for uniform provider init; the streaming-only path never
    /// retries and is not bounded by `request_timeout_ms`.
    retry_policy: RetryPolicy = .{},
    request_timeout_ms: ?u32 = null,
};

pub fn init(io: std.Io, allocator: std.mem.Allocator, access_token: []const u8, options: InitOptions) !Client {
    return .{
        .allocator = allocator,
        .access_token = access_token,
        .base_url = options.base_url,
        .account_id = if (options.account_id) |a| try allocator.dupe(u8, a) else null,
        .session_id = options.session_id,
        .originator = options.originator,
        .user_agent = options.user_agent,
        .http_client = .{ .allocator = allocator, .io = io },
    };
}

pub fn deinit(self: *Client) void {
    if (self.account_id) |a| self.allocator.free(a);
    if (self.authorization) |a| self.allocator.free(a);
    if (self.last_error_message) |m| self.allocator.free(m);
    self.http_client.deinit();
}

/// Repoint at a refreshed OAuth access token without tearing down connections.
/// The caller keeps ownership of `token` and frees the previous buffer only
/// after this returns.
pub fn setApiKey(self: *Client, token: []const u8) void {
    self.access_token = token;
    if (self.authorization) |a| {
        self.allocator.free(a);
        self.authorization = null;
    }
}

// Same wire protocol and transport as the openai client, so the same failures.
pub const ApiError = openai_mod.ApiError;
pub const StreamError = openai_mod.StreamError;

fn authHeaders(self: *Client, buf: *[5]std.http.Header) ![]const std.http.Header {
    if (self.authorization == null)
        self.authorization = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.access_token});
    var n: usize = 0;
    buf[n] = .{ .name = "authorization", .value = self.authorization.? };
    n += 1;
    buf[n] = .{ .name = "originator", .value = self.originator };
    n += 1;
    buf[n] = .{ .name = "User-Agent", .value = self.user_agent };
    n += 1;
    if (self.account_id) |a| {
        buf[n] = .{ .name = "ChatGPT-Account-Id", .value = a };
        n += 1;
    }
    if (self.session_id) |s| {
        buf[n] = .{ .name = "session-id", .value = s };
        n += 1;
    }
    return buf[0..n];
}

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error_status = status_code;
    if (self.last_error_message) |m| {
        self.allocator.free(m);
        self.last_error_message = null;
    }
    if (body.len > 0) {
        std.log.err("Codex API error (HTTP {d}): {s}", .{ status_code, body });
        self.last_error_message = http.extractErrorMessage(self.allocator, body);
    }
}

/// Stream a model response via the Responses API; `request.stream` is forced on
/// (the backend rejects non-streaming requests outright).
pub fn createResponseStream(
    self: *Client,
    request: ResponsesRequest,
    context: anytype,
    callback: *const fn (@TypeOf(context), ResponseStreamEvent) void,
) StreamError!void {
    if (self.access_token.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/responses", .{self.base_url});
    defer self.allocator.free(url);

    var req_body = request;
    req_body.stream = true;

    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(req_body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    var hdr_buf: [5]std.http.Header = undefined;
    const auth = try self.authHeaders(&hdr_buf);
    return http.streamSse(self.allocator, &self.http_client, url, auth, payload_buf.written(), ResponseStreamEvent, self, context, callback);
}

fn headerValue(headers: []const std.http.Header, name: []const u8) ?[]const u8 {
    for (headers) |h| if (std.mem.eql(u8, h.name, name)) return h.value;
    return null;
}

test "authHeaders: bearer + account-id + originator + session-id, no api-key" {
    var client = try Client.init(std.testing.io, std.testing.allocator, "tok-abc", .{
        .account_id = "acct-1",
        .session_id = "sess-1",
    });
    defer client.deinit();
    var buf: [5]std.http.Header = undefined;
    const headers = try client.authHeaders(&buf);
    try std.testing.expectEqualStrings("Bearer tok-abc", headerValue(headers, "authorization").?);
    try std.testing.expectEqualStrings("acct-1", headerValue(headers, "ChatGPT-Account-Id").?);
    try std.testing.expectEqualStrings("sess-1", headerValue(headers, "session-id").?);
    try std.testing.expectEqualStrings("lightpanda", headerValue(headers, "originator").?);
    try std.testing.expectEqual(@as(?[]const u8, null), headerValue(headers, "x-api-key"));
}

test "authHeaders: account-id and session-id omitted when null" {
    var client = try Client.init(std.testing.io, std.testing.allocator, "tok", .{});
    defer client.deinit();
    var buf: [5]std.http.Header = undefined;
    const headers = try client.authHeaders(&buf);
    try std.testing.expectEqual(@as(usize, 3), headers.len);
    try std.testing.expectEqual(@as(?[]const u8, null), headerValue(headers, "ChatGPT-Account-Id"));
    try std.testing.expectEqual(@as(?[]const u8, null), headerValue(headers, "session-id"));
}

test "setApiKey rebuilds the cached bearer value" {
    var client = try Client.init(std.testing.io, std.testing.allocator, "tok-old", .{});
    defer client.deinit();
    var buf: [5]std.http.Header = undefined;
    try std.testing.expectEqualStrings("Bearer tok-old", headerValue(try client.authHeaders(&buf), "authorization").?);
    client.setApiKey("tok-new");
    try std.testing.expectEqual(@as(?[]const u8, null), client.authorization);
    try std.testing.expectEqualStrings("Bearer tok-new", headerValue(try client.authHeaders(&buf), "authorization").?);
}

test "Codex ResponsesRequest serializes store/include/reasoning.summary, omits max_output_tokens" {
    const input = [_]types.ResponseInputItem{.{ .type = "message", .role = "user", .content = .{ .text = "hi" } }};
    const req: ResponsesRequest = .{
        .model = "gpt-5-codex",
        .input = &input,
        .reasoning = .{ .effort = .medium, .summary = "auto" },
        .store = false,
        .include = &.{"reasoning.encrypted_content"},
    };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(req, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.find(u8, json, "\"store\":false") != null);
    try std.testing.expect(std.mem.find(u8, json, "\"reasoning.encrypted_content\"") != null);
    try std.testing.expect(std.mem.find(u8, json, "\"summary\":\"auto\"") != null);
    try std.testing.expect(std.mem.find(u8, json, "max_output_tokens") == null);
}
