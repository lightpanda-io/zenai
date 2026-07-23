const std = @import("std");
const types = @import("types.zig");
const http = @import("../http.zig");
const retry = @import("../retry.zig");

pub const RetryPolicy = retry.RetryPolicy;

const MessageParam = types.MessageParam;
const ContentBlockParam = types.ContentBlockParam;
const MessageRequest = types.MessageRequest;
const MessageResponse = types.MessageResponse;
const StreamEvent = types.StreamEvent;

/// Anthropic Messages API client. Provides access to message creation,
/// streaming, and tool use.
const Client = @This();

allocator: std.mem.Allocator,
api_key: []const u8,
base_url: []const u8,
api_version: []const u8,
/// How `api_key` is presented: an `x-api-key` API key, or an OAuth access token
/// via `authorization: Bearer` (Claude Pro/Max subscription).
auth: Auth,
/// Cached "Bearer {token}" value for `.bearer` auth. Built on first request;
/// owned, freed in `deinit` and cleared by `setApiKey`.
authorization: ?[]u8 = null,
http_client: std.http.Client,
/// Retry policy applied to every non-streaming request.
retry_policy: RetryPolicy,
/// Human-readable message from the most recent API error, owned by the client
/// and freed on the next failure or `deinit`. Set on `error.ApiError`.
last_error_message: ?[]u8 = null,
last_error_status: ?u10 = null,
/// Set by the host so a SIGINT can abort an in-flight request mid-read.
interrupt: ?*http.Interrupt = null,

/// How the client presents `api_key`.
pub const Auth = enum {
    /// Standard API key sent as `x-api-key`.
    api_key,
    /// OAuth access token (Claude subscription) sent as `authorization: Bearer`,
    /// with the `anthropic-beta: oauth-2025-04-20` header and the required first
    /// system block (see `oauth_system_prompt`).
    bearer,
};

/// `anthropic-beta` value required on OAuth (subscription) requests.
pub const oauth_beta = "oauth-2025-04-20";

/// The OAuth (subscription) endpoint requires this exact sentence as the first
/// system block, verbatim. Callers building a `.bearer` request must place it
/// ahead of any other system text.
pub const oauth_system_prompt = "You are Claude Code, Anthropic's official CLI for Claude.";

/// Options for customizing the API endpoint.
pub const InitOptions = struct {
    /// Base URL for the Anthropic API.
    base_url: []const u8 = "https://api.anthropic.com/v1",
    /// API version header value.
    api_version: []const u8 = "2023-06-01",
    /// Retry policy for transient HTTP failures (5xx including 529
    /// `overloaded_error`, 429, and known flaky network errors). Pass
    /// `RetryPolicy.disabled` to opt out.
    retry_policy: RetryPolicy = .{},
    /// Whether `api_key` is an API key or an OAuth bearer token.
    auth: Auth = .api_key,
};

/// Create a new Anthropic API client.
pub fn init(io: std.Io, allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .api_version = options.api_version,
        .auth = options.auth,
        .http_client = .{ .allocator = allocator, .io = io },
        .retry_policy = options.retry_policy,
        .last_error_message = null,
        .last_error_status = null,
    };
}

/// Release all resources held by the client, including HTTP connections.
pub fn deinit(self: *Client) void {
    if (self.authorization) |a| self.allocator.free(a);
    if (self.last_error_message) |m| self.allocator.free(m);
    self.http_client.deinit();
}

/// Repoint the client at a new key/token (e.g. a refreshed OAuth access token)
/// without tearing down connections. The caller retains ownership of `key` and
/// must keep the previous buffer alive until this returns.
pub fn setApiKey(self: *Client, key: []const u8) void {
    self.api_key = key;
    if (self.authorization) |a| {
        self.allocator.free(a);
        self.authorization = null;
    }
}

pub const Response = http.Response;

pub const ApiError = error{
    ApiError,
    MissingApiKey,
    EmptyResponse,
} || std.http.Client.FetchError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

// --- Internal helpers ---

fn authHeaders(self: *Client, buf: *[3]std.http.Header) ![]const std.http.Header {
    buf[0] = .{ .name = "anthropic-version", .value = self.api_version };
    switch (self.auth) {
        .api_key => {
            buf[1] = .{ .name = "x-api-key", .value = self.api_key };
            return buf[0..2];
        },
        .bearer => {
            if (self.authorization == null)
                self.authorization = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_key});
            buf[1] = .{ .name = "authorization", .value = self.authorization.? };
            buf[2] = .{ .name = "anthropic-beta", .value = oauth_beta };
            return buf[0..3];
        },
    }
}

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error_status = status_code;
    if (self.last_error_message) |m| {
        self.allocator.free(m);
        self.last_error_message = null;
    }
    if (body.len > 0) {
        std.log.err("Anthropic API error (HTTP {d}): {s}", .{ status_code, body });
        self.last_error_message = http.extractErrorMessage(self.allocator, body);
    }
}

fn fetchGet(self: *Client, url: []const u8, comptime T: type) ApiError!Response(T) {
    var hdr_buf: [3]std.http.Header = undefined;
    const auth = try self.authHeaders(&hdr_buf);
    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, .{
        .location = .{ .url = url },
        .method = .GET,
        .extra_headers = auth,
    }, T, self);
}

fn fetchPost(self: *Client, url: []const u8, body: anytype, comptime T: type) ApiError!Response(T) {
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    var hdr_buf: [3]std.http.Header = undefined;
    const auth = try self.authHeaders(&hdr_buf);
    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, .{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload_buf.written(),
        .extra_headers = auth,
        .headers = .{ .content_type = .{ .override = "application/json" } },
    }, T, self);
}

// --- Message Creation ---

/// Configuration for message requests.
pub const MessageConfig = struct {
    /// System prompt (separate from messages).
    system: ?[]const types.TextBlock = null,
    /// Sampling temperature (0.0-1.0).
    temperature: ?f32 = null,
    /// Top-p (nucleus) sampling.
    top_p: ?f32 = null,
    /// Top-k sampling.
    top_k: ?i32 = null,
    /// Custom stop sequences.
    stop_sequences: ?[]const []const u8 = null,
    /// Tools the model may use.
    tools: ?[]const types.Tool = null,
    /// How the model should use tools.
    tool_choice: ?types.ToolChoice = null,
    /// Extended thinking configuration.
    thinking: ?types.ThinkingConfig = null,
    /// Output-level controls.
    output_config: ?types.OutputConfig = null,
};

/// Create a message.
pub fn createMessage(
    self: *Client,
    model: []const u8,
    messages: []const MessageParam,
    max_tokens: i32,
    config: MessageConfig,
) ApiError!Response(MessageResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/messages", .{self.base_url});
    defer self.allocator.free(url);

    return self.fetchPost(url, MessageRequest{
        .model = model,
        .messages = messages,
        .max_tokens = max_tokens,
        .system = config.system,
        .temperature = config.temperature,
        .top_p = config.top_p,
        .top_k = config.top_k,
        .stop_sequences = config.stop_sequences,
        .tools = config.tools,
        .tool_choice = config.tool_choice,
        .thinking = config.thinking,
        .output_config = config.output_config,
        // Apply an ephemeral cache_control marker to the last cacheable block
        // (default 5-minute TTL). For agent loops this caches the system
        // prompt + tool definitions across turns, dropping repeated-prefix
        // input cost to ~$0.30/M (10× discount). Setting this top-level field
        // is server-side automatic and harmless when the prefix is short.
        .cache_control = .{},
    }, MessageResponse);
}

/// Convenience: create a message from a single text prompt.
pub fn createMessageFromText(
    self: *Client,
    model: []const u8,
    prompt: []const u8,
    max_tokens: i32,
    config: MessageConfig,
) ApiError!Response(MessageResponse) {
    const content = [_]ContentBlockParam{.{ .text = prompt }};
    const messages = [_]MessageParam{.{ .role = .user, .content = &content }};
    return self.createMessage(model, &messages, max_tokens, config);
}

// --- Streaming ---

pub const StreamError = error{
    ApiError,
    MissingApiKey,
    InvalidSseData,
} || std.http.Client.RequestError || std.http.Client.Request.ReceiveHeadError || std.Io.Writer.Error || std.Io.Reader.DelimiterError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

/// Stream a message creation via Server-Sent Events.
/// The `callback` is invoked for each event as it arrives.
/// Anthropic SSE uses `event:` lines followed by `data:` lines.
pub fn createMessageStream(
    self: *Client,
    model: []const u8,
    messages: []const MessageParam,
    max_tokens: i32,
    config: MessageConfig,
    context: anytype,
    callback: *const fn (@TypeOf(context), StreamEvent) void,
) StreamError!void {
    if (self.api_key.len == 0) return error.MissingApiKey;

    const url = try std.fmt.allocPrint(self.allocator, "{s}/messages", .{self.base_url});
    defer self.allocator.free(url);

    const req_body = MessageRequest{
        .model = model,
        .messages = messages,
        .max_tokens = max_tokens,
        .stream = true,
        .system = config.system,
        .temperature = config.temperature,
        .top_p = config.top_p,
        .top_k = config.top_k,
        .stop_sequences = config.stop_sequences,
        .tools = config.tools,
        .tool_choice = config.tool_choice,
        .thinking = config.thinking,
        .output_config = config.output_config,
        // See createMessage above for cache_control rationale.
        .cache_control = .{},
    };
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(req_body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;
    var hdr_buf: [3]std.http.Header = undefined;
    const auth = try self.authHeaders(&hdr_buf);
    return http.streamSse(self.allocator, &self.http_client, url, auth, payload_buf.written(), StreamEvent, self, context, callback);
}

/// Convenience: stream a message from a single text prompt.
pub fn createMessageStreamFromText(
    self: *Client,
    model: []const u8,
    prompt: []const u8,
    max_tokens: i32,
    config: MessageConfig,
    context: anytype,
    callback: *const fn (@TypeOf(context), StreamEvent) void,
) StreamError!void {
    const content = [_]ContentBlockParam{.{ .text = prompt }};
    const messages = [_]MessageParam{.{ .role = .user, .content = &content }};
    return self.createMessageStream(model, &messages, max_tokens, config, context, callback);
}

/// Reassembles a streamed Messages response into the same `MessageResponse` the
/// non-streaming `createMessage` returns, firing `on_text_fn` for each text
/// delta as it arrives. Drive it by passing `&acc` and `onEvent` to
/// `createMessageStream`, then call `response()`. All storage lives in `arena`.
pub const StreamAccumulator = struct {
    arena: std.mem.Allocator,
    on_text_ctx: *anyopaque,
    on_text_fn: *const fn (*anyopaque, []const u8) void,
    text: std.ArrayListUnmanaged(u8) = .empty,
    tool_blocks: std.ArrayListUnmanaged(ToolBlock) = .empty,
    stop_reason: ?types.StopReason = null,
    usage: types.Usage = .{},
    err: ?error{OutOfMemory} = null,

    const ToolBlock = struct {
        index: i32,
        id: []const u8,
        name: []const u8,
        json: std.ArrayListUnmanaged(u8) = .empty,
    };

    pub fn init(
        arena: std.mem.Allocator,
        on_text_ctx: *anyopaque,
        on_text_fn: *const fn (*anyopaque, []const u8) void,
    ) StreamAccumulator {
        return .{ .arena = arena, .on_text_ctx = on_text_ctx, .on_text_fn = on_text_fn };
    }

    pub fn onEvent(self: *StreamAccumulator, event: StreamEvent) void {
        self.handle(event) catch |e| {
            self.err = e;
        };
    }

    fn handle(self: *StreamAccumulator, event: StreamEvent) error{OutOfMemory}!void {
        const et = event.type orelse return;
        if (std.mem.eql(u8, et, "content_block_start")) {
            const cb = event.content_block orelse return;
            const ct = cb.type orelse return;
            if (std.mem.eql(u8, ct, "tool_use")) {
                try self.tool_blocks.append(self.arena, .{
                    .index = event.index orelse 0,
                    .id = if (cb.id) |id| try self.arena.dupe(u8, id) else "",
                    .name = if (cb.name) |n| try self.arena.dupe(u8, n) else "",
                });
            }
        } else if (std.mem.eql(u8, et, "content_block_delta")) {
            const d = event.delta orelse return;
            const dt = d.type orelse return;
            if (std.mem.eql(u8, dt, "text_delta")) {
                if (d.text) |txt| {
                    try self.text.appendSlice(self.arena, txt);
                    self.on_text_fn(self.on_text_ctx, txt);
                }
            } else if (std.mem.eql(u8, dt, "input_json_delta")) {
                if (d.partial_json) |pj| {
                    const idx = event.index orelse return;
                    for (self.tool_blocks.items) |*b| {
                        if (b.index == idx) {
                            try b.json.appendSlice(self.arena, pj);
                            break;
                        }
                    }
                }
            }
        } else if (std.mem.eql(u8, et, "message_start")) {
            if (event.message) |m| {
                if (m.usage) |u| self.usage = u;
            }
        } else if (std.mem.eql(u8, et, "message_delta")) {
            if (event.delta) |d| {
                if (d.stop_reason) |sr| self.stop_reason = sr;
            }
            // message_delta reports the cumulative output count; keep the
            // input/cache counts from message_start.
            if (event.usage) |u| {
                if (u.output_tokens) |ot| self.usage.output_tokens = ot;
            }
        }
    }

    /// Assemble the accumulated deltas into a `MessageResponse`. Call once, after
    /// the stream completes and `err` is clear.
    pub fn response(self: *StreamAccumulator) error{OutOfMemory}!MessageResponse {
        var blocks: std.ArrayListUnmanaged(types.ContentBlock) = .empty;
        if (self.text.items.len > 0) {
            try blocks.append(self.arena, .{ .type = "text", .text = self.text.items });
        }
        for (self.tool_blocks.items) |tb| {
            const input: ?std.json.Value = if (tb.json.items.len > 0)
                (std.json.parseFromSliceLeaky(std.json.Value, self.arena, tb.json.items, .{}) catch null)
            else
                null;
            try blocks.append(self.arena, .{ .type = "tool_use", .id = tb.id, .name = tb.name, .input = input });
        }
        return .{
            .content = if (blocks.items.len > 0) blocks.items else null,
            .stop_reason = self.stop_reason,
            .usage = self.usage,
        };
    }
};

// --- Models ---

/// List available models.
pub fn listModels(self: *Client) ApiError!Response(types.ListModelsResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/models", .{self.base_url});
    defer self.allocator.free(url);
    return self.fetchGet(url, types.ListModelsResponse);
}

/// Whether `m` is a chat model. Anthropic only ships Claude (chat-only),
/// so this is unconditionally true. Provided for API symmetry with the
/// other providers' `isChatModel` predicates.
pub fn isChatModel(_: types.Model) bool {
    return true;
}

test "Client init and deinit" {
    var client = Client.init(std.testing.io, std.testing.allocator, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expectEqualStrings("https://api.anthropic.com/v1", client.base_url);
    try std.testing.expectEqualStrings("2023-06-01", client.api_version);
}

test "listModels: missing api key" {
    var client = Client.init(std.testing.io, std.testing.allocator, "", .{});
    defer client.deinit();
    try std.testing.expectError(error.MissingApiKey, client.listModels());
}

fn headerValue(headers: []const std.http.Header, name: []const u8) ?[]const u8 {
    for (headers) |h| if (std.mem.eql(u8, h.name, name)) return h.value;
    return null;
}

test "authHeaders: api_key mode sends x-api-key, no bearer" {
    var client = Client.init(std.testing.io, std.testing.allocator, "sk-test", .{});
    defer client.deinit();
    var buf: [3]std.http.Header = undefined;
    const headers = try client.authHeaders(&buf);
    try std.testing.expectEqualStrings("sk-test", headerValue(headers, "x-api-key").?);
    try std.testing.expectEqualStrings("2023-06-01", headerValue(headers, "anthropic-version").?);
    try std.testing.expectEqual(@as(?[]const u8, null), headerValue(headers, "authorization"));
    try std.testing.expectEqual(@as(?[]const u8, null), headerValue(headers, "anthropic-beta"));
}

test "authHeaders: bearer mode sends Authorization + oauth beta, no x-api-key" {
    var client = Client.init(std.testing.io, std.testing.allocator, "tok-abc", .{ .auth = .bearer });
    defer client.deinit();
    var buf: [3]std.http.Header = undefined;
    const headers = try client.authHeaders(&buf);
    try std.testing.expectEqualStrings("Bearer tok-abc", headerValue(headers, "authorization").?);
    try std.testing.expectEqualStrings(oauth_beta, headerValue(headers, "anthropic-beta").?);
    try std.testing.expectEqual(@as(?[]const u8, null), headerValue(headers, "x-api-key"));
}

test "setApiKey: rebuilds the cached bearer value" {
    var client = Client.init(std.testing.io, std.testing.allocator, "tok-old", .{ .auth = .bearer });
    defer client.deinit();
    var buf: [3]std.http.Header = undefined;
    try std.testing.expectEqualStrings("Bearer tok-old", headerValue(try client.authHeaders(&buf), "authorization").?);
    client.setApiKey("tok-new");
    try std.testing.expectEqual(@as(?[]u8, null), client.authorization);
    try std.testing.expectEqualStrings("Bearer tok-new", headerValue(try client.authHeaders(&buf), "authorization").?);
}

const TextProbe = struct {
    count: usize = 0,
    text: std.ArrayListUnmanaged(u8) = .empty,

    fn cb(ptr: *anyopaque, delta: []const u8) void {
        const self: *TextProbe = @ptrCast(@alignCast(ptr));
        self.count += 1;
        self.text.appendSlice(std.testing.allocator, delta) catch {};
    }
};

test "StreamAccumulator reassembles text and a tool_use block into a MessageResponse" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var probe = TextProbe{};
    defer probe.text.deinit(std.testing.allocator);

    var acc = StreamAccumulator.init(arena.allocator(), &probe, TextProbe.cb);
    const events = [_]StreamEvent{
        .{ .type = "message_start", .message = .{ .usage = .{ .input_tokens = 10, .output_tokens = 1 } } },
        .{ .type = "content_block_start", .index = 0, .content_block = .{ .type = "text" } },
        .{ .type = "content_block_delta", .index = 0, .delta = .{ .type = "text_delta", .text = "Let me " } },
        .{ .type = "content_block_delta", .index = 0, .delta = .{ .type = "text_delta", .text = "search." } },
        .{ .type = "content_block_start", .index = 1, .content_block = .{ .type = "tool_use", .id = "toolu_1", .name = "search" } },
        .{ .type = "content_block_delta", .index = 1, .delta = .{ .type = "input_json_delta", .partial_json = "{\"q\":" } },
        .{ .type = "content_block_delta", .index = 1, .delta = .{ .type = "input_json_delta", .partial_json = "\"cats\"}" } },
        .{ .type = "content_block_stop", .index = 1 },
        .{ .type = "message_delta", .delta = .{ .stop_reason = .tool_use }, .usage = .{ .output_tokens = 25 } },
        .{ .type = "message_stop" },
    };
    for (events) |ev| acc.onEvent(ev);
    try std.testing.expect(acc.err == null);

    const msg = try acc.response();
    try std.testing.expectEqual(@as(usize, 2), probe.count);
    try std.testing.expectEqualStrings("Let me search.", probe.text.items);
    try std.testing.expectEqualStrings("Let me search.", msg.text().?);
    try std.testing.expectEqual(types.StopReason.tool_use, msg.stop_reason.?);
    try std.testing.expectEqual(@as(?i32, 10), msg.usage.?.input_tokens);
    try std.testing.expectEqual(@as(?i32, 25), msg.usage.?.output_tokens);

    const tool = msg.firstToolUse().?;
    try std.testing.expectEqualStrings("toolu_1", tool.id.?);
    try std.testing.expectEqualStrings("search", tool.name.?);
    try std.testing.expectEqualStrings("cats", tool.input.?.object.get("q").?.string);
}
