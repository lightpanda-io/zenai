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
http_client: std.http.Client,
/// Retry policy applied to every non-streaming request.
retry_policy: RetryPolicy,
/// The most recent API error detail, if any. Set on `error.ApiError`.
last_error: ?types.ApiErrorDetail = null,
last_error_status: ?u10 = null,

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
};

/// Create a new Anthropic API client.
pub fn init(allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .api_version = options.api_version,
        .http_client = .{ .allocator = allocator },
        .retry_policy = options.retry_policy,
        .last_error = null,
        .last_error_status = null,
    };
}

/// Release all resources held by the client, including HTTP connections.
pub fn deinit(self: *Client) void {
    self.http_client.deinit();
}

pub const Response = http.Response;

pub const ApiError = error{
    ApiError,
    MissingApiKey,
    EmptyResponse,
} || std.http.Client.FetchError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

// --- Internal helpers ---

fn authHeaders(self: *const Client) [2]std.http.Header {
    return .{
        .{ .name = "x-api-key", .value = self.api_key },
        .{ .name = "anthropic-version", .value = self.api_version },
    };
}

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error_status = status_code;
    self.last_error = null;
    if (body.len > 0) {
        std.log.err("Anthropic API error (HTTP {d}): {s}", .{ status_code, body });
        if (std.json.parseFromSlice(types.ApiErrorResponse, self.allocator, body, .{ .ignore_unknown_fields = true })) |parsed| {
            self.last_error = parsed.value.@"error";
            parsed.deinit();
        } else |_| {}
    }
}

fn fetchGet(self: *Client, url: []const u8, comptime T: type) ApiError!Response(T) {
    const auth = self.authHeaders();
    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, .{
        .location = .{ .url = url },
        .method = .GET,
        .extra_headers = &auth,
    }, T, self);
}

fn fetchPost(self: *Client, url: []const u8, body: anytype, comptime T: type) ApiError!Response(T) {
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    const auth = self.authHeaders();
    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, .{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload_buf.written(),
        .extra_headers = &auth,
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
    };
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(req_body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;
    const payload = payload_buf.written();

    const auth = self.authHeaders();
    const uri = try std.Uri.parse(url);
    var req = try self.http_client.request(.POST, uri, .{
        .extra_headers = &auth,
        .headers = .{
            .content_type = .{ .override = "application/json" },
        },
        .redirect_behavior = .init(5),
    });
    defer req.deinit();

    req.transfer_encoding = .{ .content_length = payload.len };
    var bw = try req.sendBodyUnflushed(&.{});
    try bw.writer.writeAll(payload);
    try bw.end();
    try req.connection.?.flush();

    var redirect_buf: [0]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);

    const status_code: u10 = @intFromEnum(response.head.status);
    if (status_code < 200 or status_code >= 300) {
        self.setErrorDetail(status_code, "");
        return error.ApiError;
    }

    const transfer_buf = try self.allocator.alloc(u8, 256 * 1024);
    defer self.allocator.free(transfer_buf);
    const reader = response.reader(transfer_buf);

    // Anthropic SSE sends `event: <type>` then `data: <json>` lines
    while (true) {
        const line = reader.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => return error.InvalidSseData,
            error.ReadFailed => return,
        } orelse return;

        const trimmed = std.mem.trimRight(u8, line, "\r");
        if (trimmed.len == 0) continue;

        // Skip event: lines — type info is in the data JSON
        if (std.mem.startsWith(u8, trimmed, "event: ")) continue;

        if (std.mem.startsWith(u8, trimmed, "data: ")) {
            const json_data = trimmed["data: ".len..];

            const parsed = std.json.parseFromSlice(StreamEvent, self.allocator, json_data, .{ .ignore_unknown_fields = true }) catch |err| {
                std.log.err("Anthropic streaming: failed to parse SSE chunk: {}", .{err});
                return error.InvalidSseData;
            };
            defer parsed.deinit();
            callback(context, parsed.value);
        }
    }
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
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expectEqualStrings("https://api.anthropic.com/v1", client.base_url);
    try std.testing.expectEqualStrings("2023-06-01", client.api_version);
}

test "listModels: missing api key" {
    var client = Client.init(std.testing.allocator, "", .{});
    defer client.deinit();
    try std.testing.expectError(error.MissingApiKey, client.listModels());
}
