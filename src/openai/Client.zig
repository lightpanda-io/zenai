const std = @import("std");
const types = @import("types.zig");
const http = @import("../http.zig");
const retry = @import("../retry.zig");

pub const RetryPolicy = retry.RetryPolicy;

const Message = types.Message;
const ChatCompletionRequest = types.ChatCompletionRequest;
const ChatCompletionResponse = types.ChatCompletionResponse;

/// OpenAI API client. Provides access to chat completions, embeddings,
/// and model management.
const Client = @This();

allocator: std.mem.Allocator,
api_key: []const u8,
base_url: []const u8,
organization: ?[]const u8,
project: ?[]const u8,
http_client: std.http.Client,
/// Retry policy applied to every non-streaming request.
retry_policy: RetryPolicy,
/// The most recent API error detail, if any. Set on `error.ApiError`.
last_error: ?types.ApiErrorDetail = null,
last_error_status: ?u10 = null,

/// Options for customizing the API endpoint.
pub const InitOptions = struct {
    /// Base URL for the OpenAI API.
    base_url: []const u8 = "https://api.openai.com/v1",
    /// Organization ID for API requests.
    organization: ?[]const u8 = null,
    /// Project ID for API requests.
    project: ?[]const u8 = null,
    /// Retry policy for transient HTTP failures (5xx, 429, and known
    /// flaky network errors). Pass `RetryPolicy.disabled` to opt out.
    retry_policy: RetryPolicy = .{},
};

/// Create a new OpenAI API client.
pub fn init(allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .organization = options.organization,
        .project = options.project,
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

fn authHeaders(self: *const Client) [3]std.http.Header {
    return .{
        .{ .name = "Authorization", .value = self.api_key },
        .{ .name = "OpenAI-Organization", .value = self.organization orelse "" },
        .{ .name = "OpenAI-Project", .value = self.project orelse "" },
    };
}

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error_status = status_code;
    self.last_error = null;
    if (body.len > 0) {
        std.log.err("OpenAI API error (HTTP {d}): {s}", .{ status_code, body });
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

// --- Chat Completions ---

/// Configuration for chat completion requests.
pub const ChatCompletionConfig = struct {
    /// Sampling temperature (0.0-2.0).
    temperature: ?f32 = null,
    /// Maximum number of tokens to generate.
    max_tokens: ?i32 = null,
    /// Nucleus sampling threshold (0.0-1.0).
    top_p: ?f32 = null,
    /// Frequency penalty (-2.0 to 2.0).
    frequency_penalty: ?f32 = null,
    /// Presence penalty (-2.0 to 2.0).
    presence_penalty: ?f32 = null,
    /// Up to 4 sequences where generation stops.
    stop: ?[]const []const u8 = null,
    /// Tools the model may call.
    tools: ?[]const types.Tool = null,
    /// Controls which tool is called ("none", "auto", "required").
    tool_choice: ?[]const u8 = null,
    /// Random seed for deterministic output.
    seed: ?i32 = null,
    /// Response format specification.
    response_format: ?types.ResponseFormat = null,
};

/// Create a chat completion.
pub fn chatCompletion(
    self: *Client,
    model: []const u8,
    messages: []const Message,
    config: ChatCompletionConfig,
) ApiError!Response(ChatCompletionResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/chat/completions", .{self.base_url});
    defer self.allocator.free(url);

    return self.fetchPost(url, ChatCompletionRequest{
        .model = model,
        .messages = messages,
        .temperature = config.temperature,
        .max_tokens = config.max_tokens,
        .top_p = config.top_p,
        .frequency_penalty = config.frequency_penalty,
        .presence_penalty = config.presence_penalty,
        .stop = config.stop,
        .tools = config.tools,
        .tool_choice = config.tool_choice,
        .seed = config.seed,
        .response_format = config.response_format,
    }, ChatCompletionResponse);
}

/// Convenience: create a chat completion from a single text prompt.
pub fn chatCompletionFromText(
    self: *Client,
    model: []const u8,
    prompt: []const u8,
    config: ChatCompletionConfig,
) ApiError!Response(ChatCompletionResponse) {
    const messages = [_]Message{.{ .role = .user, .content = prompt }};
    return self.chatCompletion(model, &messages, config);
}

// --- Streaming ---

pub const StreamError = error{
    ApiError,
    MissingApiKey,
    InvalidSseData,
} || std.http.Client.RequestError || std.http.Client.Request.ReceiveHeadError || std.Io.Writer.Error || std.Io.Reader.DelimiterError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

/// Stream a chat completion via Server-Sent Events.
/// The `callback` is invoked for each chunk as it arrives.
pub fn chatCompletionStream(
    self: *Client,
    model: []const u8,
    messages: []const Message,
    config: ChatCompletionConfig,
    context: anytype,
    callback: *const fn (@TypeOf(context), ChatCompletionResponse) void,
) StreamError!void {
    if (self.api_key.len == 0) return error.MissingApiKey;

    const url = try std.fmt.allocPrint(self.allocator, "{s}/chat/completions", .{self.base_url});
    defer self.allocator.free(url);

    const req_body = ChatCompletionRequest{
        .model = model,
        .messages = messages,
        .stream = true,
        .temperature = config.temperature,
        .max_tokens = config.max_tokens,
        .top_p = config.top_p,
        .frequency_penalty = config.frequency_penalty,
        .presence_penalty = config.presence_penalty,
        .stop = config.stop,
        .tools = config.tools,
        .tool_choice = config.tool_choice,
        .seed = config.seed,
        .response_format = config.response_format,
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

    while (true) {
        const line = reader.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => return error.InvalidSseData,
            error.ReadFailed => return,
        } orelse return;

        const trimmed = std.mem.trimRight(u8, line, "\r");
        if (trimmed.len == 0) continue;

        if (std.mem.startsWith(u8, trimmed, "data: ")) {
            const json_data = trimmed["data: ".len..];

            // OpenAI signals end of stream with [DONE]
            if (std.mem.eql(u8, json_data, "[DONE]")) return;

            const parsed = std.json.parseFromSlice(ChatCompletionResponse, self.allocator, json_data, .{ .ignore_unknown_fields = true }) catch |err| {
                std.log.err("OpenAI streaming: failed to parse SSE chunk: {}", .{err});
                return error.InvalidSseData;
            };
            defer parsed.deinit();
            callback(context, parsed.value);
        }
    }
}

/// Convenience: stream a chat completion from a single text prompt.
pub fn chatCompletionStreamFromText(
    self: *Client,
    model: []const u8,
    prompt: []const u8,
    config: ChatCompletionConfig,
    context: anytype,
    callback: *const fn (@TypeOf(context), ChatCompletionResponse) void,
) StreamError!void {
    const messages = [_]Message{.{ .role = .user, .content = prompt }};
    return self.chatCompletionStream(model, &messages, config, context, callback);
}

// --- Embeddings ---

/// Configuration for embedding requests.
pub const EmbedConfig = struct {
    /// The encoding format ("float" or "base64").
    encoding_format: ?[]const u8 = null,
    /// The number of dimensions for the output embedding.
    dimensions: ?i32 = null,
};

/// Create an embedding for the given text.
pub fn createEmbedding(
    self: *Client,
    model: []const u8,
    input: []const u8,
    config: EmbedConfig,
) ApiError!Response(types.EmbeddingResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/embeddings", .{self.base_url});
    defer self.allocator.free(url);

    return self.fetchPost(url, types.EmbeddingRequest{
        .input = input,
        .model = model,
        .encoding_format = config.encoding_format,
        .dimensions = config.dimensions,
    }, types.EmbeddingResponse);
}

/// Convenience: create an embedding with default config.
pub fn embedText(
    self: *Client,
    model: []const u8,
    input: []const u8,
) ApiError!Response(types.EmbeddingResponse) {
    return self.createEmbedding(model, input, .{});
}

// --- Models ---

/// List available models.
pub fn listModels(self: *Client) ApiError!Response(types.ListModelsResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/models", .{self.base_url});
    defer self.allocator.free(url);
    return self.fetchGet(url, types.ListModelsResponse);
}

/// Get metadata for a specific model.
pub fn getModel(self: *Client, model: []const u8) ApiError!Response(types.Model) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/models/{s}", .{ self.base_url, model });
    defer self.allocator.free(url);
    return self.fetchGet(url, types.Model);
}

/// Whether `m` looks like a chat/text-generation model.
///
/// OpenAI's `/v1/models` exposes no modality metadata, so this is a
/// name-based heuristic: the id must start with a known chat family prefix
/// (gpt-, chatgpt-, o1, o3, o4) and must not contain a substring marking a
/// non-chat variant (embedding, image, audio, realtime, transcribe, tts,
/// moderation). The list will need updating as OpenAI introduces new
/// families or non-chat variants.
pub fn isChatModel(m: types.Model) bool {
    const id = m.id orelse return false;

    const chat_prefixes = [_][]const u8{ "gpt-", "chatgpt-", "o1", "o3", "o4" };
    var matches_prefix = false;
    for (chat_prefixes) |p| {
        if (std.mem.startsWith(u8, id, p)) {
            matches_prefix = true;
            break;
        }
    }
    if (!matches_prefix) return false;

    const non_chat_substrings = [_][]const u8{
        "embedding", "image", "audio", "realtime", "transcribe", "tts", "moderation",
    };
    for (non_chat_substrings) |s| {
        if (std.mem.indexOf(u8, id, s) != null) return false;
    }
    return true;
}

test "Client init and deinit" {
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expectEqualStrings("https://api.openai.com/v1", client.base_url);
}

test "isChatModel keeps chat families" {
    const T = struct {
        fn m(id: []const u8) types.Model {
            return .{ .id = id };
        }
    };
    try std.testing.expect(isChatModel(T.m("gpt-4o")));
    try std.testing.expect(isChatModel(T.m("gpt-4o-mini")));
    try std.testing.expect(isChatModel(T.m("gpt-5")));
    try std.testing.expect(isChatModel(T.m("o1-mini")));
    try std.testing.expect(isChatModel(T.m("o3")));
    try std.testing.expect(isChatModel(T.m("o4-mini")));
    try std.testing.expect(isChatModel(T.m("chatgpt-4o-latest")));
}

test "isChatModel drops non-chat" {
    const T = struct {
        fn m(id: []const u8) types.Model {
            return .{ .id = id };
        }
    };
    try std.testing.expect(!isChatModel(T.m("text-embedding-3-small")));
    try std.testing.expect(!isChatModel(T.m("dall-e-3")));
    try std.testing.expect(!isChatModel(T.m("whisper-1")));
    try std.testing.expect(!isChatModel(T.m("tts-1")));
    try std.testing.expect(!isChatModel(T.m("omni-moderation-latest")));
    try std.testing.expect(!isChatModel(T.m("babbage-002")));
    try std.testing.expect(!isChatModel(T.m("davinci-002")));
    try std.testing.expect(!isChatModel(T.m("gpt-image-1")));
    try std.testing.expect(!isChatModel(T.m("gpt-4o-audio-preview")));
    try std.testing.expect(!isChatModel(T.m("gpt-4o-realtime-preview")));
    try std.testing.expect(!isChatModel(T.m("gpt-4o-transcribe")));
    try std.testing.expect(!isChatModel(T.m("gpt-4o-mini-tts")));
}
