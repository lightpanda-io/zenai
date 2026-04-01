const std = @import("std");
const types = @import("types.zig");
const http = @import("../http.zig");

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
        .last_error = null,
        .last_error_status = null,
    };
}

/// Release all resources held by the client, including HTTP connections.
pub fn deinit(self: *Client) void {
    self.http_client.deinit();
}

/// Owns the parsed response and its backing memory.
/// Call `deinit()` when done to free all resources.
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

fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
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
    var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
    errdefer response_buf.deinit();

    const auth = self.authHeaders();
    const result = try self.http_client.fetch(.{
        .location = .{ .url = url },
        .extra_headers = &auth,
        .response_writer = &response_buf.writer,
    });

    const body = response_buf.written();
    const status_code: u10 = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        self.setErrorDetail(status_code, body);
        return error.ApiError;
    }
    if (body.len == 0) return error.EmptyResponse;

    const parsed = try std.json.parseFromSlice(T, self.allocator, body, .{ .ignore_unknown_fields = true });
    return .{ .value = parsed.value, .json_buf = response_buf, .parsed = parsed };
}

fn fetchPost(self: *Client, url: []const u8, body: anytype, comptime T: type) ApiError!Response(T) {
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;
    const payload = payload_buf.written();

    var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
    errdefer response_buf.deinit();

    const auth = self.authHeaders();
    const result = try self.http_client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload,
        .extra_headers = &auth,
        .headers = .{
            .content_type = .{ .override = "application/json" },
        },
        .response_writer = &response_buf.writer,
    });

    const resp_body = response_buf.written();
    const status_code: u10 = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        self.setErrorDetail(status_code, resp_body);
        return error.ApiError;
    }
    if (resp_body.len == 0) return error.EmptyResponse;

    const parsed = try std.json.parseFromSlice(T, self.allocator, resp_body, .{ .ignore_unknown_fields = true });
    return .{ .value = parsed.value, .json_buf = response_buf, .parsed = parsed };
}

fn fetchDelete(self: *Client, url: []const u8) ApiError!void {
    const auth = self.authHeaders();
    const result = try self.http_client.fetch(.{
        .location = .{ .url = url },
        .method = .DELETE,
        .extra_headers = &auth,
    });

    const status_code: u10 = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        self.setErrorDetail(status_code, "");
        return error.ApiError;
    }
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

test "Client init and deinit" {
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expectEqualStrings("https://api.openai.com/v1", client.base_url);
}
