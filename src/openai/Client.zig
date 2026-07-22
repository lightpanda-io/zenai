const std = @import("std");
const types = @import("types.zig");
const http = @import("../http.zig");
const retry = @import("../retry.zig");

pub const RetryPolicy = retry.RetryPolicy;

const Message = types.Message;
const ChatCompletionRequest = types.ChatCompletionRequest;
const ChatCompletionResponse = types.ChatCompletionResponse;
const ResponsesRequest = types.ResponsesRequest;
const ResponsesResponse = types.ResponsesResponse;
const ResponseStreamEvent = types.ResponseStreamEvent;

/// OpenAI API client. Provides access to chat completions, embeddings,
/// and model management.
const Client = @This();

allocator: std.mem.Allocator,
api_key: []const u8,
base_url: []const u8,
organization: ?[]const u8,
project: ?[]const u8,
/// Hugging Face org to bill via `X-HF-Bill-To`, so the router charges a
/// Team/Enterprise org instead of the token owner. Ignored by OpenAI/Ollama.
bill_to: ?[]const u8,
http_client: std.http.Client,
/// Retry policy applied to every non-streaming request.
retry_policy: RetryPolicy,
/// Human-readable message from the most recent API error, owned by the client
/// and freed on the next failure or `deinit`. Set on `error.ApiError`.
last_error_message: ?[]u8 = null,
last_error_status: ?u10 = null,
/// Set by the host so a SIGINT can abort an in-flight request mid-read.
interrupt: ?*http.Interrupt = null,
/// Cached `Bearer <api_key>` header value, built on first request.
authorization: ?[]const u8 = null,
/// Per-model cache of the Ollama context window (see `ollama.zig`), so the
/// `/api/show` lookup runs once per model. `model` is owned by `allocator`;
/// `len` is null when the lookup missed (cached so it isn't retried).
ollama_ctx: ?struct { model: []const u8, len: ?i32 } = null,

/// Options for customizing the API endpoint.
pub const InitOptions = struct {
    /// Base URL for the OpenAI API.
    base_url: []const u8 = "https://api.openai.com/v1",
    /// Organization ID for API requests.
    organization: ?[]const u8 = null,
    /// Project ID for API requests.
    project: ?[]const u8 = null,
    /// Hugging Face org to bill via the `X-HF-Bill-To` header (see field docs).
    bill_to: ?[]const u8 = null,
    /// Retry policy for transient HTTP failures (5xx, 429, and known
    /// flaky network errors). Pass `RetryPolicy.disabled` to opt out.
    retry_policy: RetryPolicy = .{},
};

/// Create a new OpenAI API client.
pub fn init(io: std.Io, allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .organization = options.organization,
        .project = options.project,
        .bill_to = options.bill_to,
        .http_client = .{ .allocator = allocator, .io = io },
        .retry_policy = options.retry_policy,
        .last_error_message = null,
        .last_error_status = null,
    };
}

/// Release all resources held by the client, including HTTP connections.
pub fn deinit(self: *Client) void {
    if (self.authorization) |a| self.allocator.free(a);
    if (self.ollama_ctx) |c| self.allocator.free(c.model);
    if (self.last_error_message) |m| self.allocator.free(m);
    self.http_client.deinit();
}

pub const Response = http.Response;

pub const ApiError = error{
    ApiError,
    MissingApiKey,
    EmptyResponse,
} || std.http.Client.FetchError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

// --- Internal helpers ---

/// `X-HF-Bill-To` is only emitted when `bill_to` is set, so providers that
/// don't recognize it never see it.
fn authHeaders(self: *Client, buf: *[4]std.http.Header) ![]const std.http.Header {
    if (self.authorization == null)
        self.authorization = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_key});
    buf[0] = .{ .name = "Authorization", .value = self.authorization.? };
    buf[1] = .{ .name = "OpenAI-Organization", .value = self.organization orelse "" };
    buf[2] = .{ .name = "OpenAI-Project", .value = self.project orelse "" };
    if (self.bill_to) |org| {
        buf[3] = .{ .name = "X-HF-Bill-To", .value = org };
        return buf[0..4];
    }
    return buf[0..3];
}

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error_status = status_code;
    if (self.last_error_message) |m| {
        self.allocator.free(m);
        self.last_error_message = null;
    }
    if (body.len > 0) {
        std.log.err("OpenAI API error (HTTP {d}): {s}", .{ status_code, body });
        self.last_error_message = http.extractErrorMessage(self.allocator, body);
    }
}

fn fetchGet(self: *Client, url: []const u8, comptime T: type) ApiError!Response(T) {
    var hdr_buf: [4]std.http.Header = undefined;
    const auth = try self.authHeaders(&hdr_buf);
    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, .{
        .location = .{ .url = url },
        .extra_headers = auth,
    }, T, self);
}

fn fetchPost(self: *Client, url: []const u8, body: anytype, comptime T: type) ApiError!Response(T) {
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    var hdr_buf: [4]std.http.Header = undefined;
    const auth = try self.authHeaders(&hdr_buf);
    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, .{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload_buf.written(),
        .extra_headers = auth,
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
    /// The effort level for model reasoning.
    reasoning_effort: ?types.ReasoningEffort = null,
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
        .max_completion_tokens = config.max_tokens,
        .top_p = config.top_p,
        .frequency_penalty = config.frequency_penalty,
        .presence_penalty = config.presence_penalty,
        .stop = config.stop,
        .tools = config.tools,
        .tool_choice = config.tool_choice,
        .seed = config.seed,
        .response_format = config.response_format,
        .reasoning_effort = config.reasoning_effort,
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

// --- Responses ---

/// Create a model response via the Responses API (`POST /responses`).
/// Required over `chatCompletion` for gpt-5.x, which rejects function tools
/// combined with reasoning on the chat completions endpoint.
pub fn createResponse(self: *Client, request: ResponsesRequest) ApiError!Response(ResponsesResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/responses", .{self.base_url});
    defer self.allocator.free(url);

    return self.fetchPost(url, request, ResponsesResponse);
}

/// Stream a model response via the Responses API (`POST /responses`). The
/// `callback` fires per SSE event; `request.stream` is forced on. Unlike Chat
/// Completions, this endpoint supports function tools together with reasoning
/// on the gpt-5 family, so native OpenAI streams through here.
pub fn createResponseStream(
    self: *Client,
    request: ResponsesRequest,
    context: anytype,
    callback: *const fn (@TypeOf(context), ResponseStreamEvent) void,
) StreamError!void {
    if (self.api_key.len == 0) return error.MissingApiKey;

    const url = try std.fmt.allocPrint(self.allocator, "{s}/responses", .{self.base_url});
    defer self.allocator.free(url);

    var req_body = request;
    req_body.stream = true;

    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(req_body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    var hdr_buf: [4]std.http.Header = undefined;
    const auth = try self.authHeaders(&hdr_buf);
    return http.streamSse(self.allocator, &self.http_client, url, auth, payload_buf.written(), ResponseStreamEvent, self, context, callback);
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
        // Ask for the trailing usage chunk so streamed turns still report cost.
        .stream_options = .{ .include_usage = true },
        .temperature = config.temperature,
        .max_completion_tokens = config.max_tokens,
        .top_p = config.top_p,
        .frequency_penalty = config.frequency_penalty,
        .presence_penalty = config.presence_penalty,
        .stop = config.stop,
        .tools = config.tools,
        .tool_choice = config.tool_choice,
        .seed = config.seed,
        .response_format = config.response_format,
        .reasoning_effort = config.reasoning_effort,
    };
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(req_body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    var hdr_buf: [4]std.http.Header = undefined;
    const auth = try self.authHeaders(&hdr_buf);
    return http.streamSse(self.allocator, &self.http_client, url, auth, payload_buf.written(), ChatCompletionResponse, self, context, callback);
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

/// Reassembles a streamed chat completion into the same `ChatCompletionResponse`
/// the non-streaming `chatCompletion` returns, correlating tool-call argument
/// fragments by their delta index and firing `on_text_fn` per content delta.
/// Drive it by passing `&acc` and `onEvent` to `chatCompletionStream`, then call
/// `response()`. All storage lives in `arena`.
pub const StreamAccumulator = struct {
    arena: std.mem.Allocator,
    on_text_ctx: *anyopaque,
    on_text_fn: *const fn (*anyopaque, []const u8) void,
    content: std.ArrayListUnmanaged(u8) = .empty,
    calls: std.ArrayListUnmanaged(CallFragment) = .empty,
    finish_reason: ?types.FinishReason = null,
    usage: ?types.Usage = null,
    err: ?error{OutOfMemory} = null,

    const CallFragment = struct {
        index: i32,
        id: std.ArrayListUnmanaged(u8) = .empty,
        name: std.ArrayListUnmanaged(u8) = .empty,
        args: std.ArrayListUnmanaged(u8) = .empty,
    };

    pub fn init(
        arena: std.mem.Allocator,
        on_text_ctx: *anyopaque,
        on_text_fn: *const fn (*anyopaque, []const u8) void,
    ) StreamAccumulator {
        return .{ .arena = arena, .on_text_ctx = on_text_ctx, .on_text_fn = on_text_fn };
    }

    pub fn onEvent(self: *StreamAccumulator, chunk: ChatCompletionResponse) void {
        self.handle(chunk) catch |e| {
            self.err = e;
        };
    }

    fn slot(self: *StreamAccumulator, idx: i32) error{OutOfMemory}!*CallFragment {
        for (self.calls.items) |*c| {
            if (c.index == idx) return c;
        }
        try self.calls.append(self.arena, .{ .index = idx });
        return &self.calls.items[self.calls.items.len - 1];
    }

    fn handle(self: *StreamAccumulator, chunk: ChatCompletionResponse) error{OutOfMemory}!void {
        // The final `include_usage` chunk carries usage with no choices.
        if (chunk.usage != null) self.usage = chunk.usage;
        const choices = chunk.choices orelse return;
        if (choices.len == 0) return;
        const c0 = choices[0];
        if (c0.finish_reason) |fr| self.finish_reason = fr;
        const delta = c0.delta orelse return;
        if (delta.content) |txt| {
            if (txt.len > 0) {
                try self.content.appendSlice(self.arena, txt);
                self.on_text_fn(self.on_text_ctx, txt);
            }
        }
        if (delta.tool_calls) |tcs| {
            for (tcs, 0..) |tc, i| {
                const call = try self.slot(tc.index orelse @intCast(i));
                if (tc.id) |id| {
                    call.id.clearRetainingCapacity();
                    try call.id.appendSlice(self.arena, id);
                }
                if (tc.function) |f| {
                    if (f.name) |n| {
                        call.name.clearRetainingCapacity();
                        try call.name.appendSlice(self.arena, n);
                    }
                    if (f.arguments) |a| try call.args.appendSlice(self.arena, a);
                }
            }
        }
    }

    /// Assemble the accumulated deltas into a `ChatCompletionResponse`. Call
    /// once, after the stream completes and `err` is clear.
    pub fn response(self: *StreamAccumulator) error{OutOfMemory}!ChatCompletionResponse {
        var tool_calls: std.ArrayListUnmanaged(types.ToolCall) = .empty;
        for (self.calls.items) |c| {
            try tool_calls.append(self.arena, .{
                .id = c.id.items,
                .type = "function",
                .function = .{ .name = c.name.items, .arguments = c.args.items },
            });
        }
        const message = types.Message{
            .role = .assistant,
            .content = if (self.content.items.len > 0) self.content.items else null,
            .tool_calls = if (tool_calls.items.len > 0) tool_calls.items else null,
        };
        const choices = try self.arena.dupe(types.Choice, &.{.{ .index = 0, .message = message, .finish_reason = self.finish_reason }});
        return .{ .choices = choices, .usage = self.usage };
    }
};

/// Reassembles a streamed Responses-API response into the same `ResponsesResponse`
/// the non-streaming `createResponse` returns. Text arrives as `output_text.delta`
/// events (forwarded to `on_text_fn`); each `output_item.done` carries a complete
/// `function_call` item; the terminal `response.completed`/`incomplete` event
/// carries usage and status. All storage lives in `arena`.
pub const ResponsesStreamAccumulator = struct {
    arena: std.mem.Allocator,
    on_text_ctx: *anyopaque,
    on_text_fn: *const fn (*anyopaque, []const u8) void,
    text: std.ArrayListUnmanaged(u8) = .empty,
    calls: std.ArrayListUnmanaged(types.ResponseOutputItem) = .empty,
    usage: ?types.ResponsesUsage = null,
    incomplete_reason: ?[]const u8 = null,
    err: ?error{OutOfMemory} = null,

    pub fn init(
        arena: std.mem.Allocator,
        on_text_ctx: *anyopaque,
        on_text_fn: *const fn (*anyopaque, []const u8) void,
    ) ResponsesStreamAccumulator {
        return .{ .arena = arena, .on_text_ctx = on_text_ctx, .on_text_fn = on_text_fn };
    }

    pub fn onEvent(self: *ResponsesStreamAccumulator, event: ResponseStreamEvent) void {
        self.handle(event) catch |e| {
            self.err = e;
        };
    }

    fn handle(self: *ResponsesStreamAccumulator, event: ResponseStreamEvent) error{OutOfMemory}!void {
        const t = event.type orelse return;
        if (std.mem.eql(u8, t, "response.output_text.delta")) {
            if (event.delta) |d| {
                if (d.len > 0) {
                    try self.text.appendSlice(self.arena, d);
                    self.on_text_fn(self.on_text_ctx, d);
                }
            }
        } else if (std.mem.eql(u8, t, "response.output_item.done")) {
            const item = event.item orelse return;
            const it = item.type orelse return;
            if (std.mem.eql(u8, it, "function_call")) {
                // Strings point into the transient parsed event; dupe them.
                try self.calls.append(self.arena, .{
                    .type = "function_call",
                    .call_id = if (item.call_id) |c| try self.arena.dupe(u8, c) else null,
                    .name = if (item.name) |n| try self.arena.dupe(u8, n) else null,
                    .arguments = if (item.arguments) |a| try self.arena.dupe(u8, a) else null,
                });
            }
        } else if (std.mem.eql(u8, t, "response.completed") or std.mem.eql(u8, t, "response.incomplete")) {
            if (event.response) |r| {
                // Usage is all ints — a value copy outlives the event; the
                // incomplete reason is a string and must be duped.
                if (r.usage) |u| self.usage = u;
                if (r.incomplete_details) |d| {
                    self.incomplete_reason = if (d.reason) |rr| try self.arena.dupe(u8, rr) else null;
                }
            }
        }
    }

    /// Assemble the accumulated deltas into a `ResponsesResponse`. Call once,
    /// after the stream completes and `err` is clear.
    pub fn response(self: *ResponsesStreamAccumulator) error{OutOfMemory}!ResponsesResponse {
        var output: std.ArrayListUnmanaged(types.ResponseOutputItem) = .empty;
        if (self.text.items.len > 0) {
            const content = try self.arena.dupe(types.ResponseOutputContent, &.{.{ .type = "output_text", .text = self.text.items }});
            try output.append(self.arena, .{ .type = "message", .role = "assistant", .content = content });
        }
        try output.appendSlice(self.arena, self.calls.items);
        return .{
            .output = if (output.items.len > 0) output.items else null,
            .usage = self.usage,
            .incomplete_details = if (self.incomplete_reason) |r| .{ .reason = r } else null,
        };
    }
};

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
        if (std.mem.find(u8, id, s) != null) return false;
    }
    return true;
}

test "Client init and deinit" {
    var client = Client.init(std.testing.io, std.testing.allocator, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expectEqualStrings("https://api.openai.com/v1", client.base_url);
}

test "authHeaders emits X-HF-Bill-To only when bill_to is set" {
    var buf: [4]std.http.Header = undefined;

    var plain = Client.init(std.testing.io, std.testing.allocator, "k", .{});
    defer plain.deinit();
    try std.testing.expectEqual(@as(usize, 3), (try plain.authHeaders(&buf)).len);

    var billed = Client.init(std.testing.io, std.testing.allocator, "k", .{ .bill_to = "my-org" });
    defer billed.deinit();
    const headers = try billed.authHeaders(&buf);
    try std.testing.expectEqual(@as(usize, 4), headers.len);
    try std.testing.expectEqualStrings("X-HF-Bill-To", headers[3].name);
    try std.testing.expectEqualStrings("my-org", headers[3].value);
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

const TextProbe = struct {
    count: usize = 0,
    text: std.ArrayListUnmanaged(u8) = .empty,

    fn cb(ptr: *anyopaque, delta: []const u8) void {
        const self: *TextProbe = @ptrCast(@alignCast(ptr));
        self.count += 1;
        self.text.appendSlice(std.testing.allocator, delta) catch {};
    }
};

test "StreamAccumulator reassembles chat text and correlates tool-call fragments by index" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var probe = TextProbe{};
    defer probe.text.deinit(std.testing.allocator);

    var acc = StreamAccumulator.init(arena.allocator(), &probe, TextProbe.cb);
    const chunks = [_]ChatCompletionResponse{
        .{ .choices = &.{.{ .index = 0, .delta = .{ .content = "Hel" } }} },
        .{ .choices = &.{.{ .index = 0, .delta = .{ .content = "lo" } }} },
        .{ .choices = &.{.{ .index = 0, .delta = .{ .tool_calls = &.{.{ .index = 0, .id = "call_1", .type = "function", .function = .{ .name = "search", .arguments = "" } }} } }} },
        .{ .choices = &.{.{ .index = 0, .delta = .{ .tool_calls = &.{.{ .index = 0, .function = .{ .arguments = "{\"q\":" } }} } }} },
        .{ .choices = &.{.{ .index = 0, .delta = .{ .tool_calls = &.{.{ .index = 0, .function = .{ .arguments = "\"dogs\"}" } }} } }} },
        .{ .choices = &.{.{ .index = 0, .delta = .{}, .finish_reason = .tool_calls }} },
        .{ .choices = &.{}, .usage = .{ .prompt_tokens = 5, .completion_tokens = 7, .total_tokens = 12 } },
    };
    for (chunks) |c| acc.onEvent(c);
    try std.testing.expect(acc.err == null);

    const resp = try acc.response();
    try std.testing.expectEqual(@as(usize, 2), probe.count);
    try std.testing.expectEqualStrings("Hello", resp.text().?);
    try std.testing.expectEqual(@as(?i32, 7), resp.usage.?.completion_tokens);

    const tc = resp.firstToolCall().?;
    try std.testing.expectEqualStrings("call_1", tc.id.?);
    try std.testing.expectEqualStrings("search", tc.function.?.name.?);
    try std.testing.expectEqualStrings("{\"q\":\"dogs\"}", tc.function.?.arguments.?);
}

test "ResponsesStreamAccumulator reassembles text and a completed function-call item" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var probe = TextProbe{};
    defer probe.text.deinit(std.testing.allocator);

    var acc = ResponsesStreamAccumulator.init(arena.allocator(), &probe, TextProbe.cb);
    const events = [_]ResponseStreamEvent{
        .{ .type = "response.output_text.delta", .delta = "Search" },
        .{ .type = "response.output_text.delta", .delta = "ing." },
        .{ .type = "response.output_item.done", .item = .{ .type = "function_call", .call_id = "call_9", .name = "search", .arguments = "{\"q\":\"birds\"}" } },
        .{ .type = "response.completed", .response = .{
            .status = "completed",
            .usage = .{ .input_tokens = 8, .output_tokens = 4, .total_tokens = 12 },
        } },
    };
    for (events) |ev| acc.onEvent(ev);
    try std.testing.expect(acc.err == null);

    const resp = try acc.response();
    try std.testing.expectEqual(@as(usize, 2), probe.count);
    try std.testing.expectEqualStrings("Searching.", resp.text().?);
    try std.testing.expectEqual(@as(?i32, 4), resp.usage.?.output_tokens);

    const items = resp.output.?;
    var fc: ?types.ResponseOutputItem = null;
    for (items) |it| {
        if (it.type) |t| if (std.mem.eql(u8, t, "function_call")) {
            fc = it;
        };
    }
    try std.testing.expectEqualStrings("call_9", fc.?.call_id.?);
    try std.testing.expectEqualStrings("search", fc.?.name.?);
    try std.testing.expectEqualStrings("{\"q\":\"birds\"}", fc.?.arguments.?);
}
