const std = @import("std");

// --- Enums ---

/// The role of a message in a conversation.
pub const Role = enum {
    system,
    user,
    assistant,
    tool,
};

/// The reason why the model stopped generating tokens.
pub const FinishReason = enum {
    stop,
    length,
    tool_calls,
    content_filter,
};

// --- Core Types ---

/// A function call requested by the model.
pub const FunctionCall = struct {
    /// The name of the function to call.
    name: ?[]const u8 = null,
    /// The arguments as a JSON string.
    arguments: ?[]const u8 = null,
};

/// A tool call within an assistant message.
pub const ToolCall = struct {
    /// The unique ID of this tool call.
    id: ?[]const u8 = null,
    /// The type of tool (always "function").
    type: ?[]const u8 = null,
    /// The function call details.
    function: ?FunctionCall = null,
};

/// A message in a conversation.
pub const Message = struct {
    /// The role of the message author.
    role: ?Role = null,
    /// The text content of the message.
    content: ?[]const u8 = null,
    /// Optional participant name.
    name: ?[]const u8 = null,
    /// Tool calls requested by the assistant (only for role=assistant).
    tool_calls: ?[]const ToolCall = null,
    /// The tool call ID this message responds to (only for role=tool).
    tool_call_id: ?[]const u8 = null,
    /// Refusal message from the model.
    refusal: ?[]const u8 = null,
};

// --- Tools ---

/// A function definition for tool use.
pub const FunctionDef = struct {
    /// The name of the function.
    name: ?[]const u8 = null,
    /// A description of what the function does.
    description: ?[]const u8 = null,
    /// The function parameters as a JSON Schema object.
    parameters: ?std.json.Value = null,
    /// Whether to enforce strict schema compliance.
    strict: ?bool = null,
};

/// A tool the model may use.
pub const Tool = struct {
    /// The type of tool (always "function").
    type: ?[]const u8 = null,
    /// The function definition.
    function: ?FunctionDef = null,
};

/// Response format specification.
pub const ResponseFormat = struct {
    /// The format type ("text" or "json_object").
    type: ?[]const u8 = null,
};

// --- Request ---

/// Request body for the chat completions endpoint.
pub const ChatCompletionRequest = struct {
    /// The model to use (e.g. "gpt-4o", "gpt-4o-mini").
    model: []const u8,
    /// The messages in the conversation.
    messages: []const Message,
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
    /// Whether to stream the response.
    stream: ?bool = null,
    /// Tools the model may call.
    tools: ?[]const Tool = null,
    /// Controls which tool is called ("none", "auto", "required").
    tool_choice: ?[]const u8 = null,
    /// Random seed for deterministic output.
    seed: ?i32 = null,
    /// Response format specification.
    response_format: ?ResponseFormat = null,
    /// Number of completions to generate.
    n: ?i32 = null,
    /// Whether to return log probabilities.
    logprobs: ?bool = null,
    /// Number of most likely tokens to return (0-20).
    top_logprobs: ?i32 = null,
    /// Identifier for end-user monitoring.
    user: ?[]const u8 = null,
};

// --- Response ---

/// Token usage statistics.
pub const Usage = struct {
    /// Tokens in the prompt.
    prompt_tokens: ?i32 = null,
    /// Tokens in the completion.
    completion_tokens: ?i32 = null,
    /// Total tokens used.
    total_tokens: ?i32 = null,
};

/// Log probability information for a token.
pub const TokenLogprob = struct {
    /// The token string.
    token: ?[]const u8 = null,
    /// The log probability.
    logprob: ?f64 = null,
    /// The top alternative tokens and their log probabilities.
    top_logprobs: ?[]const TopLogprob = null,
};

/// A top alternative token with its log probability.
pub const TopLogprob = struct {
    /// The token string.
    token: ?[]const u8 = null,
    /// The log probability.
    logprob: ?f64 = null,
};

/// Log probability information for a choice.
pub const ChoiceLogprobs = struct {
    /// Log probability information for each token.
    content: ?[]const TokenLogprob = null,
};

/// A completion choice.
pub const Choice = struct {
    /// The index of this choice.
    index: ?i32 = null,
    /// The generated message (non-streaming).
    message: ?Message = null,
    /// The incremental message delta (streaming only).
    delta: ?Message = null,
    /// Why the model stopped generating.
    finish_reason: ?FinishReason = null,
    /// Log probability information.
    logprobs: ?ChoiceLogprobs = null,
};

/// Response from the chat completions endpoint.
pub const ChatCompletionResponse = struct {
    /// Unique response identifier.
    id: ?[]const u8 = null,
    /// Object type (always "chat.completion" or "chat.completion.chunk").
    object: ?[]const u8 = null,
    /// Unix timestamp of creation.
    created: ?i64 = null,
    /// The model used.
    model: ?[]const u8 = null,
    /// The completion choices.
    choices: ?[]const Choice = null,
    /// Token usage statistics (non-streaming only).
    usage: ?Usage = null,
    /// System fingerprint for determinism tracking.
    system_fingerprint: ?[]const u8 = null,

    /// Extract text from the first choice's message.
    pub fn text(self: ChatCompletionResponse) ?[]const u8 {
        const choices = self.choices orelse return null;
        if (choices.len == 0) return null;
        if (choices[0].message) |msg| return msg.content;
        if (choices[0].delta) |d| return d.content;
        return null;
    }

    /// Extract the first tool call from the first choice.
    pub fn firstToolCall(self: ChatCompletionResponse) ?ToolCall {
        const choices = self.choices orelse return null;
        if (choices.len == 0) return null;
        const msg = choices[0].message orelse return null;
        const tool_calls = msg.tool_calls orelse return null;
        if (tool_calls.len == 0) return null;
        return tool_calls[0];
    }
};

// --- Embeddings ---

/// Request body for the embeddings endpoint.
pub const EmbeddingRequest = struct {
    /// The input text to embed.
    input: []const u8,
    /// The model to use (e.g. "text-embedding-3-small").
    model: []const u8,
    /// The encoding format ("float" or "base64").
    encoding_format: ?[]const u8 = null,
    /// The number of dimensions for the output embedding.
    dimensions: ?i32 = null,
    /// Identifier for end-user monitoring.
    user: ?[]const u8 = null,
};

/// A single embedding result.
pub const Embedding = struct {
    /// The embedding vector.
    embedding: ?[]const f32 = null,
    /// The index of this embedding in the input list.
    index: ?i32 = null,
    /// Object type (always "embedding").
    object: ?[]const u8 = null,
};

/// Token usage for embeddings.
pub const EmbeddingUsage = struct {
    /// Tokens in the prompt.
    prompt_tokens: ?i32 = null,
    /// Total tokens used.
    total_tokens: ?i32 = null,
};

/// Response from the embeddings endpoint.
pub const EmbeddingResponse = struct {
    /// The list of embeddings.
    data: ?[]const Embedding = null,
    /// The model used.
    model: ?[]const u8 = null,
    /// Object type (always "list").
    object: ?[]const u8 = null,
    /// Token usage.
    usage: ?EmbeddingUsage = null,
};

// --- Models ---

/// An available model.
pub const Model = struct {
    /// The model identifier.
    id: ?[]const u8 = null,
    /// Object type (always "model").
    object: ?[]const u8 = null,
    /// Unix timestamp of creation.
    created: ?i64 = null,
    /// The organization that owns the model.
    owned_by: ?[]const u8 = null,
};

/// Response from the list models endpoint.
pub const ListModelsResponse = struct {
    data: ?[]const Model = null,
    object: ?[]const u8 = null,
};

// --- Error Types ---

/// API error response wrapper.
pub const ApiErrorResponse = struct {
    @"error": ?ApiErrorDetail = null,
};

/// Details of an API error.
pub const ApiErrorDetail = struct {
    /// Human-readable error message.
    message: ?[]const u8 = null,
    /// Error type (e.g. "invalid_request_error").
    type: ?[]const u8 = null,
    /// Error code.
    code: ?[]const u8 = null,
    /// The parameter that caused the error.
    param: ?[]const u8 = null,
};

// --- Tests ---

test "ChatCompletionRequest serializes to JSON" {
    const messages = [_]Message{.{ .role = .user, .content = "hello" }};
    const req = ChatCompletionRequest{
        .model = "gpt-4o",
        .messages = &messages,
        .temperature = 0.5,
    };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(req, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "gpt-4o") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "temperature") != null);
}

test "ChatCompletionResponse.text extracts text" {
    const json =
        \\{"id":"chatcmpl-123","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}]}
    ;
    const parsed = try std.json.parseFromSlice(
        ChatCompletionResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    try std.testing.expectEqualStrings("Hello!", parsed.value.text().?);
}

test "ChatCompletionResponse parses tool calls" {
    const json =
        \\{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"tool_calls"}]}
    ;
    const parsed = try std.json.parseFromSlice(
        ChatCompletionResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    const tc = parsed.value.firstToolCall().?;
    try std.testing.expectEqualStrings("get_weather", tc.function.?.name.?);
    try std.testing.expectEqualStrings("call_123", tc.id.?);
}

test "FinishReason parses from JSON" {
    const json =
        \\{"choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"length"}]}
    ;
    const parsed = try std.json.parseFromSlice(
        ChatCompletionResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    try std.testing.expect(parsed.value.choices.?[0].finish_reason.? == .length);
}

test "Role serializes correctly" {
    const msg = Message{ .role = .user, .content = "hi" };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(msg, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"user\"") != null);
}
