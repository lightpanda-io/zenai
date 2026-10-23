const std = @import("std");
const jsonutil = @import("../json.zig");

// --- Enums ---

/// The role of a message in a conversation.
pub const Role = enum {
    system,
    user,
    assistant,
    tool,
};

/// The reason why the model stopped generating tokens. Known values are void
/// tags; any value the API adds later is preserved in `unknown` rather than
/// failing the parse.
pub const FinishReason = union(enum) {
    stop,
    length,
    tool_calls,
    content_filter,
    unknown: []const u8,

    pub const jsonParse = jsonutil.StringUnionMethods(@This()).jsonParse;
    pub const jsonStringify = jsonutil.StringUnionMethods(@This()).jsonStringify;
};

/// The effort level for model reasoning.
pub const ReasoningEffort = enum {
    none,
    minimal,
    low,
    medium,
    high,
    xhigh,
    max,
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
    /// Position of this tool call within the assistant message. Streaming
    /// deltas carry it so argument fragments can be correlated across chunks;
    /// omitted (null) in requests.
    index: ?i32 = null,
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
    /// Multimodal content (request only). Serialized as `content` in place
    /// of the plain text when set.
    content_parts: ?[]const ContentPart = null,
    /// Optional participant name.
    name: ?[]const u8 = null,
    /// Tool calls requested by the assistant (only for role=assistant).
    tool_calls: ?[]const ToolCall = null,
    /// The tool call ID this message responds to (only for role=tool).
    tool_call_id: ?[]const u8 = null,
    /// Refusal message from the model.
    refusal: ?[]const u8 = null,
    /// Reasoning from `thinking` content blocks (response only).
    thinking: ?[]const u8 = null,

    /// Default field parsing, except that `content` may be an array of blocks
    /// (Mistral reasoning models): `text` blocks join into `content`,
    /// `thinking` blocks into `thinking`.
    pub fn jsonParse(
        allocator: std.mem.Allocator,
        source: anytype,
        options: std.json.ParseOptions,
    ) std.json.ParseError(@TypeOf(source.*))!Message {
        if (.object_begin != try source.next()) return error.UnexpectedToken;
        var msg: Message = .{};
        fields: while (true) {
            const name_token = try source.nextAllocMax(allocator, options.allocate.?, options.max_value_len.?);
            const key = switch (name_token) {
                inline .string, .allocated_string => |slice| slice,
                .object_end => break,
                else => return error.UnexpectedToken,
            };
            if (std.mem.eql(u8, key, "content")) {
                switch (try std.json.innerParse(?Content, allocator, source, options) orelse continue) {
                    .text => |t| msg.content = t,
                    .blocks => |blocks| {
                        msg.content = try joinBlocks(allocator, blocks, .text);
                        msg.thinking = try joinBlocks(allocator, blocks, .thinking);
                    },
                }
                continue;
            }
            inline for (@typeInfo(Message).@"struct".field_names, @typeInfo(Message).@"struct".field_types) |name, T| {
                // Not wire fields.
                if (comptime std.mem.eql(u8, name, "content") or
                    std.mem.eql(u8, name, "content_parts") or
                    std.mem.eql(u8, name, "thinking")) continue;
                if (std.mem.eql(u8, key, name)) {
                    @field(msg, name) = try std.json.innerParse(T, allocator, source, options);
                    continue :fields;
                }
            }
            if (!options.ignore_unknown_fields) return error.UnknownField;
            try source.skipValue();
        }
        return msg;
    }

    const Content = union(enum) {
        text: []const u8,
        blocks: []const ContentBlock,

        pub fn jsonParse(
            allocator: std.mem.Allocator,
            source: anytype,
            options: std.json.ParseOptions,
        ) std.json.ParseError(@TypeOf(source.*))!Content {
            return switch (try source.peekNextTokenType()) {
                .array_begin => .{ .blocks = try std.json.innerParse([]const ContentBlock, allocator, source, options) },
                else => .{ .text = try std.json.innerParse([]const u8, allocator, source, options) },
            };
        }
    };

    /// `thinking` blocks nest their reasoning as `text` blocks.
    const ContentBlock = struct {
        type: []const u8,
        text: ?[]const u8 = null,
        thinking: ?[]const ContentBlock = null,
        closed: ?bool = null,
    };

    fn joinBlocks(
        allocator: std.mem.Allocator,
        blocks: []const ContentBlock,
        comptime kind: enum { text, thinking },
    ) std.mem.Allocator.Error!?[]const u8 {
        var out: std.ArrayList(u8) = .empty;
        for (blocks) |b| {
            if (!std.mem.eql(u8, b.type, @tagName(kind))) continue;
            switch (kind) {
                .text => if (b.text) |t| try out.appendSlice(allocator, t),
                .thinking => for (b.thinking orelse continue) |inner| {
                    if (inner.text) |t| try out.appendSlice(allocator, t);
                },
            }
        }
        if (out.items.len == 0) return null;
        return try out.toOwnedSlice(allocator);
    }

    /// Default field emission, except that `content_parts` goes out as
    /// `content` in place of the text, and `thinking` is omitted.
    pub fn jsonStringify(self: Message, jws: anytype) !void {
        try jws.beginObject();
        inline for (@typeInfo(Message).@"struct".field_names) |name| {
            if (comptime std.mem.eql(u8, name, "content_parts") or std.mem.eql(u8, name, "thinking")) {
                // `content_parts` is emitted under `content` below.
            } else if (comptime std.mem.eql(u8, name, "content")) {
                if (self.content_parts) |parts| {
                    try jws.objectField("content");
                    try jws.write(parts);
                } else if (self.content) |v| {
                    try jws.objectField("content");
                    try jws.write(v);
                }
            } else if (@field(self, name)) |v| {
                try jws.objectField(name);
                try jws.write(v);
            }
        }
        try jws.endObject();
    }
};

/// One entry of a multimodal `content` array: `text` or `image_url`.
pub const ContentPart = struct {
    type: []const u8,
    text: ?[]const u8 = null,
    image_url: ?ImageUrl = null,
};

pub const ImageUrl = struct {
    url: DataUrl,
};

/// Serializes as `data:<mime>;base64,<data>` without building the string:
/// the payload is written in place, so an image costs no extra copy.
pub const DataUrl = struct {
    mime_type: []const u8,
    data: []const u8,

    pub fn jsonStringify(self: DataUrl, jws: anytype) !void {
        try jws.beginWriteRaw();
        try jws.writer.writeAll("\"data:");
        try jws.writer.writeAll(self.mime_type);
        try jws.writer.writeAll(";base64,");
        try jws.writer.writeAll(self.data);
        try jws.writer.writeByte('"');
        jws.endWriteRaw();
    }
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

/// Streaming-only request options.
pub const StreamOptions = struct {
    /// Emit a final chunk carrying token usage after the stream completes.
    include_usage: ?bool = null,
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
    /// Upper bound on generated tokens (visible output + reasoning). Replaces
    /// the deprecated `max_tokens`, which o-series and gpt-5 models reject.
    max_completion_tokens: ?i32 = null,
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
    /// Streaming-only options (e.g. request a final usage chunk).
    stream_options: ?StreamOptions = null,
    /// Tools the model may call.
    tools: ?[]const Tool = null,
    /// Controls which tool is called ("none", "auto", "required").
    tool_choice: ?[]const u8 = null,
    /// Random seed for deterministic output.
    seed: ?i32 = null,
    /// The effort level for model reasoning.
    reasoning_effort: ?ReasoningEffort = null,
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
    /// OpenRouter extension.
    cache_control: ?CacheControl = null,
    /// Vercel AI Gateway extension.
    providerOptions: ?ProviderOptions = null,
};

pub const CacheControl = struct {
    type: []const u8 = "ephemeral",
};

pub const ProviderOptions = struct {
    gateway: struct {
        caching: []const u8,
    },
};

// --- Response ---

/// Per-prompt token breakdown. OpenAI auto-caches prompts ≥1024 tokens and
/// bills the cached prefix at a discounted rate; the count shows up here.
pub const PromptTokensDetails = struct {
    cached_tokens: ?i32 = null,
    /// OpenRouter extension.
    cache_write_tokens: ?i32 = null,
    /// Text input tokens present in the prompt.
    text_tokens: ?i32 = null,
    /// Image input tokens present in the prompt.
    image_tokens: ?i32 = null,
};

/// Per-completion token breakdown.
pub const CompletionTokensDetails = struct {
    /// Tokens generated by the model for reasoning.
    reasoning_tokens: ?i32 = null,
    /// Text output tokens generated by the model.
    text_tokens: ?i32 = null,
};

/// Token usage statistics.
pub const Usage = struct {
    /// Tokens in the prompt.
    prompt_tokens: ?i32 = null,
    /// Tokens in the completion.
    completion_tokens: ?i32 = null,
    /// Total tokens used.
    total_tokens: ?i32 = null,
    /// Per-prompt breakdown. Set when the API reports cached tokens.
    prompt_tokens_details: ?PromptTokensDetails = null,
    /// Per-completion breakdown (e.g. reasoning tokens).
    completion_tokens_details: ?CompletionTokensDetails = null,
    /// Vercel AI Gateway extension.
    cache_creation_input_tokens: ?i32 = null,
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
    /// Key-value pairs attached to the completion (up to 16; response only).
    metadata: ?std.json.Value = null,
    /// Padding string added to streamed chunks to normalize their size as a
    /// side-channel mitigation (chunk only). Carries no information; ignore.
    obfuscation: ?[]const u8 = null,

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

// --- Responses API ---
//
// The Responses endpoint (`POST /responses`) is OpenAI's successor to chat
// completions. gpt-5.x rejects function tools combined with reasoning on
// `/chat/completions` and requires this endpoint instead. The conversation is
// a flat list of typed `input` items rather than role-tagged messages, and the
// model replies with a list of typed `output` items.

/// A function tool exposed to the model. Flat shape — unlike chat completions,
/// the function fields are not nested under a `function` object.
pub const ResponseTool = struct {
    type: []const u8 = "function",
    name: []const u8,
    description: ?[]const u8 = null,
    parameters: ?std.json.Value = null,
    strict: ?bool = null,
};

/// Reasoning controls for reasoning-capable models. `summary` (e.g. "auto") is
/// used by the Codex backend to echo reasoning summaries.
pub const ResponseReasoning = struct {
    effort: ReasoningEffort,
    summary: ?[]const u8 = null,
};

/// One item in the request `input` list. The `type` selects which fields are
/// meaningful: a `message` carries `role`/`content`; a `function_call` carries
/// `call_id`/`name`/`arguments`; a `function_call_output` carries
/// `call_id`/`output`.
pub const ResponseInputItem = struct {
    type: []const u8,
    role: ?[]const u8 = null,
    content: ?ResponseContent = null,
    call_id: ?[]const u8 = null,
    name: ?[]const u8 = null,
    arguments: ?[]const u8 = null,
    output: ?[]const u8 = null,
};

/// `message` item content: plain text, or parts when it carries images.
pub const ResponseContent = union(enum) {
    text: []const u8,
    parts: []const ResponseContentPart,

    pub const jsonStringify = jsonutil.PayloadUnionMethods(@This()).jsonStringify;
};

/// `input_text` or `input_image`.
pub const ResponseContentPart = struct {
    type: []const u8,
    text: ?[]const u8 = null,
    image_url: ?DataUrl = null,
};

/// Request body for the responses endpoint.
pub const ResponsesRequest = struct {
    model: []const u8,
    input: []const ResponseInputItem,
    tools: ?[]const ResponseTool = null,
    tool_choice: ?[]const u8 = null,
    reasoning: ?ResponseReasoning = null,
    max_output_tokens: ?i32 = null,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    stream: ?bool = null,
    /// Codex (ChatGPT backend) fields; omitted (null) for the standard OpenAI
    /// endpoint. `store=false` makes the stateless backend accept the request;
    /// `include`/`prompt_cache_key` carry encrypted reasoning across turns.
    store: ?bool = null,
    include: ?[]const []const u8 = null,
    prompt_cache_key: ?[]const u8 = null,
    /// Prompt-cache breakpoint and diagnostics controls.
    prompt_cache_options: ?PromptCacheOptions = null,
};

/// Prompt-caching options for a responses request.
pub const PromptCacheOptions = struct {
    /// A response ID to compare against; requests `prompt_cache_diagnostics`
    /// in the response.
    comparison_response_id: ?[]const u8 = null,
    /// "implicit" (default) or "explicit" (no implicit breakpoint).
    mode: ?[]const u8 = null,
    /// Minimum lifetime of each cache breakpoint; "30m" is the only value.
    ttl: ?[]const u8 = null,
};

/// Why the prompt cache was or wasn't reused against the response named by
/// `prompt_cache_options.comparison_response_id`.
pub const PromptCacheDiagnostics = struct {
    /// "cache_miss", "cache_hit", "comparison_response_not_found", "unavailable".
    type: ?[]const u8 = null,
    /// Why reuse did not occur (type="cache_miss"), e.g. "model_changed",
    /// "tools_changed", "input_changed".
    reason: ?[]const u8 = null,
    /// Estimated input tokens affected after the first divergence (type="cache_miss").
    cache_missed_tokens: ?i64 = null,
    /// Token count of the reusable prefix in the compared response (type="cache_miss").
    comparison_reusable_tokens: ?i64 = null,
};

/// A content block within an output `message` item. `output_text` blocks carry
/// the visible answer in `text`.
pub const ResponseOutputContent = struct {
    type: ?[]const u8 = null,
    text: ?[]const u8 = null,
};

/// One item in the response `output` list. `reasoning` items are summaries we
/// ignore; `message` items carry `content`; `function_call` items carry a tool
/// call to dispatch.
pub const ResponseOutputItem = struct {
    type: ?[]const u8 = null,
    id: ?[]const u8 = null,
    role: ?[]const u8 = null,
    content: ?[]const ResponseOutputContent = null,
    call_id: ?[]const u8 = null,
    name: ?[]const u8 = null,
    arguments: ?[]const u8 = null,
    status: ?[]const u8 = null,
};

pub const ResponseInputTokensDetails = struct {
    cached_tokens: ?i32 = null,
    /// Input tokens written to the cache.
    cache_write_tokens: ?i32 = null,
};

pub const ResponseOutputTokensDetails = struct {
    reasoning_tokens: ?i32 = null,
};

/// Token usage for the responses endpoint. Field names differ from chat
/// completions (`input_tokens`/`output_tokens` rather than
/// `prompt_tokens`/`completion_tokens`).
pub const ResponsesUsage = struct {
    input_tokens: ?i32 = null,
    output_tokens: ?i32 = null,
    total_tokens: ?i32 = null,
    input_tokens_details: ?ResponseInputTokensDetails = null,
    output_tokens_details: ?ResponseOutputTokensDetails = null,
};

/// Why a response stopped short of completion. `reason` is `max_output_tokens`
/// or `content_filter`.
pub const IncompleteDetails = struct {
    reason: ?[]const u8 = null,
};

/// Response from the responses endpoint.
pub const ResponsesResponse = struct {
    id: ?[]const u8 = null,
    object: ?[]const u8 = null,
    status: ?[]const u8 = null,
    output: ?[]const ResponseOutputItem = null,
    usage: ?ResponsesUsage = null,
    incomplete_details: ?IncompleteDetails = null,
    /// Set when the request supplied `prompt_cache_options.comparison_response_id`.
    prompt_cache_diagnostics: ?PromptCacheDiagnostics = null,

    /// First `output_text` across the output items, or null if the model
    /// returned only tool calls / reasoning.
    pub fn text(self: ResponsesResponse) ?[]const u8 {
        const items = self.output orelse return null;
        for (items) |item| {
            const t = item.type orelse continue;
            if (!std.mem.eql(u8, t, "message")) continue;
            const content = item.content orelse continue;
            for (content) |c| if (c.text) |txt| return txt;
        }
        return null;
    }
};

/// A streaming event from the responses endpoint. The `type` field selects
/// which of the other fields is populated: `response.output_text.delta` carries
/// an incremental `delta`; `response.output_item.done` carries a complete
/// `item` (a `function_call` item has its full `arguments`); `response.completed`
/// / `response.incomplete` carry the terminal `response` with usage and status.
pub const ResponseStreamEvent = struct {
    type: ?[]const u8 = null,
    delta: ?[]const u8 = null,
    item: ?ResponseOutputItem = null,
    response: ?ResponsesResponse = null,
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
    /// The embedding vector, decoded from base64 when the request asked for
    /// `encoding_format = "base64"`.
    embedding: ?[]const f32 = null,
    /// The index of this embedding in the input list.
    index: ?i32 = null,
    /// Object type (always "embedding").
    object: ?[]const u8 = null,

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !Embedding {
        const raw = try std.json.innerParse(struct {
            embedding: ?Vector = null,
            index: ?i32 = null,
            object: ?[]const u8 = null,
        }, allocator, source, options);
        return .{ .embedding = if (raw.embedding) |v| v.values else null, .index = raw.index, .object = raw.object };
    }

    /// A JSON array of floats, or a base64 string of little-endian float32s.
    const Vector = struct {
        values: []const f32,

        pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !Vector {
            if (try source.peekNextTokenType() != .string) {
                return .{ .values = try std.json.innerParse([]const f32, allocator, source, options) };
            }
            const encoded = try std.json.innerParse([]const u8, allocator, source, options);
            const decoder = std.base64.standard.Decoder;
            const len = decoder.calcSizeForSlice(encoded) catch return error.UnexpectedToken;
            if (len % 4 != 0) return error.UnexpectedToken;
            const bytes = try allocator.alloc(u8, len);
            decoder.decode(bytes, encoded) catch return error.UnexpectedToken;
            const values = try allocator.alloc(f32, len / 4);
            for (values, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
            return .{ .values = values };
        }
    };
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
    /// The date when the model will shut down ("YYYY-MM-DD"), or null if not
    /// announced.
    shutdown_date: ?[]const u8 = null,
};

/// Response from the list models endpoint.
pub const ListModelsResponse = struct {
    data: ?[]const Model = null,
    object: ?[]const u8 = null,
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
    try std.testing.expect(std.mem.find(u8, json, "hello") != null);
    try std.testing.expect(std.mem.find(u8, json, "gpt-4o") != null);
    try std.testing.expect(std.mem.find(u8, json, "temperature") != null);
}

test "ChatCompletionRequest serializes gateway cache hints" {
    const messages = [_]Message{.{ .role = .user, .content = "hello" }};
    const req: ChatCompletionRequest = .{
        .model = "anthropic/claude-sonnet-5",
        .messages = &messages,
        .cache_control = .{},
        .providerOptions = .{ .gateway = .{ .caching = "auto" } },
    };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(req, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.find(u8, json, "\"cache_control\":{\"type\":\"ephemeral\"}") != null);
    try std.testing.expect(std.mem.find(u8, json, "\"providerOptions\":{\"gateway\":{\"caching\":\"auto\"}}") != null);
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

test "Message parses array content: text answers, thinking set aside" {
    const json =
        \\{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":null,"content":[{"type":"thinking","thinking":[{"type":"text","text":"The user "},{"type":"text","text":"said hi."}],"closed":true},{"type":"text","text":"Hello!"}]},"finish_reason":"stop"}]}
    ;
    const parsed = try std.json.parseFromSlice(ChatCompletionResponse, std.testing.allocator, json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const msg = parsed.value.choices.?[0].message.?;
    try std.testing.expectEqual(Role.assistant, msg.role.?);
    try std.testing.expectEqualStrings("Hello!", parsed.value.text().?);
    try std.testing.expectEqualStrings("The user said hi.", msg.thinking.?);
    try std.testing.expect(msg.tool_calls == null);
}

test "Message parses streamed thinking deltas without content" {
    const json =
        \\{"choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"Hmm"}],"closed":true}]},"finish_reason":null}]}
    ;
    const parsed = try std.json.parseFromSlice(ChatCompletionResponse, std.testing.allocator, json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const delta = parsed.value.choices.?[0].delta.?;
    try std.testing.expect(delta.content == null);
    try std.testing.expectEqualStrings("Hmm", delta.thinking.?);
}

test "Message parses string and null content, rejects unknown fields when asked" {
    const parsed = try std.json.parseFromSlice(Message, std.testing.allocator,
        \\{"role":"assistant","content":null,"refusal":"no"}
    , .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.content == null);
    try std.testing.expectEqualStrings("no", parsed.value.refusal.?);
    try std.testing.expectError(error.UnknownField, std.json.parseFromSlice(Message, std.testing.allocator,
        \\{"content":"hi","extra":1}
    , .{}));
}

test "Message does not serialize thinking" {
    const msg: Message = .{ .role = .assistant, .content = "Hello!", .thinking = "reasoning" };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(msg, .{ .emit_null_optional_fields = false }, &buf.writer);
    try std.testing.expectEqualStrings("{\"role\":\"assistant\",\"content\":\"Hello!\"}", buf.written());
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
    try std.testing.expect(std.meta.activeTag(parsed.value.choices.?[0].finish_reason.?) == .length);
}

test "ResponsesResponse extracts text and tool calls" {
    const json =
        \\{"id":"resp_1","status":"completed","output":[{"type":"reasoning","id":"rs_1","summary":[]},{"type":"function_call","id":"fc_1","call_id":"call_1","name":"goto","arguments":"{\"url\":\"https://news.ycombinator.com\"}"},{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"Done."}]}],"usage":{"input_tokens":42,"output_tokens":7,"total_tokens":49,"input_tokens_details":{"cached_tokens":10}}}
    ;
    const parsed = try std.json.parseFromSlice(
        ResponsesResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    try std.testing.expectEqualStrings("Done.", parsed.value.text().?);
    const items = parsed.value.output.?;
    try std.testing.expectEqualStrings("function_call", items[1].type.?);
    try std.testing.expectEqualStrings("call_1", items[1].call_id.?);
    try std.testing.expectEqualStrings("goto", items[1].name.?);
    try std.testing.expectEqual(@as(i32, 10), parsed.value.usage.?.input_tokens_details.?.cached_tokens.?);
}

test "ResponsesRequest serializes flat tools and reasoning" {
    const input = [_]ResponseInputItem{.{ .type = "message", .role = "user", .content = .{ .text = "hi" } }};
    const tools = [_]ResponseTool{.{ .name = "goto", .description = "navigate" }};
    const req = ResponsesRequest{
        .model = "gpt-5.5",
        .input = &input,
        .tools = &tools,
        .reasoning = .{ .effort = .medium },
        .max_output_tokens = 4096,
    };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(req, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.find(u8, json, "\"max_output_tokens\":4096") != null);
    try std.testing.expect(std.mem.find(u8, json, "\"effort\":\"medium\"") != null);
    // Flat function tool: name is a sibling of type, not nested under "function".
    try std.testing.expect(std.mem.find(u8, json, "\"type\":\"function\",\"name\":\"goto\"") != null);
}

test "Role serializes correctly" {
    const msg = Message{ .role = .user, .content = "hi" };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(msg, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.find(u8, json, "\"user\"") != null);
}

test "EmbeddingResponse parses float and base64 vectors" {
    const json =
        \\{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.5,0.25]},{"object":"embedding","index":1,"embedding":"AACAPwAAIMA="}],"model":"text-embedding-3-small"}
    ;
    const parsed = try std.json.parseFromSlice(EmbeddingResponse, std.testing.allocator, json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const data = parsed.value.data.?;
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.25 }, data[0].embedding.?);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, -2.5 }, data[1].embedding.?);
    try std.testing.expectEqual(@as(i32, 1), data[1].index.?);
}
