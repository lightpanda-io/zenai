const std = @import("std");

// --- Enums ---

/// The role of a message in a conversation.
pub const Role = enum {
    user,
    assistant,
};

/// The reason the model stopped generating.
pub const StopReason = enum {
    end_turn,
    max_tokens,
    stop_sequence,
    tool_use,
    pause_turn,
    refusal,
};

// --- Content Blocks ---

/// A content block in a response. Uses a flat struct with a `type` discriminator
/// and optional fields for each block type, matching how Zig's JSON parser handles
/// polymorphic types.
pub const ContentBlock = struct {
    /// Block type: "text", "tool_use", "thinking", "redacted_thinking".
    type: ?[]const u8 = null,
    /// Text content (type="text").
    text: ?[]const u8 = null,
    /// Tool use ID (type="tool_use").
    id: ?[]const u8 = null,
    /// Tool name (type="tool_use").
    name: ?[]const u8 = null,
    /// Tool input as JSON (type="tool_use").
    input: ?std.json.Value = null,
    /// Model thinking/reasoning (type="thinking").
    thinking: ?[]const u8 = null,
    /// Opaque signature for thinking continuation (type="thinking").
    signature: ?[]const u8 = null,
};

/// A content block in a request. Same flat struct approach.
pub const ContentBlockParam = struct {
    /// Block type: "text", "tool_use", "tool_result", "thinking", "redacted_thinking", "image".
    type: []const u8 = "text",
    /// Text content (type="text" or type="tool_result" as inline text).
    text: ?[]const u8 = null,
    /// Tool use ID (type="tool_use").
    id: ?[]const u8 = null,
    /// Tool name (type="tool_use").
    name: ?[]const u8 = null,
    /// Tool input as JSON (type="tool_use").
    input: ?std.json.Value = null,
    /// The tool_use ID this result responds to (type="tool_result").
    tool_use_id: ?[]const u8 = null,
    /// Text content for tool result (type="tool_result").
    content: ?[]const u8 = null,
    /// Whether the tool result is an error (type="tool_result").
    is_error: ?bool = null,
    /// Model thinking/reasoning (type="thinking").
    thinking: ?[]const u8 = null,
    /// Opaque signature for thinking continuation (type="thinking").
    signature: ?[]const u8 = null,
};

/// A text block used for system prompts.
pub const TextBlock = struct {
    type: []const u8 = "text",
    text: []const u8,
};

// --- Messages ---

/// A message in the conversation (for request).
pub const MessageParam = struct {
    /// The role: "user" or "assistant".
    role: Role,
    /// The content blocks.
    content: []const ContentBlockParam,
};

// --- Tools ---

/// A tool the model may use.
pub const Tool = struct {
    /// The tool name.
    name: ?[]const u8 = null,
    /// Description of what the tool does.
    description: ?[]const u8 = null,
    /// JSON Schema for the tool's input parameters.
    input_schema: ?std.json.Value = null,
};

/// Controls how the model uses tools.
pub const ToolChoice = struct {
    /// "auto", "any", "tool", "none".
    type: ?[]const u8 = null,
    /// Tool name (only when type="tool").
    name: ?[]const u8 = null,
};

// --- Thinking Config ---

/// Configuration for extended thinking.
pub const ThinkingConfig = struct {
    /// "enabled", "disabled", "adaptive".
    type: []const u8 = "disabled",
    /// Token budget for thinking (when type="enabled").
    budget_tokens: ?i32 = null,
    /// "summarized", "omitted".
    display: ?[]const u8 = null,
};

// --- Request ---

/// Request body for the messages endpoint.
pub const MessageRequest = struct {
    /// The model to use (e.g. "claude-sonnet-4-6").
    model: []const u8,
    /// The messages in the conversation.
    messages: []const MessageParam,
    /// Maximum number of tokens to generate (required).
    max_tokens: i32,
    /// System prompt (separate from messages).
    system: ?[]const TextBlock = null,
    /// Sampling temperature (0.0-1.0).
    temperature: ?f32 = null,
    /// Top-p (nucleus) sampling (0.0-1.0).
    top_p: ?f32 = null,
    /// Top-k sampling.
    top_k: ?i32 = null,
    /// Custom stop sequences.
    stop_sequences: ?[]const []const u8 = null,
    /// Tools the model may use.
    tools: ?[]const Tool = null,
    /// How the model should use tools.
    tool_choice: ?ToolChoice = null,
    /// Whether to stream the response.
    stream: ?bool = null,
    /// Extended thinking configuration.
    thinking: ?ThinkingConfig = null,
    /// Request metadata.
    metadata: ?Metadata = null,
    /// Top-level cache control — applies an ephemeral cache_control marker to
    /// the last cacheable block in the request (automatic caching).
    cache_control: ?CacheControlEphemeral = null,
};

/// Request metadata.
pub const Metadata = struct {
    /// An external identifier for the user making the request.
    user_id: ?[]const u8 = null,
};

/// Ephemeral cache control breakpoint.
pub const CacheControlEphemeral = struct {
    /// Always "ephemeral".
    type: []const u8 = "ephemeral",
    /// Time-to-live for the cache breakpoint: "5m" or "1h". Defaults to "5m".
    ttl: ?[]const u8 = null,
};

// --- Response ---

/// Read-only breakdown of output tokens by category. `output_tokens` remains
/// the inclusive, authoritative total used for billing; this object decomposes
/// it for observability.
pub const OutputTokensDetails = struct {
    /// Output tokens the model generated as internal reasoning, including the
    /// thinking-block delimiter tokens. Reflects the raw reasoning produced,
    /// not the (possibly shorter) summarized thinking returned in the response,
    /// and is computed by re-tokenizing the raw reasoning so it may differ from
    /// the model's exact generation count by a few tokens. Always
    /// <= `output_tokens`; `output_tokens - thinking_tokens` approximates the
    /// non-reasoning output.
    thinking_tokens: ?i64 = null,
};

/// Token usage statistics.
pub const Usage = struct {
    /// Number of input tokens.
    input_tokens: ?i32 = null,
    /// Number of output tokens.
    output_tokens: ?i32 = null,
    /// Breakdown of output tokens by category (e.g. thinking tokens).
    output_tokens_details: ?OutputTokensDetails = null,
    /// Tokens written to the cache.
    cache_creation_input_tokens: ?i32 = null,
    /// Tokens read from the cache.
    cache_read_input_tokens: ?i32 = null,
};

/// Policy category that triggered a refusal.
pub const RefusalCategory = enum {
    cyber,
    bio,
};

/// Structured information about a refusal.
pub const RefusalStopDetails = struct {
    /// The policy category that triggered the refusal, or null when it doesn't
    /// map to a named category.
    category: ?RefusalCategory = null,
    /// Human-readable explanation of the refusal, or null when unavailable.
    explanation: ?[]const u8 = null,
    /// Always "refusal".
    type: ?[]const u8 = null,
};

/// Response from the messages endpoint.
pub const MessageResponse = struct {
    /// Unique message identifier.
    id: ?[]const u8 = null,
    /// Object type (always "message").
    type: ?[]const u8 = null,
    /// The role (always "assistant").
    role: ?Role = null,
    /// The generated content blocks.
    content: ?[]const ContentBlock = null,
    /// The model used.
    model: ?[]const u8 = null,
    /// Structured information about a refusal (present when stop_reason is "refusal").
    stop_details: ?RefusalStopDetails = null,
    /// Why the model stopped generating.
    stop_reason: ?StopReason = null,
    /// Which stop sequence was matched, if any.
    stop_sequence: ?[]const u8 = null,
    /// Token usage statistics.
    usage: ?Usage = null,

    /// Extract text from the first text content block.
    pub fn text(self: MessageResponse) ?[]const u8 {
        const blocks = self.content orelse return null;
        for (blocks) |block| {
            if (block.type) |t| {
                if (std.mem.eql(u8, t, "text")) {
                    return block.text;
                }
            }
        }
        return null;
    }

    /// Extract the first tool_use content block.
    pub fn firstToolUse(self: MessageResponse) ?ContentBlock {
        const blocks = self.content orelse return null;
        for (blocks) |block| {
            if (block.type) |t| {
                if (std.mem.eql(u8, t, "tool_use")) {
                    return block;
                }
            }
        }
        return null;
    }
};

// --- Streaming ---

/// A streaming delta for content blocks.
pub const StreamDelta = struct {
    /// Delta type: "text_delta", "input_json_delta", "thinking_delta", "signature_delta".
    type: ?[]const u8 = null,
    /// Incremental text (for text_delta).
    text: ?[]const u8 = null,
    /// Incremental JSON (for input_json_delta).
    partial_json: ?[]const u8 = null,
    /// Incremental thinking (for thinking_delta).
    thinking: ?[]const u8 = null,
    /// Per-frame increment of a coarse running estimate of tokens this
    /// thinking block has produced. Emitted only when the
    /// `thinking-token-count-2026-05-13` beta is enabled and the response
    /// would otherwise omit thinking content. Sum across frames for a
    /// progress hint; `usage.output_tokens` remains authoritative.
    estimated_tokens: ?i64 = null,
    /// Incremental signature (for signature_delta).
    signature: ?[]const u8 = null,
    /// Structured refusal details (for message_delta events).
    stop_details: ?RefusalStopDetails = null,
    /// Stop reason (for message_delta events).
    stop_reason: ?StopReason = null,
    /// Stop sequence (for message_delta events).
    stop_sequence: ?[]const u8 = null,
};

/// A streaming event from the messages endpoint.
pub const StreamEvent = struct {
    /// Event type: "message_start", "content_block_start", "content_block_delta",
    /// "content_block_stop", "message_delta", "message_stop".
    type: ?[]const u8 = null,
    /// The full message (message_start events).
    message: ?MessageResponse = null,
    /// The content block delta (content_block_delta events).
    delta: ?StreamDelta = null,
    /// The content block (content_block_start events).
    content_block: ?ContentBlock = null,
    /// Block index (content_block_start/delta/stop events).
    index: ?i32 = null,
    /// Usage update (message_delta events).
    usage: ?Usage = null,
};

// --- Models ---

/// An available Anthropic model.
pub const Model = struct {
    /// Object type (always "model").
    type: ?[]const u8 = null,
    /// Model identifier (e.g. "claude-haiku-4-5-20251001").
    id: ?[]const u8 = null,
    /// Human-readable name.
    display_name: ?[]const u8 = null,
    /// ISO 8601 creation timestamp.
    created_at: ?[]const u8 = null,
};

/// Response from the list models endpoint.
pub const ListModelsResponse = struct {
    data: ?[]const Model = null,
    has_more: ?bool = null,
    first_id: ?[]const u8 = null,
    last_id: ?[]const u8 = null,
};

// --- Tests ---

test "MessageRequest serializes to JSON" {
    const content = [_]ContentBlockParam{.{ .text = "hello" }};
    const messages = [_]MessageParam{.{ .role = .user, .content = &content }};
    const req = MessageRequest{
        .model = "claude-sonnet-4-6",
        .messages = &messages,
        .max_tokens = 1024,
        .temperature = 0.5,
    };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(req, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "claude-sonnet-4-6") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "max_tokens") != null);
}

test "MessageRequest with system prompt serializes correctly" {
    const content = [_]ContentBlockParam{.{ .text = "hello" }};
    const messages = [_]MessageParam{.{ .role = .user, .content = &content }};
    const system = [_]TextBlock{.{ .text = "You are helpful." }};
    const req = MessageRequest{
        .model = "claude-sonnet-4-6",
        .messages = &messages,
        .max_tokens = 1024,
        .system = &system,
    };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(req, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"system\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "You are helpful.") != null);
}

test "MessageResponse.text extracts text" {
    const json =
        \\{"id":"msg_123","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}
    ;
    const parsed = try std.json.parseFromSlice(
        MessageResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    try std.testing.expectEqualStrings("Hello!", parsed.value.text().?);
}

test "MessageResponse parses tool_use" {
    const json =
        \\{"id":"msg_123","type":"message","role":"assistant","content":[{"type":"tool_use","id":"toolu_123","name":"get_weather","input":{"city":"Paris"}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}
    ;
    const parsed = try std.json.parseFromSlice(
        MessageResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    const tu = parsed.value.firstToolUse().?;
    try std.testing.expectEqualStrings("get_weather", tu.name.?);
    try std.testing.expectEqualStrings("toolu_123", tu.id.?);
}

test "StopReason parses from JSON" {
    const json =
        \\{"id":"msg_123","type":"message","role":"assistant","content":[{"type":"text","text":"done"}],"stop_reason":"max_tokens","usage":{"input_tokens":10,"output_tokens":100}}
    ;
    const parsed = try std.json.parseFromSlice(
        MessageResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    try std.testing.expect(parsed.value.stop_reason.? == .max_tokens);
}

test "Usage parses output_tokens_details" {
    const json =
        \\{"id":"msg_123","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":120,"output_tokens_details":{"thinking_tokens":80}}}
    ;
    const parsed = try std.json.parseFromSlice(
        MessageResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    const usage = parsed.value.usage.?;
    try std.testing.expectEqual(@as(i32, 120), usage.output_tokens.?);
    try std.testing.expectEqual(@as(i64, 80), usage.output_tokens_details.?.thinking_tokens.?);
}

test "Role serializes correctly" {
    const content = [_]ContentBlockParam{.{ .text = "hi" }};
    const msg = MessageParam{ .role = .user, .content = &content };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(msg, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"user\"") != null);
}
