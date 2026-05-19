const std = @import("std");
const http = @import("http.zig");
const gemini_mod = @import("gemini/Client.zig");
const openai_mod = @import("openai/Client.zig");
const anthropic_mod = @import("anthropic/Client.zig");
const gemini_types = @import("gemini/types.zig");
const openai_types = @import("openai/types.zig");
const anthropic_types = @import("anthropic/types.zig");

// --- Types ---

pub const Tool = struct {
    name: []const u8,
    description: []const u8,
    /// JSON Schema of the parameters, represented as a std.json.Value.
    parameters: std.json.Value,
};

pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    /// Parsed JSON arguments. Anthropic/Gemini deliver this as a structured
    /// object on the wire; OpenAI delivers it as a JSON string that we parse
    /// once at the boundary so downstream consumers never have to re-parse.
    arguments: ?std.json.Value = null,
    /// Gemini thought signature — must be echoed back with the tool result.
    thought_signature: ?[]const u8 = null,
};

pub const ToolResult = struct {
    id: []const u8,
    name: []const u8, // Required specifically by Gemini
    content: []const u8, // String representation of the result
    /// Signals that the tool call failed. Propagated to Anthropic's native
    /// `is_error` wire field; providers without an equivalent ignore it.
    is_error: bool = false,
    /// Gemini thought signature — echoed from the corresponding ToolCall.
    thought_signature: ?[]const u8 = null,
};

/// Controls how the model uses tools.
pub const ToolChoice = enum {
    /// Model decides whether to call tools.
    auto,
    /// Model must call at least one tool.
    any,
    /// Model must not call tools.
    none,
};

/// Response format hint.
pub const ResponseFormat = enum {
    /// Default unstructured text.
    text,
    /// Model returns valid JSON.
    json,
};

/// A content part for multimodal messages.
pub const ContentPart = union(enum) {
    text: []const u8,
    image: ImageData,
};

/// Inline media data. Despite the name, this carries any inline payload —
/// images, audio, and PDFs — because the wire format (e.g. Gemini
/// `inlineData`) accepts any mime type. `mime_type` drives provider dispatch.
pub const ImageData = struct {
    /// Base64-encoded bytes.
    data: []const u8,
    /// MIME type, e.g. "image/png", "audio/mp3", "application/pdf".
    mime_type: []const u8,
};

/// Return a MIME type string suitable for `ImageData.mime_type`, inferred
/// from a file path's extension. Returns `null` for unknown extensions —
/// callers should surface a clear error rather than guess. The mapping
/// covers the common inputs accepted by multimodal models (images, audio,
/// PDF, plain text).
pub fn inferInlineMimeType(path: []const u8) ?[]const u8 {
    const ext = std.fs.path.extension(path);
    const table = [_]struct { ext: []const u8, mime: []const u8 }{
        .{ .ext = ".png", .mime = "image/png" },
        .{ .ext = ".jpg", .mime = "image/jpeg" },
        .{ .ext = ".jpeg", .mime = "image/jpeg" },
        .{ .ext = ".gif", .mime = "image/gif" },
        .{ .ext = ".webp", .mime = "image/webp" },
        .{ .ext = ".heic", .mime = "image/heic" },
        .{ .ext = ".heif", .mime = "image/heif" },
        .{ .ext = ".mp3", .mime = "audio/mp3" },
        .{ .ext = ".wav", .mime = "audio/wav" },
        .{ .ext = ".ogg", .mime = "audio/ogg" },
        .{ .ext = ".m4a", .mime = "audio/mp4" },
        .{ .ext = ".flac", .mime = "audio/flac" },
        .{ .ext = ".pdf", .mime = "application/pdf" },
        .{ .ext = ".txt", .mime = "text/plain" },
        .{ .ext = ".md", .mime = "text/plain" },
        .{ .ext = ".py", .mime = "text/plain" },
        .{ .ext = ".js", .mime = "text/plain" },
        .{ .ext = ".ts", .mime = "text/plain" },
        .{ .ext = ".json", .mime = "text/plain" },
        .{ .ext = ".csv", .mime = "text/plain" },
        .{ .ext = ".html", .mime = "text/plain" },
        .{ .ext = ".xml", .mime = "text/plain" },
    };
    for (table) |e| if (std.ascii.eqlIgnoreCase(ext, e.ext)) return e.mime;
    return null;
}

test "inferInlineMimeType: extension mapping" {
    const testing = std.testing;
    try testing.expectEqualStrings("image/png", inferInlineMimeType("a/b/foo.png").?);
    try testing.expectEqualStrings("image/jpeg", inferInlineMimeType("FOO.JPG").?);
    try testing.expectEqualStrings("application/pdf", inferInlineMimeType("paper.pdf").?);
    try testing.expectEqualStrings("audio/mp3", inferInlineMimeType("x.mp3").?);
    try testing.expectEqualStrings("text/plain", inferInlineMimeType("src.py").?);
    try testing.expectEqual(@as(?[]const u8, null), inferInlineMimeType("archive.zip"));
    try testing.expectEqual(@as(?[]const u8, null), inferInlineMimeType("noext"));
}

/// A message role, normalized across providers.
pub const Role = enum {
    system,
    user,
    assistant,
    tool,
};

/// A message in a conversation, normalized across providers.
///
/// Conversation invariant: a `.tool` message (carrying `tool_results`) must
/// be immediately preceded by an `.assistant` message whose `tool_calls`
/// it answers. Gemini hard-rejects conversations that violate this; OpenAI
/// and Anthropic are looser but still prefer the same shape. Callers that
/// trim history must respect this — see `safeTruncationStart`.
pub const Message = struct {
    role: Role,
    content: ?[]const u8 = null,
    tool_calls: ?[]const ToolCall = null,
    tool_results: ?[]const ToolResult = null,
    /// Rich content parts (text + images). When set, takes precedence over `content` for
    /// building provider content blocks. Currently only supported by Gemini.
    parts: ?[]const ContentPart = null,
};

/// Smallest index `i` with `i >= min` such that truncating the head of
/// `messages` to `messages[i..]` keeps the conversation valid: the kept
/// slice begins on a `.user` turn, so it can never start with an orphan
/// `.tool` (function response without its function call) or an `.assistant`
/// continuation that has no driving user prompt.
///
/// Returns `null` if no `.user` message exists at or after `min`. Callers
/// that prepend a separately-stored `.system` message can pass the result
/// directly as the start of the kept slice.
pub fn safeTruncationStart(messages: []const Message, min: usize) ?usize {
    var i = min;
    while (i < messages.len) : (i += 1) {
        if (messages[i].role == .user) return i;
    }
    return null;
}

test "safeTruncationStart skips tool and assistant continuations" {
    const msgs = [_]Message{
        .{ .role = .system, .content = "sys" },
        .{ .role = .user, .content = "hi" },
        .{ .role = .assistant, .content = "calling tool" },
        .{ .role = .tool, .content = "tool result" },
        .{ .role = .assistant, .content = "answer" },
        .{ .role = .user, .content = "next" },
    };
    // min lands on `.tool` → walk forward to the next `.user`.
    try std.testing.expectEqual(@as(?usize, 5), safeTruncationStart(&msgs, 3));
    // min already on `.user` → return as-is.
    try std.testing.expectEqual(@as(?usize, 1), safeTruncationStart(&msgs, 1));
    // No user at or after min → null.
    try std.testing.expectEqual(@as(?usize, null), safeTruncationStart(&msgs, 6));
}

/// Deep-copy a slice of `Message`s into `alloc`, including all nested
/// strings, tool calls, tool results, and content parts. Intended for
/// callers that prune conversation history into a fresh arena and need
/// the kept tail to outlive the old arena.
pub fn dupeMessages(alloc: std.mem.Allocator, msgs: []const Message) ![]Message {
    const out = try alloc.alloc(Message, msgs.len);
    for (msgs, 0..) |msg, i| out[i] = try dupeMessage(alloc, msg);
    return out;
}

pub fn dupeMessage(alloc: std.mem.Allocator, msg: Message) !Message {
    return .{
        .role = msg.role,
        .content = if (msg.content) |c| try alloc.dupe(u8, c) else null,
        .tool_calls = if (msg.tool_calls) |tcs| try dupeToolCalls(alloc, tcs) else null,
        .tool_results = if (msg.tool_results) |trs| try dupeToolResults(alloc, trs) else null,
        .parts = if (msg.parts) |ps| try dupeParts(alloc, ps) else null,
    };
}

pub fn dupeToolCalls(alloc: std.mem.Allocator, calls: []const ToolCall) ![]const ToolCall {
    const out = try alloc.alloc(ToolCall, calls.len);
    for (calls, 0..) |tc, i| {
        out[i] = .{
            .id = try alloc.dupe(u8, tc.id),
            .name = try alloc.dupe(u8, tc.name),
            .arguments = if (tc.arguments) |v| try http.dupeJsonValue(alloc, v) else null,
            .thought_signature = if (tc.thought_signature) |ts| try alloc.dupe(u8, ts) else null,
        };
    }
    return out;
}

pub fn dupeToolResults(alloc: std.mem.Allocator, results: []const ToolResult) ![]const ToolResult {
    const out = try alloc.alloc(ToolResult, results.len);
    for (results, 0..) |tr, i| {
        out[i] = .{
            .id = try alloc.dupe(u8, tr.id),
            .name = try alloc.dupe(u8, tr.name),
            .content = try alloc.dupe(u8, tr.content),
            .is_error = tr.is_error,
            .thought_signature = if (tr.thought_signature) |ts| try alloc.dupe(u8, ts) else null,
        };
    }
    return out;
}

pub fn dupeParts(alloc: std.mem.Allocator, parts: []const ContentPart) ![]const ContentPart {
    const out = try alloc.alloc(ContentPart, parts.len);
    for (parts, 0..) |p, i| {
        out[i] = switch (p) {
            .text => |t| .{ .text = try alloc.dupe(u8, t) },
            .image => |img| .{ .image = .{
                .data = try alloc.dupe(u8, img.data),
                .mime_type = try alloc.dupe(u8, img.mime_type),
            } },
        };
    }
    return out;
}

test "dupeMessages: deep-copies content and breaks aliasing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    const src = [_]Message{
        .{ .role = .user, .content = "hello" },
        .{ .role = .assistant, .content = "world" },
    };

    const out = try dupeMessages(arena.allocator(), &src);
    try std.testing.expectEqual(@as(usize, 2), out.len);
    try std.testing.expectEqualStrings("hello", out[0].content.?);
    try std.testing.expectEqualStrings("world", out[1].content.?);
    try std.testing.expect(out[0].content.?.ptr != src[0].content.?.ptr);
}

test "dupeMessages: surfaces OOM without partial mutation of inputs" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    var failing = std.testing.FailingAllocator.init(arena.allocator(), .{ .fail_index = 2 });
    const src = [_]Message{
        .{ .role = .user, .content = "hello" },
        .{ .role = .assistant, .content = "world" },
        .{ .role = .user, .content = "third" },
    };
    try std.testing.expectError(error.OutOfMemory, dupeMessages(failing.allocator(), &src));
    try std.testing.expect(failing.has_induced_failure);
    try std.testing.expectEqualStrings("hello", src[0].content.?);
}

test "dupeToolResults: preserves is_error flag" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    const src = [_]ToolResult{
        .{ .id = "1", .name = "t", .content = "ok" },
        .{ .id = "2", .name = "t", .content = "boom", .is_error = true },
    };
    const out = try dupeToolResults(arena.allocator(), &src);
    try std.testing.expect(!out[0].is_error);
    try std.testing.expect(out[1].is_error);
}

/// Categorical effort level for model reasoning.
pub const ThinkingLevel = enum {
    minimal,
    low,
    medium,
    high,
};

/// Minimal generation config — the intersection of what both providers support.
pub const GenerationConfig = struct {
    temperature: ?f32 = null,
    max_tokens: ?i32 = null,
    top_p: ?f32 = null,
    stop: ?[]const []const u8 = null,
    frequency_penalty: ?f32 = null,
    presence_penalty: ?f32 = null,
    seed: ?i32 = null,
    tools: ?[]const Tool = null,
    tool_choice: ?ToolChoice = null,
    response_format: ?ResponseFormat = null,
    /// Per-turn token budget for model reasoning. Provider-specific: Gemini
    /// thinking models use it for their `thinkingConfig.thinkingBudget` (0
    /// disables thinking). Ignored by OpenAI/Ollama. Null means use the
    /// provider default.
    thinking_budget: ?i32 = null,
    /// Per-turn effort level for model reasoning. Provider-specific: Gemini
    /// 3.5+ thinking models use it for their `thinkingConfig.thinkingLevel`.
    /// Ignored by OpenAI/Ollama. Null means use the provider default.
    thinking_level: ?ThinkingLevel = null,
};

/// Unified finish reason.
pub const FinishReason = enum {
    stop,
    max_tokens,
    tool_call,
    safety,
    unknown,
};

/// Token usage statistics.
pub const Usage = struct {
    prompt_tokens: ?i32 = null,
    completion_tokens: ?i32 = null,
    total_tokens: ?i32 = null,
};

/// Unified generation result.
pub const GenerateResult = struct {
    text: ?[]const u8 = null,
    tool_calls: ?[]const ToolCall = null,
    finish_reason: FinishReason = .unknown,
    usage: Usage = .{},
    /// ArenaAllocator owns all dynamic strings and tool_call slices.
    arena: std.heap.ArenaAllocator,

    pub fn init(allocator: std.mem.Allocator) GenerateResult {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: *GenerateResult) void {
        self.arena.deinit();
    }
};

/// Unified embedding result.
pub const EmbedResult = struct {
    values: ?[]const f32 = null,
    /// ArenaAllocator owns the duped values slice.
    arena: std.heap.ArenaAllocator,

    pub fn init(allocator: std.mem.Allocator) EmbedResult {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: *EmbedResult) void {
        self.arena.deinit();
    }
};

/// Unified AI client. Comptime-dispatched tagged union — no vtable, no runtime overhead.
/// Use this when you want to swap providers with minimal code changes.
/// For provider-specific features, switch on the union tag directly.
pub const Client = union(enum) {
    gemini: *gemini_mod,
    openai: *openai_mod,
    anthropic: *anthropic_mod,
    ollama: *openai_mod,

    pub const Error = gemini_mod.ApiError || openai_mod.ApiError || anthropic_mod.ApiError;
    pub const StreamError = gemini_mod.StreamError || openai_mod.StreamError || anthropic_mod.StreamError;

    fn clientAllocator(self: Client) std.mem.Allocator {
        return switch (self) {
            inline else => |c| c.allocator,
        };
    }

    /// Generate content from a list of messages.
    pub fn generateContent(
        self: Client,
        model: []const u8,
        messages: []const Message,
        config: GenerationConfig,
    ) Error!GenerateResult {
        switch (self) {
            .gemini => |g| {
                var req_arena = std.heap.ArenaAllocator.init(g.allocator);
                defer req_arena.deinit();
                const req_alloc = req_arena.allocator();

                const separated = try separateSystemMessages(req_alloc, messages);
                const contents = separated.contents;
                const sys_instruction: ?gemini_types.Content = if (separated.system_text) |sys|
                    gemini_types.Content{ .parts = @as([]const gemini_types.Part, &.{.{ .text = sys }}) }
                else
                    null;

                const tools = if (config.tools) |t| try mapGeminiTools(req_alloc, t) else null;

                var response = try g.generateContent(model, contents, mapGeminiGenerationConfig(config), .{
                    .systemInstruction = sys_instruction,
                    .tools = tools,
                    .toolConfig = mapToolChoiceToGemini(config.tool_choice),
                });
                defer response.deinit();

                var result = GenerateResult.init(g.allocator);
                errdefer result.deinit();
                if (response.value.text()) |t| {
                    result.text = try result.arena.allocator().dupe(u8, t);
                }
                result.finish_reason = mapGeminiFinishReason(response.value);
                result.usage = mapGeminiUsage(response.value);

                if (response.value.candidates) |candidates| {
                    if (candidates.len > 0) {
                        if (candidates[0].content) |content| {
                            var tool_calls: std.ArrayList(ToolCall) = .empty;
                            for (content.parts) |p| {
                                if (p.functionCall) |fc| {
                                    try tool_calls.append(result.arena.allocator(), .{
                                        .id = if (fc.id) |id| try result.arena.allocator().dupe(u8, id) else "",
                                        .name = if (fc.name) |n| try result.arena.allocator().dupe(u8, n) else "",
                                        .arguments = if (fc.args) |v| try http.dupeJsonValue(result.arena.allocator(), v) else null,
                                        .thought_signature = if (p.thoughtSignature) |ts| try result.arena.allocator().dupe(u8, ts) else null,
                                    });
                                }
                            }
                            if (tool_calls.items.len > 0) {
                                result.tool_calls = try tool_calls.toOwnedSlice(result.arena.allocator());
                            }
                        }
                    }
                }
                return result;
            },
            .openai, .ollama => |o| {
                var req_arena = std.heap.ArenaAllocator.init(o.allocator);
                defer req_arena.deinit();
                const req_alloc = req_arena.allocator();

                const oai_messages = try messagesToOpenAIMessages(req_alloc, messages);
                const tools = if (config.tools) |t| try mapOpenAITools(req_alloc, t) else null;

                var response = try o.chatCompletion(model, oai_messages, mapOpenAICompletionConfig(config, tools));
                defer response.deinit();

                var result = GenerateResult.init(o.allocator);
                errdefer result.deinit();
                if (response.value.text()) |t| {
                    result.text = try result.arena.allocator().dupe(u8, t);
                }
                result.finish_reason = mapOpenAIFinishReason(response.value);
                result.usage = mapOpenAIUsage(response.value);

                if (response.value.choices) |choices| {
                    if (choices.len > 0) {
                        if (choices[0].message) |msg| {
                            if (msg.tool_calls) |calls| {
                                var tool_calls: std.ArrayList(ToolCall) = .empty;
                                for (calls) |call| {
                                    if (call.function) |f| {
                                        const args_val: ?std.json.Value = if (f.arguments) |a|
                                            (std.json.parseFromSliceLeaky(std.json.Value, result.arena.allocator(), a, .{}) catch null)
                                        else
                                            null;
                                        try tool_calls.append(result.arena.allocator(), .{
                                            .id = if (call.id) |id| try result.arena.allocator().dupe(u8, id) else "",
                                            .name = if (f.name) |n| try result.arena.allocator().dupe(u8, n) else "",
                                            .arguments = args_val,
                                        });
                                    }
                                }
                                if (tool_calls.items.len > 0) {
                                    result.tool_calls = try tool_calls.toOwnedSlice(result.arena.allocator());
                                }
                            }
                        }
                    }
                }
                return result;
            },
            .anthropic => |a| {
                var req_arena = std.heap.ArenaAllocator.init(a.allocator);
                defer req_arena.deinit();
                const req_alloc = req_arena.allocator();

                const system_text = try extractSystemText(req_alloc, messages);
                const ant_messages = try messagesToAnthropicMessages(req_alloc, messages);

                const system_blocks: ?[]const anthropic_types.TextBlock = if (system_text) |sys|
                    @as([]const anthropic_types.TextBlock, &.{.{ .text = sys }})
                else
                    null;

                const tools = if (config.tools) |t| try mapAnthropicTools(req_alloc, t) else null;

                var response = try a.createMessage(model, ant_messages, config.max_tokens orelse 4096, .{
                    .system = system_blocks,
                    .temperature = config.temperature,
                    .top_p = config.top_p,
                    .stop_sequences = config.stop,
                    .tools = tools,
                    .tool_choice = mapToolChoiceToAnthropic(config.tool_choice),
                    // Anthropic has no native JSON mode; response_format is ignored.
                });
                defer response.deinit();

                var result = GenerateResult.init(a.allocator);
                errdefer result.deinit();
                if (response.value.text()) |t| {
                    result.text = try result.arena.allocator().dupe(u8, t);
                }
                result.finish_reason = mapAnthropicFinishReason(response.value);
                result.usage = mapAnthropicUsage(response.value);

                if (response.value.content) |blocks| {
                    var tool_calls: std.ArrayList(ToolCall) = .empty;
                    for (blocks) |block| {
                        if (block.type) |t| {
                            if (std.mem.eql(u8, t, "tool_use")) {
                                try tool_calls.append(result.arena.allocator(), .{
                                    .id = if (block.id) |id| try result.arena.allocator().dupe(u8, id) else "",
                                    .name = if (block.name) |n| try result.arena.allocator().dupe(u8, n) else "",
                                    .arguments = if (block.input) |v| try http.dupeJsonValue(result.arena.allocator(), v) else null,
                                });
                            }
                        }
                    }
                    if (tool_calls.items.len > 0) {
                        result.tool_calls = try tool_calls.toOwnedSlice(result.arena.allocator());
                    }
                }
                return result;
            },
        }
    }

    /// Stream generated content from a list of messages.
    pub fn generateContentStream(
        self: Client,
        model: []const u8,
        messages: []const Message,
        config: GenerationConfig,
        context: anytype,
        callback: *const fn (@TypeOf(context), GenerateResult) void,
    ) StreamError!void {
        switch (self) {
            .gemini => |g| {
                var req_arena = std.heap.ArenaAllocator.init(g.allocator);
                defer req_arena.deinit();
                const req_alloc = req_arena.allocator();

                const separated = separateSystemMessages(req_alloc, messages) catch return error.OutOfMemory;
                const contents = separated.contents;
                const sys_instruction: ?gemini_types.Content = if (separated.system_text) |sys|
                    gemini_types.Content{ .parts = @as([]const gemini_types.Part, &.{.{ .text = sys }}) }
                else
                    null;

                const tools = if (config.tools) mapGeminiTools(req_alloc, config.tools.?) catch return error.OutOfMemory else null;

                const Wrapper = struct {
                    fn wrap(ctx: struct { user_ctx: @TypeOf(context), user_cb: *const fn (@TypeOf(context), GenerateResult) void, alloc: std.mem.Allocator }, response: gemini_types.GenerateContentResponse) void {
                        var result = GenerateResult.init(ctx.alloc);
                        defer result.deinit();
                        result.text = response.text();
                        result.finish_reason = mapGeminiFinishReason(response);
                        result.usage = mapGeminiUsage(response);
                        ctx.user_cb(ctx.user_ctx, result);
                    }
                };

                try g.generateContentStream(model, contents, mapGeminiGenerationConfig(config), .{
                    .systemInstruction = sys_instruction,
                    .tools = tools,
                    .toolConfig = mapToolChoiceToGemini(config.tool_choice),
                }, .{ .user_ctx = context, .user_cb = callback, .alloc = g.allocator }, &Wrapper.wrap);
            },
            .openai, .ollama => |o| {
                var req_arena = std.heap.ArenaAllocator.init(o.allocator);
                defer req_arena.deinit();
                const req_alloc = req_arena.allocator();

                const oai_messages = messagesToOpenAIMessages(req_alloc, messages) catch return error.OutOfMemory;
                const tools = if (config.tools) mapOpenAITools(req_alloc, config.tools.?) catch return error.OutOfMemory else null;

                const Wrapper = struct {
                    fn wrap(ctx: struct { user_ctx: @TypeOf(context), user_cb: *const fn (@TypeOf(context), GenerateResult) void, alloc: std.mem.Allocator }, response: openai_types.ChatCompletionResponse) void {
                        var result = GenerateResult.init(ctx.alloc);
                        defer result.deinit();
                        result.text = response.text();
                        result.finish_reason = mapOpenAIFinishReason(response);
                        result.usage = mapOpenAIUsage(response);
                        ctx.user_cb(ctx.user_ctx, result);
                    }
                };

                try o.chatCompletionStream(model, oai_messages, mapOpenAICompletionConfig(config, tools), .{ .user_ctx = context, .user_cb = callback, .alloc = o.allocator }, &Wrapper.wrap);
            },
            .anthropic => |a| {
                var req_arena = std.heap.ArenaAllocator.init(a.allocator);
                defer req_arena.deinit();
                const req_alloc = req_arena.allocator();

                const system_text = extractSystemText(req_alloc, messages) catch return error.OutOfMemory;
                const ant_messages = messagesToAnthropicMessages(req_alloc, messages) catch return error.OutOfMemory;

                const system_blocks: ?[]const anthropic_types.TextBlock = if (system_text) |sys|
                    @as([]const anthropic_types.TextBlock, &.{.{ .text = sys }})
                else
                    null;

                const tools = if (config.tools) mapAnthropicTools(req_alloc, config.tools.?) catch return error.OutOfMemory else null;

                const Wrapper = struct {
                    fn wrap(ctx: struct { user_ctx: @TypeOf(context), user_cb: *const fn (@TypeOf(context), GenerateResult) void, alloc: std.mem.Allocator }, event: anthropic_types.StreamEvent) void {
                        const text_content = if (event.delta) |delta| delta.text else null;
                        var result = GenerateResult.init(ctx.alloc);
                        defer result.deinit();
                        result.text = text_content;
                        if (event.delta) |delta| {
                            if (delta.stop_reason) |reason| {
                                result.finish_reason = mapAnthropicStopReason(reason);
                            }
                        }
                        result.usage = convertAnthropicUsage(event.usage);
                        ctx.user_cb(ctx.user_ctx, result);
                    }
                };

                try a.createMessageStream(model, ant_messages, config.max_tokens orelse 4096, .{
                    .system = system_blocks,
                    .temperature = config.temperature,
                    .top_p = config.top_p,
                    .stop_sequences = config.stop,
                    .tools = tools,
                    .tool_choice = mapToolChoiceToAnthropic(config.tool_choice),
                    // Anthropic has no native JSON mode; response_format is ignored.
                }, .{ .user_ctx = context, .user_cb = callback, .alloc = a.allocator }, &Wrapper.wrap);
            },
        }
    }

    /// Generate an embedding for a text string.
    pub fn embed(
        self: Client,
        model: []const u8,
        text: []const u8,
    ) Error!EmbedResult {
        switch (self) {
            .gemini => |g| {
                var response = try g.embedText(model, text);
                defer response.deinit();
                var result = EmbedResult.init(g.allocator);
                if (response.value.embedding) |e| {
                    if (e.values) |v| {
                        result.values = try result.arena.allocator().dupe(f32, v);
                    }
                }
                return result;
            },
            .openai, .ollama => |o| {
                var response = try o.embedText(model, text);
                defer response.deinit();
                var result = EmbedResult.init(o.allocator);
                if (response.value.data) |data| {
                    if (data.len > 0) {
                        if (data[0].embedding) |v| {
                            result.values = try result.arena.allocator().dupe(f32, v);
                        }
                    }
                }
                return result;
            },
            .anthropic => {
                return error.ApiError;
            },
        }
    }

    // --- Agentic tool-use loop ---

    /// Callback interface for executing tool calls.
    pub const ToolHandler = struct {
        context: *anyopaque,
        callFn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tool_name: []const u8, arguments: ?std.json.Value) Result,

        /// What the tool callback returns: the string content the model should
        /// see, plus whether the call errored. `is_error` is the authoritative
        /// failure signal — the content may legitimately describe an error.
        pub const Result = struct {
            content: []const u8,
            is_error: bool = false,
        };

        pub fn call(self: ToolHandler, allocator: std.mem.Allocator, name: []const u8, args: ?std.json.Value) Result {
            return self.callFn(self.context, allocator, name, args);
        }
    };

    /// Configuration for the agentic tool-use loop.
    pub const RunToolsConfig = struct {
        tools: []const Tool,
        max_turns: u32 = 20,
        /// Maximum total number of tool calls to execute across all turns.
        /// When reached, the loop returns with the results so far.
        max_tool_calls: ?u32 = null,
        max_tokens: ?i32 = 4096,
        tool_choice: ?ToolChoice = .auto,
        temperature: ?f32 = null,
        /// See `GenerationConfig.thinking_budget`. Forwarded per-turn.
        thinking_budget: ?i32 = null,
        /// Polled before each turn, between the LLM response and tool
        /// execution, and after each tool call. Returning true stops the
        /// loop with `cancelled = true` in the result. The in-flight LLM
        /// HTTP request still runs to completion before the check fires
        /// (the loop owns iteration, not the HTTP layer).
        cancel: ?CancelHook = null,

        pub fn cancelRequested(self: RunToolsConfig) bool {
            return if (self.cancel) |c| c.check() else false;
        }
    };

    pub const CancelHook = struct {
        context: *anyopaque,
        checkFn: *const fn (context: *anyopaque) bool,

        pub fn check(self: CancelHook) bool {
            return self.checkFn(self.context);
        }
    };

    /// Information about a tool call that was executed during the loop.
    pub const ToolCallInfo = struct {
        name: []const u8,
        arguments: ?std.json.Value,
        result: []const u8,
        is_error: bool = false,
    };

    /// Result of the agentic tool-use loop.
    pub const RunToolsResult = struct {
        text: ?[]const u8 = null,
        tool_calls_made: []const ToolCallInfo = &.{},
        cancelled: bool = false,
        arena: std.heap.ArenaAllocator,

        pub fn deinit(self: *RunToolsResult) void {
            self.arena.deinit();
        }
    };

    /// Run an agentic tool-use loop: send messages to the LLM, execute any
    /// tool calls via the handler, send results back, repeat until the LLM
    /// responds with text (or max_turns is reached).
    ///
    /// The caller owns `messages`. `list_alloc` is used to grow the messages
    /// ArrayList (must match the allocator used to create it). `data_alloc`
    /// is used to dupe message content strings so they outlive LLM responses.
    pub fn runTools(
        self: Client,
        model: []const u8,
        messages: *std.ArrayListUnmanaged(Message),
        list_alloc: std.mem.Allocator,
        data_alloc: std.mem.Allocator,
        handler: ToolHandler,
        config: RunToolsConfig,
    ) Error!RunToolsResult {
        var result_arena = std.heap.ArenaAllocator.init(self.clientAllocator());
        errdefer result_arena.deinit();
        const ra = result_arena.allocator();

        var all_tool_calls: std.ArrayListUnmanaged(ToolCallInfo) = .empty;

        var turns: u32 = config.max_turns;
        var cancelled = false;
        while (turns > 0) : (turns -= 1) {
            if (config.cancelRequested()) {
                cancelled = true;
                break;
            }
            var gen_result = try self.generateContent(model, messages.items, .{
                .tools = config.tools,
                .max_tokens = config.max_tokens,
                .tool_choice = config.tool_choice,
                .temperature = config.temperature,
                .thinking_budget = config.thinking_budget,
            });
            defer gen_result.deinit();

            if (gen_result.tool_calls) |tool_calls| {
                const duped_calls = try dupeToolCalls(data_alloc, tool_calls);
                try messages.append(list_alloc, .{
                    .role = .assistant,
                    .content = if (gen_result.text) |t| try data_alloc.dupe(u8, t) else null,
                    .tool_calls = duped_calls,
                });

                if (config.cancelRequested()) {
                    cancelled = true;
                    break;
                }

                var tool_results: std.ArrayListUnmanaged(ToolResult) = .empty;
                var limit_reached = false;
                for (tool_calls) |tc| {
                    if (config.max_tool_calls) |limit| {
                        if (all_tool_calls.items.len >= limit) {
                            limit_reached = true;
                            break;
                        }
                    }

                    var tool_arena = std.heap.ArenaAllocator.init(self.clientAllocator());
                    defer tool_arena.deinit();

                    const handler_result = handler.call(tool_arena.allocator(), tc.name, tc.arguments);

                    try tool_results.append(data_alloc, .{
                        .id = try data_alloc.dupe(u8, tc.id),
                        .name = try data_alloc.dupe(u8, tc.name),
                        .content = try data_alloc.dupe(u8, handler_result.content),
                        .is_error = handler_result.is_error,
                        .thought_signature = if (tc.thought_signature) |ts| try data_alloc.dupe(u8, ts) else null,
                    });

                    try all_tool_calls.append(ra, .{
                        .name = try ra.dupe(u8, tc.name),
                        .arguments = if (tc.arguments) |v| try http.dupeJsonValue(ra, v) else null,
                        .result = try ra.dupe(u8, handler_result.content),
                        .is_error = handler_result.is_error,
                    });

                    if (config.cancelRequested()) {
                        cancelled = true;
                        break;
                    }
                }

                if (tool_results.items.len > 0) {
                    try messages.append(list_alloc, .{
                        .role = .tool,
                        .tool_results = try tool_results.toOwnedSlice(data_alloc),
                    });
                }

                if (cancelled or limit_reached) break;
                continue;
            }

            const text = if (gen_result.text) |t| try data_alloc.dupe(u8, t) else null;

            if (text != null) {
                try messages.append(list_alloc, .{
                    .role = .assistant,
                    .content = text,
                });
            }

            return .{
                .text = text,
                .tool_calls_made = all_tool_calls.toOwnedSlice(ra) catch &.{},
                .arena = result_arena,
            };
        }

        return .{
            .text = null,
            .tool_calls_made = all_tool_calls.toOwnedSlice(ra) catch &.{},
            .cancelled = cancelled,
            .arena = result_arena,
        };
    }
};

/// Provider tag used in switches and helper APIs.
pub const ProviderKind = std.meta.Tag(Client);

/// Look up the API key for `kind` from the conventional env var(s). Ollama
/// has no key; returns the literal `"ollama"` so OpenAI-shaped clients
/// clear their non-empty-key check.
pub fn envApiKey(kind: ProviderKind) ?[:0]const u8 {
    return switch (kind) {
        .anthropic => std.posix.getenv("ANTHROPIC_API_KEY"),
        .openai => std.posix.getenv("OPENAI_API_KEY"),
        .gemini => std.posix.getenv("GOOGLE_API_KEY") orelse std.posix.getenv("GEMINI_API_KEY"),
        .ollama => "ollama",
    };
}

/// Fetch chat-capable model IDs for `kind`, allocated in `arena`. Ordering
/// is provider-defined — sort at the call site if needed. `base_url_override`
/// is only honored for openai/ollama.
pub fn listChatModelIds(
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    kind: ProviderKind,
    api_key: [:0]const u8,
    base_url_override: ?[:0]const u8,
) ![][]const u8 {
    var ids: std.ArrayList([]const u8) = .empty;

    switch (kind) {
        .anthropic => {
            var client = anthropic_mod.init(allocator, api_key, .{});
            defer client.deinit();
            var resp = try client.listModels();
            defer resp.deinit();
            for (resp.value.data orelse &.{}) |m| {
                if (!anthropic_mod.isChatModel(m)) continue;
                if (m.id) |id| try ids.append(arena, try arena.dupe(u8, id));
            }
        },
        .openai => {
            var client = openai_mod.init(allocator, api_key, if (base_url_override) |u| .{ .base_url = u } else .{});
            defer client.deinit();
            var resp = try client.listModels();
            defer resp.deinit();
            for (resp.value.data orelse &.{}) |m| {
                if (!openai_mod.isChatModel(m)) continue;
                if (m.id) |id| try ids.append(arena, try arena.dupe(u8, id));
            }
        },
        .ollama => {
            const opts: openai_mod.InitOptions = if (base_url_override) |u|
                .{ .base_url = u }
            else
                .{ .base_url = "http://localhost:11434/v1" };
            var client = openai_mod.init(allocator, api_key, opts);
            defer client.deinit();
            var resp = try client.listModels();
            defer resp.deinit();
            // Local catalogs don't follow the naming convention isChatModel
            // expects, so we don't filter.
            for (resp.value.data orelse &.{}) |m| {
                if (m.id) |id| try ids.append(arena, try arena.dupe(u8, id));
            }
        },
        .gemini => {
            var client = gemini_mod.init(allocator, api_key, .{});
            defer client.deinit();
            var resp = try client.listModels(.{});
            defer resp.deinit();
            for (resp.value.models orelse &.{}) |m| {
                if (!gemini_mod.isChatModel(m)) continue;
                const name = m.name orelse continue;
                // Gemini returns "models/<id>"; strip so the output is
                // pipe-ready into a `--model` flag.
                const stripped = if (std.mem.startsWith(u8, name, "models/")) name["models/".len..] else name;
                try ids.append(arena, try arena.dupe(u8, stripped));
            }
        },
    }

    return ids.toOwnedSlice(arena);
}

test "envApiKey: ollama returns placeholder regardless of env" {
    try std.testing.expectEqualStrings("ollama", envApiKey(.ollama).?);
}

// --- Conversion helpers ---

/// Extract and concatenate all system messages into a single text string.
fn extractSystemText(allocator: std.mem.Allocator, messages: []const Message) !?[]const u8 {
    var sys_parts: std.ArrayList([]const u8) = .empty;
    defer sys_parts.deinit(allocator);
    for (messages) |msg| {
        if (msg.role == .system) {
            if (msg.content) |c| try sys_parts.append(allocator, c);
        }
    }
    if (sys_parts.items.len == 0) return null;
    if (sys_parts.items.len == 1) return sys_parts.items[0];
    return try std.mem.join(allocator, "\n\n", sys_parts.items);
}

const SeparatedMessages = struct {
    contents: []gemini_types.Content,
    system_text: ?[]const u8,
};

fn separateSystemMessages(allocator: std.mem.Allocator, messages: []const Message) !SeparatedMessages {
    const system_text = try extractSystemText(allocator, messages);

    var non_system_count: usize = 0;
    for (messages) |msg| {
        if (msg.role != .system) non_system_count += 1;
    }

    const contents = try allocator.alloc(gemini_types.Content, non_system_count);
    var idx: usize = 0;
    for (messages) |msg| {
        if (msg.role == .system) continue;
        const role: []const u8 = switch (msg.role) {
            .user, .tool => "user",
            .assistant => "model",
            .system => unreachable,
        };

        var parts: std.ArrayList(gemini_types.Part) = .empty;

        if (msg.parts) |content_parts| {
            for (content_parts) |cp| {
                switch (cp) {
                    .text => |t| try parts.append(allocator, .{ .text = t }),
                    .image => |img| try parts.append(allocator, .{ .inlineData = .{
                        .data = img.data,
                        .mimeType = img.mime_type,
                    } }),
                }
            }
        } else if (msg.content) |c| {
            try parts.append(allocator, .{ .text = c });
        }

        if (msg.tool_calls) |calls| {
            for (calls) |call| {
                try parts.append(allocator, .{
                    .functionCall = .{
                        .id = call.id,
                        .name = call.name,
                        .args = call.arguments,
                    },
                    .thoughtSignature = call.thought_signature,
                });
            }
        }

        if (msg.tool_results) |results| {
            for (results) |res| {
                var response_val: ?std.json.Value = null;
                if (res.content.len > 0) {
                    const parsed = std.json.parseFromSlice(std.json.Value, allocator, res.content, .{}) catch null;
                    if (parsed) |p| {
                        // Gemini requires response to be a JSON object (Struct)
                        if (p.value == .object) {
                            response_val = p.value;
                        } else {
                            var obj = std.json.ObjectMap.init(allocator);
                            try obj.put("result", p.value);
                            response_val = std.json.Value{ .object = obj };
                        }
                    } else {
                        var obj = std.json.ObjectMap.init(allocator);
                        try obj.put("result", std.json.Value{ .string = res.content });
                        response_val = std.json.Value{ .object = obj };
                    }
                }
                try parts.append(allocator, .{
                    .functionResponse = .{
                        .id = res.id,
                        .name = res.name,
                        .response = response_val,
                    },
                    .thoughtSignature = res.thought_signature,
                });
            }
        }

        contents[idx] = .{ .role = role, .parts = try parts.toOwnedSlice(allocator) };
        idx += 1;
    }

    return .{ .contents = contents, .system_text = system_text };
}

fn messagesToOpenAIMessages(allocator: std.mem.Allocator, messages: []const Message) ![]openai_types.Message {
    var out: std.ArrayList(openai_types.Message) = .empty;

    for (messages) |msg| {
        if (msg.role == .tool) {
            if (msg.tool_results) |results| {
                for (results) |res| {
                    try out.append(allocator, .{
                        .role = .tool,
                        .content = res.content,
                        .tool_call_id = res.id,
                    });
                }
            } else {
                try out.append(allocator, .{
                    .role = .tool,
                    .content = msg.content,
                });
            }
            continue;
        }

        var oai_tool_calls: ?[]const openai_types.ToolCall = null;
        if (msg.tool_calls) |calls| {
            var oai_calls = try allocator.alloc(openai_types.ToolCall, calls.len);
            for (calls, 0..) |call, i| {
                const args_str: []const u8 = if (call.arguments) |v|
                    try http.jsonValueToString(allocator, v)
                else
                    "";
                oai_calls[i] = .{
                    .id = call.id,
                    .type = "function",
                    .function = .{
                        .name = call.name,
                        .arguments = args_str,
                    },
                };
            }
            oai_tool_calls = oai_calls;
        }

        const text_content: ?[]const u8 = if (msg.parts) |content_parts| blk: {
            for (content_parts) |cp| {
                switch (cp) {
                    .text => |t| break :blk t,
                    .image => {},
                }
            }
            break :blk null;
        } else msg.content;

        try out.append(allocator, .{
            .role = switch (msg.role) {
                .system => .system,
                .user => .user,
                .assistant => .assistant,
                .tool => unreachable,
            },
            .content = text_content,
            .tool_calls = oai_tool_calls,
        });
    }

    return out.toOwnedSlice(allocator);
}

fn messagesToAnthropicMessages(allocator: std.mem.Allocator, messages: []const Message) ![]anthropic_types.MessageParam {
    var out: std.ArrayList(anthropic_types.MessageParam) = .empty;
    for (messages) |msg| {
        if (msg.role == .system) continue;

        var blocks: std.ArrayList(anthropic_types.ContentBlockParam) = .empty;

        if (msg.parts) |content_parts| {
            for (content_parts) |cp| {
                switch (cp) {
                    .text => |t| try blocks.append(allocator, .{ .type = "text", .text = t }),
                    .image => {},
                }
            }
        } else if (msg.content) |c| {
            try blocks.append(allocator, .{ .type = "text", .text = c });
        }

        if (msg.tool_calls) |calls| {
            for (calls) |call| {
                try blocks.append(allocator, .{
                    .type = "tool_use",
                    .id = call.id,
                    .name = call.name,
                    .input = call.arguments,
                });
            }
        }

        if (msg.tool_results) |results| {
            for (results) |res| {
                try blocks.append(allocator, .{
                    .type = "tool_result",
                    .tool_use_id = res.id,
                    .content = res.content,
                    .is_error = if (res.is_error) true else null,
                });
            }
        }

        const role: anthropic_types.Role = if (msg.role == .assistant) .assistant else .user;
        try out.append(allocator, .{ .role = role, .content = try blocks.toOwnedSlice(allocator) });
    }
    return out.toOwnedSlice(allocator);
}

fn mapOpenAITools(allocator: std.mem.Allocator, tools: []const Tool) ![]openai_types.Tool {
    const out = try allocator.alloc(openai_types.Tool, tools.len);
    for (tools, 0..) |t, i| {
        out[i] = .{
            .type = "function",
            .function = .{
                .name = t.name,
                .description = t.description,
                .parameters = t.parameters,
            },
        };
    }
    return out;
}

fn mapAnthropicTools(allocator: std.mem.Allocator, tools: []const Tool) ![]anthropic_types.Tool {
    const out = try allocator.alloc(anthropic_types.Tool, tools.len);
    for (tools, 0..) |t, i| {
        out[i] = .{
            .name = t.name,
            .description = t.description,
            .input_schema = t.parameters,
        };
    }
    return out;
}

fn mapGeminiTools(allocator: std.mem.Allocator, tools: []const Tool) ![]gemini_types.Tool {
    const out = try allocator.alloc(gemini_types.Tool, 1);
    const funcs = try allocator.alloc(gemini_types.FunctionDeclaration, tools.len);
    for (tools, 0..) |t, i| {
        funcs[i] = .{
            .name = t.name,
            .description = t.description,
            .parameters = try jsonValueToGeminiSchema(allocator, t.parameters),
        };
    }
    out[0] = .{
        .functionDeclarations = funcs,
    };
    return out;
}

fn jsonValueToGeminiSchema(allocator: std.mem.Allocator, val: std.json.Value) !gemini_types.Schema {
    var schema = gemini_types.Schema{};
    switch (val) {
        .object => |obj| {
            if (obj.get("type")) |t| {
                if (t == .string) {
                    if (std.mem.eql(u8, t.string, "object")) schema.type = .OBJECT;
                    if (std.mem.eql(u8, t.string, "array")) schema.type = .ARRAY;
                    if (std.mem.eql(u8, t.string, "string")) schema.type = .STRING;
                    if (std.mem.eql(u8, t.string, "number")) schema.type = .NUMBER;
                    if (std.mem.eql(u8, t.string, "integer")) schema.type = .INTEGER;
                    if (std.mem.eql(u8, t.string, "boolean")) schema.type = .BOOLEAN;
                }
            }
            if (obj.get("description")) |desc| {
                if (desc == .string) schema.description = desc.string;
            }
            if (obj.get("required")) |req| {
                if (req == .array) {
                    var req_arr = try allocator.alloc([]const u8, req.array.items.len);
                    for (req.array.items, 0..) |item, i| {
                        if (item == .string) req_arr[i] = item.string;
                    }
                    schema.required = req_arr;
                }
            }
            if (obj.get("properties")) |props| {
                if (props == .object) {
                    var prop_arr = try allocator.alloc(gemini_types.Property, props.object.count());
                    var iter = props.object.iterator();
                    var i: usize = 0;
                    while (iter.next()) |entry| {
                        prop_arr[i] = .{
                            .key = entry.key_ptr.*,
                            .value = try jsonValueToGeminiSchema(allocator, entry.value_ptr.*),
                        };
                        i += 1;
                    }
                    schema.properties = prop_arr;
                }
            }
            if (obj.get("items")) |items| {
                const items_ptr = try allocator.create(gemini_types.Schema);
                items_ptr.* = try jsonValueToGeminiSchema(allocator, items);
                schema.items = items_ptr;
            }
            if (obj.get("enum")) |enum_val| {
                if (enum_val == .array) {
                    var enum_arr = try allocator.alloc([]const u8, enum_val.array.items.len);
                    for (enum_val.array.items, 0..) |item, i| {
                        if (item == .string) enum_arr[i] = item.string;
                    }
                    schema.@"enum" = enum_arr;
                }
            }
        },
        else => {},
    }
    return schema;
}

fn mapToolChoiceToGemini(choice: ?ToolChoice) ?gemini_types.ToolConfig {
    const tc = choice orelse return null;
    return .{ .functionCallingConfig = .{ .mode = switch (tc) {
        .auto => .AUTO,
        .any => .ANY,
        .none => .NONE,
    } } };
}

fn mapToolChoiceToOpenAI(choice: ?ToolChoice) ?[]const u8 {
    const tc = choice orelse return null;
    return switch (tc) {
        .auto => "auto",
        .any => "required",
        .none => "none",
    };
}

fn mapToolChoiceToAnthropic(choice: ?ToolChoice) ?anthropic_types.ToolChoice {
    const tc = choice orelse return null;
    return .{ .type = switch (tc) {
        .auto => "auto",
        .any => "any",
        .none => "none",
    } };
}

fn mapResponseFormatToGemini(fmt: ?ResponseFormat) ?[]const u8 {
    const f = fmt orelse return null;
    return switch (f) {
        .json => "application/json",
        .text => null,
    };
}

fn mapResponseFormatToOpenAI(fmt: ?ResponseFormat) ?openai_types.ResponseFormat {
    const f = fmt orelse return null;
    return switch (f) {
        .json => .{ .type = "json_object" },
        .text => null,
    };
}

fn mapGeminiFinishReason(response: gemini_types.GenerateContentResponse) FinishReason {
    const candidates = response.candidates orelse return .unknown;
    if (candidates.len == 0) return .unknown;
    const reason = candidates[0].finishReason orelse return .unknown;
    return switch (reason) {
        .STOP => .stop,
        .MAX_TOKENS => .max_tokens,
        .SAFETY, .RECITATION, .PROHIBITED_CONTENT, .SPII, .BLOCKLIST => .safety,
        .MALFORMED_FUNCTION_CALL => .tool_call,
        else => .unknown,
    };
}

fn mapOpenAIFinishReason(response: openai_types.ChatCompletionResponse) FinishReason {
    const choices = response.choices orelse return .unknown;
    if (choices.len == 0) return .unknown;
    const reason = choices[0].finish_reason orelse return .unknown;
    return switch (reason) {
        .stop => .stop,
        .length => .max_tokens,
        .tool_calls => .tool_call,
        .content_filter => .safety,
    };
}

fn mapGeminiUsage(response: gemini_types.GenerateContentResponse) Usage {
    const meta = response.usageMetadata orelse return .{};
    return .{
        .prompt_tokens = meta.promptTokenCount,
        .completion_tokens = meta.candidatesTokenCount,
        .total_tokens = meta.totalTokenCount,
    };
}

fn mapOpenAIUsage(response: openai_types.ChatCompletionResponse) Usage {
    const usage = response.usage orelse return .{};
    return .{
        .prompt_tokens = usage.prompt_tokens,
        .completion_tokens = usage.completion_tokens,
        .total_tokens = usage.total_tokens,
    };
}

fn mapAnthropicStopReason(reason: anthropic_types.StopReason) FinishReason {
    return switch (reason) {
        .end_turn, .stop_sequence => .stop,
        .max_tokens, .pause_turn => .max_tokens,
        .tool_use => .tool_call,
        .refusal => .safety,
    };
}

fn mapAnthropicFinishReason(response: anthropic_types.MessageResponse) FinishReason {
    const reason = response.stop_reason orelse return .unknown;
    return mapAnthropicStopReason(reason);
}

fn mapAnthropicUsage(response: anthropic_types.MessageResponse) Usage {
    return convertAnthropicUsage(response.usage);
}

fn convertAnthropicUsage(usage_opt: ?anthropic_types.Usage) Usage {
    const usage = usage_opt orelse return .{};
    return .{
        .prompt_tokens = usage.input_tokens,
        .completion_tokens = usage.output_tokens,
        .total_tokens = if (usage.input_tokens != null and usage.output_tokens != null)
            usage.input_tokens.? + usage.output_tokens.?
        else
            null,
    };
}

fn mapGeminiGenerationConfig(config: GenerationConfig) gemini_types.GenerationConfig {
    return .{
        .temperature = config.temperature,
        .maxOutputTokens = config.max_tokens,
        .topP = config.top_p,
        .stopSequences = config.stop,
        .frequencyPenalty = config.frequency_penalty,
        .presencePenalty = config.presence_penalty,
        .seed = config.seed,
        .responseMimeType = mapResponseFormatToGemini(config.response_format),
        .thinkingConfig = if (config.thinking_budget != null or config.thinking_level != null)
            gemini_types.ThinkingConfig{
                .thinkingBudget = config.thinking_budget,
                .thinkingLevel = if (config.thinking_level) |tl| mapThinkingLevelToGemini(tl) else null,
            }
        else
            null,
    };
}

fn mapThinkingLevelToGemini(level: ThinkingLevel) gemini_types.ThinkingLevel {
    return switch (level) {
        .minimal => .MINIMAL,
        .low => .LOW,
        .medium => .MEDIUM,
        .high => .HIGH,
    };
}

fn mapOpenAICompletionConfig(config: GenerationConfig, tools: ?[]const openai_types.Tool) openai_mod.ChatCompletionConfig {
    return .{
        .temperature = config.temperature,
        .max_tokens = config.max_tokens,
        .top_p = config.top_p,
        .stop = config.stop,
        .frequency_penalty = config.frequency_penalty,
        .presence_penalty = config.presence_penalty,
        .seed = config.seed,
        .tools = tools,
        .tool_choice = mapToolChoiceToOpenAI(config.tool_choice),
        .response_format = mapResponseFormatToOpenAI(config.response_format),
    };
}

// --- Tests ---

test "Message and Role basics" {
    const msg = Message{ .role = .user, .content = "hello" };
    try std.testing.expect(msg.role == .user);
    try std.testing.expectEqualStrings("hello", msg.content.?);
}

test "GenerateResult deinit with no backing response" {
    var result = GenerateResult.init(std.testing.allocator);
    result.text = "test";
    result.finish_reason = .stop;
    result.deinit(); // Should not crash
}
