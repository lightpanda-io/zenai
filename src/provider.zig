const std = @import("std");
const gemini_mod = @import("gemini/Client.zig");
const openai_mod = @import("openai/Client.zig");
const anthropic_mod = @import("anthropic/Client.zig");
const gemini_types = @import("gemini/types.zig");
const openai_types = @import("openai/types.zig");
const anthropic_types = @import("anthropic/types.zig");
const http = @import("http.zig");

/// A message role, normalized across providers.
pub const Role = enum {
    system,
    user,
    assistant,
    tool,
};

/// A message in a conversation, normalized across providers.
pub const Message = struct {
    role: Role,
    content: ?[]const u8 = null,
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
    finish_reason: FinishReason = .unknown,
    usage: Usage = .{},
    allocator: std.mem.Allocator,
    /// Backing memory for owned text — null when text borrows from Response.
    _owned_text: ?[]u8 = null,
    /// Provider response backing memory.
    _gemini_response: ?GeminiResponse = null,
    _openai_response: ?OpenAIResponse = null,
    _anthropic_response: ?AnthropicResponse = null,

    const GeminiResponse = http.Response(gemini_types.GenerateContentResponse);
    const OpenAIResponse = http.Response(openai_types.ChatCompletionResponse);
    const AnthropicResponse = http.Response(anthropic_types.MessageResponse);

    pub fn deinit(self: *GenerateResult) void {
        if (self._gemini_response) |*r| r.deinit();
        if (self._openai_response) |*r| r.deinit();
        if (self._anthropic_response) |*r| r.deinit();
    }
};

/// Unified embedding result.
pub const EmbedResult = struct {
    values: ?[]const f32 = null,
    _gemini_response: ?http.Response(gemini_types.EmbedContentResponse) = null,
    _openai_response: ?http.Response(openai_types.EmbeddingResponse) = null,

    pub fn deinit(self: *EmbedResult) void {
        if (self._gemini_response) |*r| r.deinit();
        if (self._openai_response) |*r| r.deinit();
    }
};

/// Unified AI client. Comptime-dispatched tagged union — no vtable, no runtime overhead.
/// Use this when you want to swap providers with minimal code changes.
/// For provider-specific features, use `asGemini()` / `asOpenAI()` to drop down.
pub const Client = union(enum) {
    gemini: *gemini_mod,
    openai: *openai_mod,
    anthropic: *anthropic_mod,

    pub const Error = gemini_mod.ApiError || openai_mod.ApiError || anthropic_mod.ApiError;
    pub const StreamError = gemini_mod.StreamError || openai_mod.StreamError || anthropic_mod.StreamError;

    /// Generate content from a list of messages.
    pub fn generateContent(
        self: Client,
        model: []const u8,
        messages: []const Message,
        config: GenerationConfig,
    ) Error!GenerateResult {
        switch (self) {
            .gemini => |g| {
                // Convert messages to Gemini Content format
                const contents = try messagesToGeminiContents(g.allocator, messages);
                defer g.allocator.free(contents);

                var response = try g.generateContent(model, contents, gemini_types.GenerationConfig{
                    .temperature = config.temperature,
                    .maxOutputTokens = config.max_tokens,
                    .topP = config.top_p,
                    .stopSequences = config.stop,
                    .frequencyPenalty = config.frequency_penalty,
                    .presencePenalty = config.presence_penalty,
                    .seed = config.seed,
                }, .{});

                return GenerateResult{
                    .text = response.value.text(),
                    .finish_reason = mapGeminiFinishReason(response.value),
                    .usage = mapGeminiUsage(response.value),
                    .allocator = g.allocator,
                    ._gemini_response = response,
                };
            },
            .openai => |o| {
                // Convert messages to OpenAI Message format
                const oai_messages = try messagesToOpenAIMessages(o.allocator, messages);
                defer o.allocator.free(oai_messages);

                var response = try o.chatCompletion(model, oai_messages, .{
                    .temperature = config.temperature,
                    .max_tokens = config.max_tokens,
                    .top_p = config.top_p,
                    .stop = config.stop,
                    .frequency_penalty = config.frequency_penalty,
                    .presence_penalty = config.presence_penalty,
                    .seed = config.seed,
                });

                return GenerateResult{
                    .text = response.value.text(),
                    .finish_reason = mapOpenAIFinishReason(response.value),
                    .usage = mapOpenAIUsage(response.value),
                    .allocator = o.allocator,
                    ._openai_response = response,
                };
            },
            .anthropic => |a| {
                const ant_messages = try messagesToAnthropicMessages(a.allocator, messages);
                defer a.allocator.free(ant_messages);

                const response = try a.createMessage(model, ant_messages, config.max_tokens orelse 4096, .{
                    .temperature = config.temperature,
                    .top_p = config.top_p,
                    .stop_sequences = config.stop,
                });

                return GenerateResult{
                    .text = response.value.text(),
                    .finish_reason = mapAnthropicFinishReason(response.value),
                    .usage = mapAnthropicUsage(response.value),
                    .allocator = a.allocator,
                    ._anthropic_response = response,
                };
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
                const contents = messagesToGeminiContents(g.allocator, messages) catch return error.OutOfMemory;
                defer g.allocator.free(contents);

                const Wrapper = struct {
                    fn wrap(ctx: struct { user_ctx: @TypeOf(context), user_cb: *const fn (@TypeOf(context), GenerateResult) void, alloc: std.mem.Allocator }, response: gemini_types.GenerateContentResponse) void {
                        var result = GenerateResult{
                            .text = response.text(),
                            .finish_reason = mapGeminiFinishReason(response),
                            .allocator = ctx.alloc,
                        };
                        ctx.user_cb(ctx.user_ctx, result);
                        _ = &result;
                    }
                };

                try g.generateContentStream(model, contents, gemini_types.GenerationConfig{
                    .temperature = config.temperature,
                    .maxOutputTokens = config.max_tokens,
                    .topP = config.top_p,
                    .stopSequences = config.stop,
                    .frequencyPenalty = config.frequency_penalty,
                    .presencePenalty = config.presence_penalty,
                    .seed = config.seed,
                }, .{}, .{ .user_ctx = context, .user_cb = callback, .alloc = g.allocator }, &Wrapper.wrap);
            },
            .openai => |o| {
                const oai_messages = messagesToOpenAIMessages(o.allocator, messages) catch return error.OutOfMemory;
                defer o.allocator.free(oai_messages);

                const Wrapper = struct {
                    fn wrap(ctx: struct { user_ctx: @TypeOf(context), user_cb: *const fn (@TypeOf(context), GenerateResult) void, alloc: std.mem.Allocator }, response: openai_types.ChatCompletionResponse) void {
                        var result = GenerateResult{
                            .text = response.text(),
                            .finish_reason = mapOpenAIFinishReason(response),
                            .allocator = ctx.alloc,
                        };
                        ctx.user_cb(ctx.user_ctx, result);
                        _ = &result;
                    }
                };

                try o.chatCompletionStream(model, oai_messages, .{
                    .temperature = config.temperature,
                    .max_tokens = config.max_tokens,
                    .top_p = config.top_p,
                    .stop = config.stop,
                    .frequency_penalty = config.frequency_penalty,
                    .presence_penalty = config.presence_penalty,
                    .seed = config.seed,
                }, .{ .user_ctx = context, .user_cb = callback, .alloc = o.allocator }, &Wrapper.wrap);
            },
            .anthropic => |a| {
                const ant_messages = messagesToAnthropicMessages(a.allocator, messages) catch return error.OutOfMemory;
                defer a.allocator.free(ant_messages);

                const Wrapper = struct {
                    fn wrap(ctx: struct { user_ctx: @TypeOf(context), user_cb: *const fn (@TypeOf(context), GenerateResult) void, alloc: std.mem.Allocator }, event: anthropic_types.StreamEvent) void {
                        // Extract text from content_block_delta events
                        const text_content = if (event.delta) |delta| delta.text else null;
                        var result = GenerateResult{
                            .text = text_content,
                            .allocator = ctx.alloc,
                        };
                        ctx.user_cb(ctx.user_ctx, result);
                        _ = &result;
                    }
                };

                try a.createMessageStream(model, ant_messages, config.max_tokens orelse 4096, .{
                    .temperature = config.temperature,
                    .top_p = config.top_p,
                    .stop_sequences = config.stop,
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
                const response = try g.embedText(model, text);
                const values = if (response.value.embedding) |e| e.values else null;
                return EmbedResult{
                    .values = values,
                    ._gemini_response = response,
                };
            },
            .openai => |o| {
                const response = try o.embedText(model, text);
                const values = if (response.value.data) |data|
                    if (data.len > 0) data[0].embedding else null
                else
                    null;
                return EmbedResult{
                    .values = values,
                    ._openai_response = response,
                };
            },
            .anthropic => {
                // Anthropic does not offer an embeddings API.
                return error.ApiError;
            },
        }
    }

    /// Drop down to the Gemini-specific client.
    pub fn asGemini(self: Client) ?*gemini_mod {
        return switch (self) {
            .gemini => |g| g,
            else => null,
        };
    }

    /// Drop down to the OpenAI-specific client.
    pub fn asOpenAI(self: Client) ?*openai_mod {
        return switch (self) {
            .openai => |o| o,
            else => null,
        };
    }

    /// Drop down to the Anthropic-specific client.
    pub fn asAnthropic(self: Client) ?*anthropic_mod {
        return switch (self) {
            .anthropic => |a| a,
            else => null,
        };
    }
};

// --- Conversion helpers ---

fn messagesToGeminiContents(allocator: std.mem.Allocator, messages: []const Message) ![]gemini_types.Content {
    const contents = try allocator.alloc(gemini_types.Content, messages.len);
    for (messages, 0..) |msg, i| {
        const role: []const u8 = switch (msg.role) {
            .system, .user, .tool => "user",
            .assistant => "model",
        };
        const parts: []const gemini_types.Part = if (msg.content) |c|
            @as([]const gemini_types.Part, &.{.{ .text = c }})
        else
            &.{};
        contents[i] = .{ .role = role, .parts = parts };
    }
    return contents;
}

fn messagesToOpenAIMessages(allocator: std.mem.Allocator, messages: []const Message) ![]openai_types.Message {
    const oai_messages = try allocator.alloc(openai_types.Message, messages.len);
    for (messages, 0..) |msg, i| {
        oai_messages[i] = .{
            .role = switch (msg.role) {
                .system => .system,
                .user => .user,
                .assistant => .assistant,
                .tool => .tool,
            },
            .content = msg.content,
        };
    }
    return oai_messages;
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

fn messagesToAnthropicMessages(allocator: std.mem.Allocator, messages: []const Message) ![]anthropic_types.MessageParam {
    const ant_messages = try allocator.alloc(anthropic_types.MessageParam, messages.len);
    for (messages, 0..) |msg, i| {
        const role: anthropic_types.Role = switch (msg.role) {
            .system, .user, .tool => .user,
            .assistant => .assistant,
        };
        const content: []const anthropic_types.ContentBlockParam = if (msg.content) |c|
            @as([]const anthropic_types.ContentBlockParam, &.{.{ .text = c }})
        else
            &.{};
        ant_messages[i] = .{ .role = role, .content = content };
    }
    return ant_messages;
}

fn mapAnthropicFinishReason(response: anthropic_types.MessageResponse) FinishReason {
    const reason = response.stop_reason orelse return .unknown;
    return switch (reason) {
        .end_turn, .stop_sequence => .stop,
        .max_tokens, .pause_turn => .max_tokens,
        .tool_use => .tool_call,
        .refusal => .safety,
    };
}

fn mapAnthropicUsage(response: anthropic_types.MessageResponse) Usage {
    const usage = response.usage orelse return .{};
    return .{
        .prompt_tokens = usage.input_tokens,
        .completion_tokens = usage.output_tokens,
        .total_tokens = if (usage.input_tokens != null and usage.output_tokens != null)
            usage.input_tokens.? + usage.output_tokens.?
        else
            null,
    };
}

// --- Tests ---

test "Message and Role basics" {
    const msg = Message{ .role = .user, .content = "hello" };
    try std.testing.expect(msg.role == .user);
    try std.testing.expectEqualStrings("hello", msg.content.?);
}

test "GenerateResult deinit with no backing response" {
    var result = GenerateResult{
        .text = "test",
        .finish_reason = .stop,
        .allocator = std.testing.allocator,
    };
    result.deinit(); // Should not crash
}

test "Client tagged union escape hatches" {
    var gemini_client = gemini_mod.init(std.testing.allocator, "test-key", .{});
    defer gemini_client.deinit();

    const ai: Client = .{ .gemini = &gemini_client };
    try std.testing.expect(ai.asGemini() != null);
    try std.testing.expect(ai.asOpenAI() == null);
}
