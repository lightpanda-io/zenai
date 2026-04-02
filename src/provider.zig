const std = @import("std");
const gemini_mod = @import("gemini/Client.zig");
const openai_mod = @import("openai/Client.zig");
const anthropic_mod = @import("anthropic/Client.zig");
const gemini_types = @import("gemini/types.zig");
const openai_types = @import("openai/types.zig");
const anthropic_types = @import("anthropic/types.zig");
const http = @import("http.zig");

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
    /// JSON string of the arguments.
    arguments: []const u8,
};

pub const ToolResult = struct {
    id: []const u8,
    name: []const u8, // Required specifically by Gemini
    content: []const u8, // String representation of the result
};

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
    tool_calls: ?[]const ToolCall = null,
    tool_results: ?[]const ToolResult = null,
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
    /// We use an ArenaAllocator to safely manage dynamic strings and tool_call
    /// slices created when normalizing provider responses.
    arena: std.heap.ArenaAllocator,
    /// Backing memory for owned text — null when text borrows from Response.
    _owned_text: ?[]u8 = null,
    /// Provider response backing memory.
    _gemini_response: ?GeminiResponse = null,
    _openai_response: ?OpenAIResponse = null,
    _anthropic_response: ?AnthropicResponse = null,

    const GeminiResponse = http.Response(gemini_types.GenerateContentResponse);
    const OpenAIResponse = http.Response(openai_types.ChatCompletionResponse);
    const AnthropicResponse = http.Response(anthropic_types.MessageResponse);

    pub fn init(allocator: std.mem.Allocator) GenerateResult {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: *GenerateResult) void {
        if (self._gemini_response) |*r| r.deinit();
        if (self._openai_response) |*r| r.deinit();
        if (self._anthropic_response) |*r| r.deinit();
        self.arena.deinit();
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

                const response = try g.generateContent(model, contents, gemini_types.GenerationConfig{
                    .temperature = config.temperature,
                    .maxOutputTokens = config.max_tokens,
                    .topP = config.top_p,
                    .stopSequences = config.stop,
                    .frequencyPenalty = config.frequency_penalty,
                    .presencePenalty = config.presence_penalty,
                    .seed = config.seed,
                }, .{
                    .systemInstruction = sys_instruction,
                    .tools = tools,
                });

                var result = GenerateResult.init(g.allocator);
                errdefer result.deinit();
                result._gemini_response = response;
                result.text = response.value.text();
                result.finish_reason = mapGeminiFinishReason(response.value);
                result.usage = mapGeminiUsage(response.value);

                if (response.value.candidates) |candidates| {
                    if (candidates.len > 0) {
                        if (candidates[0].content) |content| {
                            var tool_calls = std.ArrayList(ToolCall).init(result.arena.allocator());
                            for (content.parts) |p| {
                                if (p.functionCall) |fc| {
                                    var args_str: []const u8 = "";
                                    if (fc.args) |args_val| {
                                        args_str = try std.json.stringifyAlloc(result.arena.allocator(), args_val, .{});
                                    }
                                    try tool_calls.append(.{
                                        .id = if (fc.id) |id| try result.arena.allocator().dupe(u8, id) else "",
                                        .name = if (fc.name) |n| try result.arena.allocator().dupe(u8, n) else "",
                                        .arguments = args_str,
                                    });
                                }
                            }
                            if (tool_calls.items.len > 0) {
                                result.tool_calls = try tool_calls.toOwnedSlice();
                            }
                        }
                    }
                }
                return result;
            },
            .openai => |o| {
                var req_arena = std.heap.ArenaAllocator.init(o.allocator);
                defer req_arena.deinit();
                const req_alloc = req_arena.allocator();

                const oai_messages = try messagesToOpenAIMessages(req_alloc, messages);
                const tools = if (config.tools) |t| try mapOpenAITools(req_alloc, t) else null;

                const response = try o.chatCompletion(model, oai_messages, .{
                    .temperature = config.temperature,
                    .max_tokens = config.max_tokens,
                    .top_p = config.top_p,
                    .stop = config.stop,
                    .frequency_penalty = config.frequency_penalty,
                    .presence_penalty = config.presence_penalty,
                    .seed = config.seed,
                    .tools = tools,
                });

                var result = GenerateResult.init(o.allocator);
                errdefer result.deinit();
                result._openai_response = response;
                result.text = response.value.text();
                result.finish_reason = mapOpenAIFinishReason(response.value);
                result.usage = mapOpenAIUsage(response.value);

                if (response.value.choices) |choices| {
                    if (choices.len > 0) {
                        if (choices[0].message) |msg| {
                            if (msg.tool_calls) |calls| {
                                var tool_calls = std.ArrayList(ToolCall).init(result.arena.allocator());
                                for (calls) |call| {
                                    if (call.function) |f| {
                                        try tool_calls.append(.{
                                            .id = if (call.id) |id| try result.arena.allocator().dupe(u8, id) else "",
                                            .name = if (f.name) |n| try result.arena.allocator().dupe(u8, n) else "",
                                            .arguments = if (f.arguments) |a| try result.arena.allocator().dupe(u8, a) else "",
                                        });
                                    }
                                }
                                if (tool_calls.items.len > 0) {
                                    result.tool_calls = try tool_calls.toOwnedSlice();
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

                const separated = try separateSystemMessages(req_alloc, messages);
                const ant_messages = try messagesToAnthropicMessages(req_alloc, messages);

                const system_blocks: ?[]const anthropic_types.TextBlock = if (separated.system_text) |sys|
                    @as([]const anthropic_types.TextBlock, &.{.{ .text = sys }})
                else
                    null;

                const tools = if (config.tools) |t| try mapAnthropicTools(req_alloc, t) else null;

                const response = try a.createMessage(model, ant_messages, config.max_tokens orelse 4096, .{
                    .system = system_blocks,
                    .temperature = config.temperature,
                    .top_p = config.top_p,
                    .stop_sequences = config.stop,
                    .tools = tools,
                });

                var result = GenerateResult.init(a.allocator);
                errdefer result.deinit();
                result._anthropic_response = response;
                result.text = response.value.text();
                result.finish_reason = mapAnthropicFinishReason(response.value);
                result.usage = mapAnthropicUsage(response.value);

                if (response.value.content) |blocks| {
                    var tool_calls = std.ArrayList(ToolCall).init(result.arena.allocator());
                    for (blocks) |block| {
                        if (block.type) |t| {
                            if (std.mem.eql(u8, t, "tool_use")) {
                                var args_str: []const u8 = "";
                                if (block.input) |input_val| {
                                    args_str = try std.json.stringifyAlloc(result.arena.allocator(), input_val, .{});
                                }
                                try tool_calls.append(.{
                                    .id = if (block.id) |id| try result.arena.allocator().dupe(u8, id) else "",
                                    .name = if (block.name) |n| try result.arena.allocator().dupe(u8, n) else "",
                                    .arguments = args_str,
                                });
                            }
                        }
                    }
                    if (tool_calls.items.len > 0) {
                        result.tool_calls = try tool_calls.toOwnedSlice();
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
                        ctx.user_cb(ctx.user_ctx, result);
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
                }, .{
                    .systemInstruction = sys_instruction,
                    .tools = tools,
                }, .{ .user_ctx = context, .user_cb = callback, .alloc = g.allocator }, &Wrapper.wrap);
            },
            .openai => |o| {
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
                        ctx.user_cb(ctx.user_ctx, result);
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
                    .tools = tools,
                }, .{ .user_ctx = context, .user_cb = callback, .alloc = o.allocator }, &Wrapper.wrap);
            },
            .anthropic => |a| {
                var req_arena = std.heap.ArenaAllocator.init(a.allocator);
                defer req_arena.deinit();
                const req_alloc = req_arena.allocator();

                const separated = separateSystemMessages(req_alloc, messages) catch return error.OutOfMemory;
                const ant_messages = messagesToAnthropicMessages(req_alloc, messages) catch return error.OutOfMemory;

                const system_blocks: ?[]const anthropic_types.TextBlock = if (separated.system_text) |sys|
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
                        ctx.user_cb(ctx.user_ctx, result);
                    }
                };

                try a.createMessageStream(model, ant_messages, config.max_tokens orelse 4096, .{
                    .system = system_blocks,
                    .temperature = config.temperature,
                    .top_p = config.top_p,
                    .stop_sequences = config.stop,
                    .tools = tools,
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

const SeparatedMessages = struct {
    contents: []gemini_types.Content,
    system_text: ?[]const u8,
};

fn separateSystemMessages(allocator: std.mem.Allocator, messages: []const Message) !SeparatedMessages {
    var system_text: ?[]const u8 = null;
    var non_system_count: usize = 0;
    for (messages) |msg| {
        if (msg.role == .system) {
            if (system_text == null) system_text = msg.content;
        } else {
            non_system_count += 1;
        }
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

        var parts = std.ArrayList(gemini_types.Part).init(allocator);

        if (msg.content) |c| {
            try parts.append(.{ .text = c });
        }

        if (msg.tool_calls) |calls| {
            for (calls) |call| {
                var args: ?std.json.Value = null;
                if (call.arguments.len > 0) {
                    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{});
                    args = parsed.value;
                }
                try parts.append(.{
                    .functionCall = .{
                        .id = call.id,
                        .name = call.name,
                        .args = args,
                    },
                });
            }
        }

        if (msg.tool_results) |results| {
            for (results) |res| {
                var response_val: ?std.json.Value = null;
                if (res.content.len > 0) {
                    const parsed = std.json.parseFromSlice(std.json.Value, allocator, res.content, .{}) catch null;
                    if (parsed) |p| {
                        response_val = p.value;
                    } else {
                        response_val = std.json.Value{ .string = res.content };
                    }
                }
                try parts.append(.{
                    .functionResponse = .{
                        .id = res.id,
                        .name = res.name,
                        .response = response_val,
                    },
                });
            }
        }

        contents[idx] = .{ .role = role, .parts = try parts.toOwnedSlice() };
        idx += 1;
    }

    return .{ .contents = contents, .system_text = system_text };
}

fn messagesToOpenAIMessages(allocator: std.mem.Allocator, messages: []const Message) ![]openai_types.Message {
    var out = std.ArrayList(openai_types.Message).init(allocator);

    for (messages) |msg| {
        if (msg.role == .tool) {
            if (msg.tool_results) |results| {
                for (results) |res| {
                    try out.append(.{
                        .role = .tool,
                        .content = res.content,
                        .tool_call_id = res.id,
                    });
                }
            } else {
                try out.append(.{
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
                oai_calls[i] = .{
                    .id = call.id,
                    .type = "function",
                    .function = .{
                        .name = call.name,
                        .arguments = call.arguments,
                    },
                };
            }
            oai_tool_calls = oai_calls;
        }

        try out.append(.{
            .role = switch (msg.role) {
                .system => .system,
                .user => .user,
                .assistant => .assistant,
                .tool => unreachable,
            },
            .content = msg.content,
            .tool_calls = oai_tool_calls,
        });
    }

    return out.toOwnedSlice();
}

fn messagesToAnthropicMessages(allocator: std.mem.Allocator, messages: []const Message) ![]anthropic_types.MessageParam {
    var out = std.ArrayList(anthropic_types.MessageParam).init(allocator);
    for (messages) |msg| {
        if (msg.role == .system) continue;

        var blocks = std.ArrayList(anthropic_types.ContentBlockParam).init(allocator);

        if (msg.content) |c| {
            try blocks.append(.{ .type = "text", .text = c });
        }

        if (msg.tool_calls) |calls| {
            for (calls) |call| {
                var input_val: ?std.json.Value = null;
                if (call.arguments.len > 0) {
                    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{});
                    input_val = parsed.value;
                }
                try blocks.append(.{
                    .type = "tool_use",
                    .id = call.id,
                    .name = call.name,
                    .input = input_val,
                });
            }
        }

        if (msg.tool_results) |results| {
            for (results) |res| {
                try blocks.append(.{
                    .type = "tool_result",
                    .tool_use_id = res.id,
                    .content = res.content,
                });
            }
        }

        const role: anthropic_types.Role = if (msg.role == .assistant) .assistant else .user;
        try out.append(.{ .role = role, .content = try blocks.toOwnedSlice() });
    }
    return out.toOwnedSlice();
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
    var result = GenerateResult.init(std.testing.allocator);
    result.text = "test";
    result.finish_reason = .stop;
    result.deinit(); // Should not crash
}

test "Client tagged union escape hatches" {
    var gemini_client = gemini_mod.init(std.testing.allocator, "test-key", .{});
    defer gemini_client.deinit();

    const ai: Client = .{ .gemini = &gemini_client };
    try std.testing.expect(ai.asGemini() != null);
    try std.testing.expect(ai.asOpenAI() == null);
}
