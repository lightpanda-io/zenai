const std = @import("std");
const json = @import("json.zig");
const retry = @import("retry.zig");
const http = @import("http.zig");
const gemini_mod = @import("gemini/Client.zig");
const openai_mod = @import("openai/Client.zig");
const anthropic_mod = @import("anthropic/Client.zig");
const gemini_types = @import("gemini/types.zig");
const openai_types = @import("openai/types.zig");
const anthropic_types = @import("anthropic/types.zig");
const ollama_native = @import("openai/ollama.zig");

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
            .arguments = if (tc.arguments) |v| try json.dupeValue(alloc, v) else null,
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
pub const Effort = enum {
    none,
    minimal,
    low,
    medium,
    high,
    xhigh,
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
    /// reasoning/thinking effort level.
    effort: ?Effort = null,
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
    /// Fresh (cache-excluded) input tokens, billed at full price. The mappers
    /// normalize every provider to this meaning: Anthropic already reports it
    /// (input_tokens excludes cache), while OpenAI and Gemini fold the cached
    /// subset into their prompt count, so `freshPrompt` subtracts it out.
    prompt_tokens: ?i32 = null,
    completion_tokens: ?i32 = null,
    total_tokens: ?i32 = null,
    /// Prompt tokens served from a cache (Anthropic: cache_read_input_tokens;
    /// OpenAI: prompt_tokens_details.cached_tokens; Gemini:
    /// cachedContentTokenCount). Billed at the provider's cached rate, and
    /// disjoint from `prompt_tokens` (see above).
    cached_tokens: ?i32 = null,
    /// Tokens written to a fresh cache entry. Anthropic-specific
    /// (cache_creation_input_tokens). OpenAI and Gemini bill cache creation
    /// at the standard input rate and don't report it separately, so this
    /// field stays null on those providers.
    cache_creation_tokens: ?i32 = null,

    /// Sum `other` into `self`, treating null as 0 — the result is non-null
    /// for any field that either operand reported. Used by agentic loops to
    /// accumulate per-turn usage into a per-task total.
    pub fn add(self: *Usage, other: Usage) void {
        self.prompt_tokens = addOpt(self.prompt_tokens, other.prompt_tokens);
        self.completion_tokens = addOpt(self.completion_tokens, other.completion_tokens);
        self.total_tokens = addOpt(self.total_tokens, other.total_tokens);
        self.cached_tokens = addOpt(self.cached_tokens, other.cached_tokens);
        self.cache_creation_tokens = addOpt(self.cache_creation_tokens, other.cache_creation_tokens);
    }

    /// Total input tokens billed: the three disjoint buckets summed — fresh
    /// (`prompt_tokens`) + cache reads (`cached_tokens`) + cache writes
    /// (`cache_creation_tokens`).
    pub fn inputTokens(self: Usage) i32 {
        return (self.prompt_tokens orelse 0) +
            (self.cached_tokens orelse 0) +
            (self.cache_creation_tokens orelse 0);
    }

    /// Percentage of input tokens served from cache, 0 when there was no input.
    /// Widened to i64 for the multiply since input can exceed i32 max / 100.
    pub fn cacheHitPercent(self: Usage) i32 {
        const input = self.inputTokens();
        if (input == 0) return 0;
        return @intCast(@divTrunc(@as(i64, self.cached_tokens orelse 0) * 100, input));
    }
};

/// OpenAI and Gemini report `prompt` as the full input with `cached` as a
/// subset of it; subtract so `Usage.prompt_tokens` means fresh input on every
/// provider (Anthropic already excludes cache from its prompt count).
fn freshPrompt(prompt: ?i32, cached: ?i32) ?i32 {
    const p = prompt orelse return null;
    return p - (cached orelse 0);
}

fn addOpt(a: ?i32, b: ?i32) ?i32 {
    if (a == null and b == null) return null;
    return (a orelse 0) + (b orelse 0);
}

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
    anthropic: *anthropic_mod,
    gemini: *gemini_mod,
    openai: *openai_mod,
    ollama: *openai_mod,
    huggingface: *openai_mod,
    llama_cpp: *openai_mod,
    vercel: *openai_mod,
    mistral: *openai_mod,

    pub const Error = gemini_mod.ApiError || openai_mod.ApiError || anthropic_mod.ApiError;
    pub const StreamError = gemini_mod.StreamError || openai_mod.StreamError || anthropic_mod.StreamError;

    fn clientAllocator(self: Client) std.mem.Allocator {
        return switch (self) {
            inline else => |c| c.allocator,
        };
    }

    pub const InitOptions = struct {
        /// Overrides the provider's default base URL. Ollama falls back to
        /// `ollama_default_base_url` when null; the others use their own.
        base_url: ?[:0]const u8 = null,
        retry_policy: retry.RetryPolicy = .{},
        /// Hugging Face org to bill via `X-HF-Bill-To`. Only applied to the
        /// OpenAI-compatible clients; ignored by gemini/anthropic.
        bill_to: ?[]const u8 = null,
    };

    /// Construct the per-provider client for `credentials`; the caller owns it
    /// and must release it with `deinit`.
    pub fn init(allocator: std.mem.Allocator, credentials: Credentials, options: InitOptions) !Client {
        return switch (credentials.provider) {
            inline else => |tag| blk: {
                const ClientPtr = @FieldType(Client, @tagName(tag));
                const Impl = @typeInfo(ClientPtr).pointer.child;
                const client = try allocator.create(Impl);
                errdefer allocator.destroy(client);
                const base_url: ?[:0]const u8 = options.base_url orelse
                    if (openAiPreset(tag)) |p| p.base_url else null;
                var impl_opts: Impl.InitOptions = .{ .retry_policy = options.retry_policy };
                if (base_url) |u| impl_opts.base_url = u;
                if (@hasField(Impl.InitOptions, "bill_to")) impl_opts.bill_to = options.bill_to;
                client.* = Impl.init(allocator, credentials.key, impl_opts);
                break :blk @unionInit(Client, @tagName(tag), client);
            },
        };
    }

    /// Free the per-provider client allocated by `init`.
    pub fn deinit(self: Client, allocator: std.mem.Allocator) void {
        switch (self) {
            inline else => |client| {
                client.deinit();
                allocator.destroy(client);
            },
        }
    }

    /// Install a cross-thread interrupt so a SIGINT can abort an in-flight
    /// request mid-read instead of waiting for the model's full response.
    pub fn setInterrupt(self: Client, it: *http.Interrupt) void {
        switch (self) {
            inline else => |client| client.interrupt = it,
        }
    }

    /// Status and message from the client's most recent failed request, to
    /// surface detail past the opaque `error.ApiError`. The message is owned by
    /// the client and valid until its next request; both null if none failed.
    pub const LastError = struct {
        status: ?u10 = null,
        message: ?[]const u8 = null,
    };

    pub fn lastError(self: Client) LastError {
        return switch (self) {
            inline else => |client| .{ .status = client.last_error_status, .message = client.last_error_message },
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

                var response = try g.generateContent(model, contents, mapGeminiGenerationConfig(model, config), .{
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
                                        .arguments = if (fc.args) |v| try json.dupeValue(result.arena.allocator(), v) else null,
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
            .openai => |o| {
                var req_arena = std.heap.ArenaAllocator.init(o.allocator);
                defer req_arena.deinit();
                const req_alloc = req_arena.allocator();

                const input = try messagesToOpenAIResponsesInput(req_alloc, messages);
                const tools = if (config.tools) |t| try mapOpenAIResponsesTools(req_alloc, t) else null;

                var response = try o.createResponse(.{
                    .model = model,
                    .input = input,
                    .tools = tools,
                    .tool_choice = mapToolChoiceToOpenAI(config.tool_choice),
                    .reasoning = if (config.effort) |tl|
                        .{ .effort = mapEffortToOpenAI(tl) }
                    else
                        null,
                    .max_output_tokens = config.max_tokens,
                    .temperature = config.temperature,
                });
                defer response.deinit();

                var result = GenerateResult.init(o.allocator);
                errdefer result.deinit();
                const ga = result.arena.allocator();

                if (response.value.text()) |t| {
                    result.text = try ga.dupe(u8, t);
                }
                result.finish_reason = mapOpenAIResponsesFinishReason(response.value);
                result.usage = mapOpenAIResponsesUsage(response.value);

                if (response.value.output) |items| {
                    var tool_calls: std.ArrayList(ToolCall) = .empty;
                    for (items) |item| {
                        const item_type = item.type orelse continue;
                        if (!std.mem.eql(u8, item_type, "function_call")) continue;
                        const args_val: ?std.json.Value = if (item.arguments) |a|
                            (std.json.parseFromSliceLeaky(std.json.Value, ga, a, .{}) catch null)
                        else
                            null;
                        try tool_calls.append(ga, .{
                            .id = if (item.call_id) |id| try ga.dupe(u8, id) else "",
                            .name = if (item.name) |n| try ga.dupe(u8, n) else "",
                            .arguments = args_val,
                        });
                    }
                    if (tool_calls.items.len > 0) {
                        result.tool_calls = try tool_calls.toOwnedSlice(ga);
                    }
                }
                return result;
            },
            // Native `/api/chat` so `num_ctx` can be sized — see `openai/ollama.zig`.
            .ollama => |o| {
                var req_arena = std.heap.ArenaAllocator.init(o.allocator);
                defer req_arena.deinit();
                const req_alloc = req_arena.allocator();

                const native_messages = try messagesToOllamaMessages(req_alloc, messages);
                // Native `/api/chat` has no tool_choice; honor `.none` by
                // withholding the tools entirely (the only enforceable case).
                const suppress_tools = if (config.tool_choice) |tc| tc == .none else false;
                const tools = if (!suppress_tools) blk: {
                    break :blk if (config.tools) |t| try mapOpenAITools(req_alloc, t) else null;
                } else null;

                const format: ?[]const u8 = if (config.response_format) |rf|
                    (if (rf == .json) "json" else null)
                else
                    null;

                var response = try ollama_native.chat(o, model, native_messages, tools, mapOllamaThink(config.effort), format, .{
                    .num_predict = config.max_tokens,
                    .temperature = config.temperature,
                    .top_p = config.top_p,
                    .seed = config.seed,
                    .stop = config.stop,
                    .frequency_penalty = config.frequency_penalty,
                    .presence_penalty = config.presence_penalty,
                });
                defer response.deinit();

                var result = GenerateResult.init(o.allocator);
                errdefer result.deinit();
                const ra = result.arena.allocator();

                result.finish_reason = mapOllamaFinishReason(response.value);
                result.usage = mapOllamaUsage(response.value);

                if (response.value.message) |msg| {
                    if (msg.content) |c| {
                        if (c.len > 0) result.text = try ra.dupe(u8, c);
                    }
                    if (msg.tool_calls) |calls| {
                        var tool_calls: std.ArrayList(ToolCall) = .empty;
                        for (calls) |call| {
                            if (call.function) |f| {
                                try tool_calls.append(ra, .{
                                    .id = if (call.id) |id| try ra.dupe(u8, id) else "",
                                    .name = if (f.name) |n| try ra.dupe(u8, n) else "",
                                    // Dupe args (an object) into the result arena.
                                    .arguments = if (f.arguments) |v| try json.dupeValue(ra, v) else null,
                                });
                            }
                        }
                        if (tool_calls.items.len > 0) {
                            result.tool_calls = try tool_calls.toOwnedSlice(ra);
                        }
                    }
                }
                return result;
            },
            // Hugging Face speaks OpenAI-compatible Chat Completions, not the
            // Responses API the `.openai` arm uses — so it gets its own arm.
            .huggingface, .llama_cpp, .vercel, .mistral => |o| {
                var req_arena = std.heap.ArenaAllocator.init(o.allocator);
                defer req_arena.deinit();
                const req_alloc = req_arena.allocator();

                const oai_messages = try messagesToOpenAIMessages(req_alloc, messages);
                const tools = if (config.tools) |t| try mapOpenAITools(req_alloc, t) else null;

                var response = try o.chatCompletion(model, oai_messages, mapOpenAICompletionConfig(config, tools));
                defer response.deinit();

                var result = GenerateResult.init(o.allocator);
                errdefer result.deinit();
                const ga = result.arena.allocator();

                if (response.value.text()) |t| {
                    if (t.len > 0) result.text = try ga.dupe(u8, t);
                }
                result.finish_reason = mapOpenAIFinishReason(response.value);
                result.usage = mapOpenAIUsage(response.value);

                extract: {
                    const choices = response.value.choices orelse break :extract;
                    if (choices.len == 0) break :extract;
                    const msg = choices[0].message orelse break :extract;
                    const calls = msg.tool_calls orelse break :extract;

                    var tool_calls: std.ArrayList(ToolCall) = .empty;
                    for (calls) |call| {
                        const f = call.function orelse continue;
                        // Chat Completions returns arguments as a JSON string.
                        const args_val: ?std.json.Value = if (f.arguments) |a|
                            (std.json.parseFromSliceLeaky(std.json.Value, ga, a, .{}) catch null)
                        else
                            null;
                        try tool_calls.append(ga, .{
                            .id = if (call.id) |id| try ga.dupe(u8, id) else "",
                            .name = if (f.name) |n| try ga.dupe(u8, n) else "",
                            .arguments = args_val,
                        });
                    }
                    if (tool_calls.items.len > 0) {
                        result.tool_calls = try tool_calls.toOwnedSlice(ga);
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
                const thinking = if (config.effort) |tl| mapEffortToAnthropic(tl) else null;
                const max_tokens = anthropicMaxTokens(config.max_tokens orelse 4096, thinking);

                var response = try a.createMessage(model, ant_messages, max_tokens, .{
                    .system = system_blocks,
                    .temperature = config.temperature,
                    .top_p = config.top_p,
                    .stop_sequences = config.stop,
                    .tools = tools,
                    .tool_choice = mapToolChoiceToAnthropic(config.tool_choice),
                    .thinking = thinking,
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
                                    .arguments = if (block.input) |v| try json.dupeValue(result.arena.allocator(), v) else null,
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

                try g.generateContentStream(model, contents, mapGeminiGenerationConfig(model, config), .{
                    .systemInstruction = sys_instruction,
                    .tools = tools,
                    .toolConfig = mapToolChoiceToGemini(config.tool_choice),
                }, .{ .user_ctx = context, .user_cb = callback, .alloc = g.allocator }, &Wrapper.wrap);
            },
            .openai, .huggingface, .llama_cpp, .vercel, .mistral => |o| {
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
            // The native chat (which sets `num_ctx`) is non-streaming, so emit
            // the full result in one callback rather than truncate via `/v1` SSE.
            .ollama => {
                var result = self.generateContent(model, messages, config) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => return error.ApiError,
                };
                defer result.deinit();
                callback(context, result);
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
                const thinking = if (config.effort) |tl| mapEffortToAnthropic(tl) else null;
                const max_tokens = anthropicMaxTokens(config.max_tokens orelse 4096, thinking);

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

                try a.createMessageStream(model, ant_messages, max_tokens, .{
                    .system = system_blocks,
                    .temperature = config.temperature,
                    .top_p = config.top_p,
                    .stop_sequences = config.stop,
                    .tools = tools,
                    .tool_choice = mapToolChoiceToAnthropic(config.tool_choice),
                    .thinking = thinking,
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
            .openai, .ollama, .huggingface, .llama_cpp, .vercel, .mistral => |o| {
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
        /// See `GenerationConfig.effort`. Forwarded per-turn.
        effort: ?Effort = null,
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
        /// Final-turn finish reason. `.safety` is a model refusal — surface it,
        /// don't re-prompt (deterministic).
        finish_reason: FinishReason = .unknown,
        /// Sum of per-turn `Usage` across every model call in this runTools
        /// invocation (including tool-call turns that produced no text).
        usage: Usage = .{},
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
        var total_usage: Usage = .{};

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
                .effort = config.effort,
            });
            defer gen_result.deinit();
            total_usage.add(gen_result.usage);

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
                        .arguments = if (tc.arguments) |v| try json.dupeValue(ra, v) else null,
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
                .usage = total_usage,
                .finish_reason = gen_result.finish_reason,
                .arena = result_arena,
            };
        }

        return .{
            .text = null,
            .tool_calls_made = all_tool_calls.toOwnedSlice(ra) catch &.{},
            .usage = total_usage,
            .cancelled = cancelled,
            .arena = result_arena,
        };
    }
};

/// Provider tag used in switches and helper APIs.
pub const Tag = std.meta.Tag(Client);

/// Config for the OpenAI-compatible providers, which share the `*openai_mod`
/// client and differ only in these fields. `openAiPreset` returns null for the
/// ones with bespoke config (native openai, anthropic, gemini).
const OpenAiPreset = struct {
    base_url: [:0]const u8,
    /// Env var holding the key; null for keyless local servers.
    env_var: ?[:0]const u8 = null,
    /// Placeholder key for keyless local servers, so the client's key check passes.
    placeholder_key: ?[:0]const u8 = null,
    default_model: []const u8,
    /// Local server: loopback-aware list retry, and excluded from auto-detection.
    local: bool = false,
};

fn openAiPreset(tag: Tag) ?OpenAiPreset {
    return switch (tag) {
        .ollama => .{ .base_url = "http://localhost:11434/v1", .placeholder_key = "ollama", .default_model = "qwen3.5:latest", .local = true },
        .huggingface => .{ .base_url = "https://router.huggingface.co/v1", .env_var = "HF_TOKEN", .default_model = "Qwen/Qwen3.5-122B-A10B" },
        // Empty default: the served model is whatever `llama-server` loaded.
        .llama_cpp => .{ .base_url = "http://localhost:8080/v1", .placeholder_key = "llama.cpp", .default_model = "", .local = true },
        .vercel => .{ .base_url = "https://ai-gateway.vercel.sh/v1", .env_var = "AI_GATEWAY_API_KEY", .default_model = "openai/gpt-5.5" },
        .mistral => .{ .base_url = "https://api.mistral.ai/v1", .env_var = "MISTRAL_API_KEY", .default_model = "mistral-large-latest" },
        .anthropic, .gemini, .openai => null,
    };
}

/// Look up the API key for `tag` from the conventional env var(s). Keyless local
/// servers return a placeholder so OpenAI-shaped clients clear their key check.
pub fn envApiKey(tag: Tag) ?[:0]const u8 {
    if (openAiPreset(tag)) |p| {
        if (p.env_var) |v| return std.posix.getenv(v);
        return p.placeholder_key;
    }
    return switch (tag) {
        .anthropic => std.posix.getenv("ANTHROPIC_API_KEY"),
        .openai => std.posix.getenv("OPENAI_API_KEY"),
        .gemini => std.posix.getenv("GOOGLE_API_KEY") orelse std.posix.getenv("GEMINI_API_KEY"),
        else => unreachable,
    };
}

/// Human-readable env var name(s) read by `envApiKey`, for diagnostics. Gemini
/// joins its two names with `/`; keyless local servers have none.
pub fn envVarName(tag: Tag) []const u8 {
    if (openAiPreset(tag)) |p| return p.env_var orelse @tagName(tag);
    return switch (tag) {
        .anthropic => "ANTHROPIC_API_KEY",
        .openai => "OPENAI_API_KEY",
        .gemini => "GOOGLE_API_KEY/GEMINI_API_KEY",
        else => unreachable,
    };
}

pub const ollama_default_base_url = openAiPreset(.ollama).?.base_url;
pub const huggingface_default_base_url = openAiPreset(.huggingface).?.base_url;
pub const llama_cpp_default_base_url = openAiPreset(.llama_cpp).?.base_url;

/// Recommended default chat model for `tag` when the user hasn't picked one.
pub fn defaultModel(tag: Tag) []const u8 {
    if (openAiPreset(tag)) |p| return p.default_model;
    return switch (tag) {
        .anthropic => "claude-sonnet-4-6",
        .openai => "gpt-5.5",
        .gemini => "gemini-3.5-flash",
        else => unreachable,
    };
}

/// A provider tag paired with the env-resolved key that authenticates it.
/// The two travel together: a tag is only meaningful with its key.
pub const Credentials = struct {
    provider: Tag,
    key: [:0]const u8,
};

/// Env-detectable providers, in enum order. Keyless local servers (preset
/// `local`) are excluded — their `envApiKey` is a placeholder, not a real key.
pub const default_candidates: []const Tag = blk: {
    const all = std.enums.values(Tag);
    var arr: [all.len]Tag = undefined;
    var n: usize = 0;
    for (all) |t| {
        if (openAiPreset(t)) |p| if (p.local) continue;
        arr[n] = t;
        n += 1;
    }
    const out = arr[0..n].*;
    break :blk &out;
};

/// Scan `candidates` and fill `buf` with a `Credentials` entry for each
/// provider that has a key in env, preserving candidate order. Returns the
/// subslice of `buf` actually filled. `buf.len` must be >= `candidates.len`.
pub fn detectKeys(buf: []Credentials, candidates: []const Tag) []Credentials {
    var n: usize = 0;
    for (candidates) |p| if (envApiKey(p)) |key| {
        buf[n] = .{ .provider = p, .key = key };
        n += 1;
    };
    return buf[0..n];
}

/// True when `url`'s host is a loopback address, where a refused connection is
/// instant and definitive rather than a transient error worth retrying. Returns
/// false on an unparseable or host-less URL, defaulting callers to retry.
fn isLoopbackUrl(url: []const u8) bool {
    const uri = std.Uri.parse(url) catch return false;
    const raw = switch (uri.host orelse return false) {
        .raw, .percent_encoded => |h| h,
    };
    // std.Uri keeps the brackets around IPv6 literals (e.g. "[::1]").
    const host = if (std.mem.startsWith(u8, raw, "[") and std.mem.endsWith(u8, raw, "]"))
        raw[1 .. raw.len - 1]
    else
        raw;
    return std.mem.eql(u8, host, "localhost") or
        std.mem.eql(u8, host, "::1") or
        std.mem.startsWith(u8, host, "127.");
}

/// Append every model ID served by an OpenAI-compatible endpoint at `base_url`
/// to `ids`. No `isChatModel` filtering — local/HF catalogs don't follow the
/// OpenAI naming convention it expects.
fn listOpenAICompatibleModelIds(
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    ids: *std.ArrayList([]const u8),
    api_key: [:0]const u8,
    base_url: [:0]const u8,
    retry_policy: retry.RetryPolicy,
) !void {
    var client = openai_mod.init(allocator, api_key, .{ .base_url = base_url, .retry_policy = retry_policy });
    defer client.deinit();
    var resp = try client.listModels();
    defer resp.deinit();
    for (resp.value.data orelse &.{}) |m| {
        if (m.id) |id| try ids.append(arena, try arena.dupe(u8, id));
    }
}

/// Fetch chat-capable model IDs for `tag`, allocated in `arena`. Ordering
/// is provider-defined — sort at the call site if needed. `base_url_override`
/// is honored for `openai` and every preset (OpenAI-compatible) provider.
pub fn listChatModelIds(
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    tag: Tag,
    api_key: [:0]const u8,
    base_url_override: ?[:0]const u8,
) ![][]const u8 {
    var ids: std.ArrayList([]const u8) = .empty;

    // Preset catalogs don't follow isChatModel's naming, so none filters. A
    // not-running loopback server refuses instantly — disable retry there.
    if (openAiPreset(tag)) |p| {
        const url = base_url_override orelse p.base_url;
        const policy: retry.RetryPolicy = if (p.local and isLoopbackUrl(url)) .disabled else .{};
        try listOpenAICompatibleModelIds(allocator, arena, &ids, api_key, url, policy);
    } else switch (tag) {
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
        else => unreachable,
    }

    const result = try ids.toOwnedSlice(arena);
    std.mem.sort([]const u8, result, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);
    return result;
}

test "envApiKey: ollama returns placeholder regardless of env" {
    try std.testing.expectEqualStrings("ollama", envApiKey(.ollama).?);
}

test "isLoopbackUrl: loopback hosts disable retry, remote hosts keep it" {
    try std.testing.expect(isLoopbackUrl(ollama_default_base_url));
    try std.testing.expect(isLoopbackUrl("http://127.0.0.1:11434/v1"));
    try std.testing.expect(isLoopbackUrl("http://[::1]:11434/v1"));
    try std.testing.expect(!isLoopbackUrl("http://ollama.example.com:11434/v1"));
    try std.testing.expect(!isLoopbackUrl("http://192.168.1.10:11434/v1"));
    try std.testing.expect(!isLoopbackUrl("not a url"));
}

test "huggingface: reads HF_TOKEN and has sensible defaults" {
    try std.testing.expectEqualStrings("HF_TOKEN", envVarName(.huggingface));
    try std.testing.expect(defaultModel(.huggingface).len > 0);
    // A real token, so it's auto-detectable like the other cloud providers.
    var found = false;
    for (default_candidates) |t| {
        if (t == .huggingface) found = true;
    }
    try std.testing.expect(found);
}

test "llama_cpp: placeholder key, loopback default, excluded from auto-detect" {
    try std.testing.expectEqualStrings("llama.cpp", envApiKey(.llama_cpp).?);
    try std.testing.expect(isLoopbackUrl(llama_cpp_default_base_url));
    // Placeholder key, so it must be live-probed — never env-auto-detected.
    for (default_candidates) |t| {
        try std.testing.expect(t != .llama_cpp);
    }
}

test "vercel/mistral: real-key cloud presets, auto-detectable" {
    try std.testing.expectEqualStrings("AI_GATEWAY_API_KEY", openAiPreset(.vercel).?.env_var.?);
    try std.testing.expectEqualStrings("MISTRAL_API_KEY", openAiPreset(.mistral).?.env_var.?);
    try std.testing.expect(defaultModel(.vercel).len > 0 and defaultModel(.mistral).len > 0);
    // Real keys, so both join env auto-detection; the keyless local servers don't.
    var saw_vercel = false;
    var saw_mistral = false;
    for (default_candidates) |t| {
        if (t == .vercel) saw_vercel = true;
        if (t == .mistral) saw_mistral = true;
        try std.testing.expect(t != .ollama and t != .llama_cpp);
    }
    try std.testing.expect(saw_vercel and saw_mistral);
}

test "reasoning_effort: omitted for none/null effort, sent otherwise" {
    const tools: ?[]const openai_types.Tool = null;
    try std.testing.expectEqual(@as(?openai_types.ReasoningEffort, null), mapOpenAICompletionConfig(.{ .effort = .none }, tools).reasoning_effort);
    try std.testing.expectEqual(@as(?openai_types.ReasoningEffort, null), mapOpenAICompletionConfig(.{ .effort = null }, tools).reasoning_effort);
    try std.testing.expectEqual(@as(?openai_types.ReasoningEffort, .low), mapOpenAICompletionConfig(.{ .effort = .low }, tools).reasoning_effort);
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

/// First text part of a message, or its plain `content` when it has no parts.
fn messageText(msg: Message) ?[]const u8 {
    const parts = msg.parts orelse return msg.content;
    for (parts) |cp| switch (cp) {
        .text => |t| return t,
        .image => {},
    };
    return null;
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
                    try json.valueToString(allocator, v)
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

        try out.append(allocator, .{
            .role = switch (msg.role) {
                .system => .system,
                .user => .user,
                .assistant => .assistant,
                .tool => unreachable,
            },
            .content = messageText(msg),
            .tool_calls = oai_tool_calls,
        });
    }

    return out.toOwnedSlice(allocator);
}

/// Like `messagesToOpenAIMessages`, but keeps tool-call arguments as objects
/// (not stringified) and answers tool results with `tool_name` (Ollama matches
/// by name, not id).
fn messagesToOllamaMessages(allocator: std.mem.Allocator, messages: []const Message) ![]ollama_native.Message {
    var out: std.ArrayList(ollama_native.Message) = .empty;

    for (messages) |msg| {
        if (msg.role == .tool) {
            if (msg.tool_results) |results| {
                for (results) |res| {
                    try out.append(allocator, .{
                        .role = "tool",
                        .content = res.content,
                        .tool_name = res.name,
                    });
                }
            } else {
                try out.append(allocator, .{ .role = "tool", .content = msg.content });
            }
            continue;
        }

        var native_tool_calls: ?[]const ollama_native.ToolCall = null;
        if (msg.tool_calls) |calls| {
            const arr = try allocator.alloc(ollama_native.ToolCall, calls.len);
            for (calls, 0..) |call, i| {
                arr[i] = .{
                    .id = if (call.id.len > 0) call.id else null,
                    .type = "function",
                    .function = .{ .name = call.name, .arguments = call.arguments },
                };
            }
            native_tool_calls = arr;
        }

        try out.append(allocator, .{
            .role = switch (msg.role) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
                .tool => unreachable,
            },
            .content = messageText(msg),
            .tool_calls = native_tool_calls,
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
                    // Anthropic rejects a tool_use block with no `input`; a
                    // no-arg tool must send `{}`.
                    .input = call.arguments orelse .{ .object = .init(allocator) },
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

/// Flatten the normalized conversation into Responses-API `input` items: each
/// assistant tool call becomes a `function_call` item and each tool result a
/// `function_call_output` item, rather than the role-tagged messages chat
/// completions uses.
fn messagesToOpenAIResponsesInput(allocator: std.mem.Allocator, messages: []const Message) ![]openai_types.ResponseInputItem {
    var out: std.ArrayList(openai_types.ResponseInputItem) = .empty;

    for (messages) |msg| {
        if (msg.role == .tool) {
            if (msg.tool_results) |results| {
                for (results) |res| {
                    try out.append(allocator, .{
                        .type = "function_call_output",
                        .call_id = res.id,
                        .output = res.content,
                    });
                }
            }
            continue;
        }

        const text_content: ?[]const u8 = if (msg.parts) |content_parts| blk: {
            for (content_parts) |cp| switch (cp) {
                .text => |t| break :blk t,
                .image => {},
            };
            break :blk null;
        } else msg.content;

        if (text_content) |c| {
            try out.append(allocator, .{
                .type = "message",
                .role = switch (msg.role) {
                    .system => "system",
                    .user => "user",
                    .assistant => "assistant",
                    .tool => unreachable,
                },
                .content = c,
            });
        }

        if (msg.tool_calls) |calls| {
            for (calls) |call| {
                // Responses requires a valid JSON string; a no-arg call sends `{}`.
                const args_str: []const u8 = if (call.arguments) |v|
                    try json.valueToString(allocator, v)
                else
                    "{}";
                try out.append(allocator, .{
                    .type = "function_call",
                    .call_id = call.id,
                    .name = call.name,
                    .arguments = args_str,
                });
            }
        }
    }

    return out.toOwnedSlice(allocator);
}

fn mapOpenAIResponsesTools(allocator: std.mem.Allocator, tools: []const Tool) ![]openai_types.ResponseTool {
    const out = try allocator.alloc(openai_types.ResponseTool, tools.len);
    for (tools, 0..) |t, i| {
        out[i] = .{
            .name = t.name,
            .description = t.description,
            .parameters = t.parameters,
        };
    }
    return out;
}

fn mapOpenAIResponsesFinishReason(response: openai_types.ResponsesResponse) FinishReason {
    if (response.output) |items| {
        for (items) |item| {
            const t = item.type orelse continue;
            if (std.mem.eql(u8, t, "function_call")) return .tool_call;
        }
    }
    if (response.incomplete_details) |d| {
        if (d.reason) |r| {
            if (std.mem.eql(u8, r, "max_output_tokens")) return .max_tokens;
            if (std.mem.eql(u8, r, "content_filter")) return .safety;
        }
    }
    return .stop;
}

fn mapOpenAIResponsesUsage(response: openai_types.ResponsesResponse) Usage {
    const usage = response.usage orelse return .{};
    const cached = if (usage.input_tokens_details) |d| d.cached_tokens else null;
    return .{
        .prompt_tokens = freshPrompt(usage.input_tokens, cached),
        .completion_tokens = usage.output_tokens,
        .total_tokens = usage.total_tokens,
        .cached_tokens = cached,
    };
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
        .unknown => .unknown,
    };
}

fn mapGeminiUsage(response: gemini_types.GenerateContentResponse) Usage {
    const meta = response.usageMetadata orelse return .{};
    return .{
        .prompt_tokens = freshPrompt(meta.promptTokenCount, meta.cachedContentTokenCount),
        .completion_tokens = meta.candidatesTokenCount,
        .total_tokens = meta.totalTokenCount,
        .cached_tokens = meta.cachedContentTokenCount,
    };
}

fn mapOpenAIUsage(response: openai_types.ChatCompletionResponse) Usage {
    const usage = response.usage orelse return .{};
    const cached = if (usage.prompt_tokens_details) |d| d.cached_tokens else null;
    return .{
        .prompt_tokens = freshPrompt(usage.prompt_tokens, cached),
        .completion_tokens = usage.completion_tokens,
        .total_tokens = usage.total_tokens,
        .cached_tokens = cached,
    };
}

/// Ollama's native `think` is a boolean toggle; map any requested level to on,
/// `.none` to an explicit off, and an unset level to the model default (null).
fn mapOllamaThink(level: ?Effort) ?bool {
    const l = level orelse return null;
    return l != .none;
}

fn mapOllamaFinishReason(response: ollama_native.ChatResponse) FinishReason {
    // A tool call wins over done_reason ("stop").
    if (response.message) |m| {
        if (m.tool_calls) |tc| {
            if (tc.len > 0) return .tool_call;
        }
    }
    const reason = response.done_reason orelse return .unknown;
    if (std.mem.eql(u8, reason, "stop")) return .stop;
    if (std.mem.eql(u8, reason, "length")) return .max_tokens;
    return .unknown;
}

fn mapOllamaUsage(response: ollama_native.ChatResponse) Usage {
    const prompt = response.prompt_eval_count;
    const completion = response.eval_count;
    return .{
        .prompt_tokens = prompt,
        .completion_tokens = completion,
        .total_tokens = if (prompt != null or completion != null)
            (prompt orelse 0) + (completion orelse 0)
        else
            null,
    };
}

fn mapAnthropicStopReason(reason: anthropic_types.StopReason) FinishReason {
    return switch (reason) {
        .end_turn, .stop_sequence => .stop,
        .max_tokens, .pause_turn => .max_tokens,
        .tool_use => .tool_call,
        .refusal => .safety,
        .unknown => .unknown,
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
        .cached_tokens = usage.cache_read_input_tokens,
        .cache_creation_tokens = usage.cache_creation_input_tokens,
    };
}

fn mapGeminiGenerationConfig(model: []const u8, config: GenerationConfig) gemini_types.GenerationConfig {
    return .{
        .temperature = config.temperature,
        .maxOutputTokens = config.max_tokens,
        .topP = config.top_p,
        .stopSequences = config.stop,
        .frequencyPenalty = config.frequency_penalty,
        .presencePenalty = config.presence_penalty,
        .seed = config.seed,
        .responseMimeType = mapResponseFormatToGemini(config.response_format),
        .thinkingConfig = if (config.effort) |tl|
            mapEffortToGeminiConfig(model, tl)
        else
            null,
    };
}

/// Gemini 2.5 models use `thinkingBudget` (token count); Gemini 3+ models use
/// the categorical `thinkingLevel`. Sending the wrong field returns HTTP 400
/// ("Thinking level is not supported for this model.").
fn geminiUsesThinkingBudget(model: []const u8) bool {
    return std.mem.indexOf(u8, model, "gemini-2.5") != null;
}

fn mapEffortToGeminiConfig(model: []const u8, level: Effort) gemini_types.ThinkingConfig {
    if (geminiUsesThinkingBudget(model)) {
        return .{ .thinkingBudget = mapEffortToGeminiBudget(level) };
    }
    return .{ .thinkingLevel = mapEffortToGemini(level) };
}

fn mapEffortToGemini(level: Effort) ?gemini_types.ThinkingLevel {
    return switch (level) {
        .none => null, // Should we disable? Gemini uses ThinkingConfig presence.
        .minimal => .MINIMAL,
        .low => .LOW,
        .medium => .MEDIUM,
        .high => .HIGH,
        .xhigh => .HIGH, // Gemini doesn't have XHIGH
    };
}

/// Approximate token budgets for the legacy Gemini 2.5 `thinkingBudget` API.
/// Note: gemini-2.5-pro does not support disabling thinking (min 128 tokens),
/// so `.none` will fail on that model — accepted tradeoff.
fn mapEffortToGeminiBudget(level: Effort) i32 {
    return switch (level) {
        .none => 0,
        .minimal => 512,
        .low => 2048,
        .medium => 8192,
        .high => 16384,
        .xhigh => -1, // Dynamic — let the model decide, up to its cap.
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
        // `.none` (and a null effort) omit the field entirely rather than send a
        // value strict providers without reasoning (e.g. Mistral) reject; use
        // `/effort none` to opt out per model.
        .reasoning_effort = if (config.effort) |tl|
            (if (tl == .none) null else mapEffortToOpenAI(tl))
        else
            null,
    };
}

/// Anthropic returns HTTP 400 when `max_tokens <= thinking.budget_tokens`.
/// Bump `max_tokens` past the budget so the model still has room for output
/// after reasoning; caller-supplied values that already clear the budget are
/// left alone.
fn anthropicMaxTokens(max_tokens: i32, thinking: ?anthropic_types.ThinkingConfig) i32 {
    const t = thinking orelse return max_tokens;
    const budget = t.budget_tokens orelse return max_tokens;
    return @max(max_tokens, budget +| 4096);
}

fn mapEffortToAnthropic(level: Effort) ?anthropic_types.ThinkingConfig {
    return switch (level) {
        .none => null,
        .minimal => .{ .type = "adaptive" },
        .low => .{ .type = "enabled", .budget_tokens = 1024 },
        .medium => .{ .type = "enabled", .budget_tokens = 4096 },
        .high => .{ .type = "enabled", .budget_tokens = 16384 },
        .xhigh => .{ .type = "enabled", .budget_tokens = 32768 },
    };
}

fn mapEffortToOpenAI(level: Effort) openai_types.ReasoningEffort {
    return switch (level) {
        .none => .none,
        .minimal => .minimal,
        .low => .low,
        .medium => .medium,
        .high => .high,
        .xhigh => .xhigh,
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

test "anthropicMaxTokens leaves request unchanged without thinking" {
    try std.testing.expectEqual(@as(i32, 4096), anthropicMaxTokens(4096, null));
    try std.testing.expectEqual(
        @as(i32, 4096),
        anthropicMaxTokens(4096, .{ .type = "disabled" }),
    );
}

test "anthropicMaxTokens raises max_tokens above thinking budget" {
    // medium budget (4096) with default max_tokens (4096) would 400.
    try std.testing.expectEqual(
        @as(i32, 8192),
        anthropicMaxTokens(4096, .{ .type = "enabled", .budget_tokens = 4096 }),
    );
    // xhigh budget (32768) requires substantial output headroom.
    try std.testing.expectEqual(
        @as(i32, 36864),
        anthropicMaxTokens(4096, .{ .type = "enabled", .budget_tokens = 32768 }),
    );
}

test "anthropicMaxTokens respects caller-supplied max_tokens when already large" {
    try std.testing.expectEqual(
        @as(i32, 64000),
        anthropicMaxTokens(64000, .{ .type = "enabled", .budget_tokens = 4096 }),
    );
}
