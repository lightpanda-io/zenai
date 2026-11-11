const std = @import("std");
const builtin = @import("builtin");
const json = @import("json.zig");
const retry = @import("retry.zig");
const http = @import("http.zig");
const env = @import("env.zig");
const gemini_mod = @import("gemini/Client.zig");
const openai_mod = @import("openai/Client.zig");
const anthropic_mod = @import("anthropic/Client.zig");
const gemini_types = @import("gemini/types.zig");
const openai_types = @import("openai/types.zig");
const anthropic_types = @import("anthropic/types.zig");
const ollama_mod = @import("ollama/Client.zig");

/// See `env.get`.
pub const envGet = env.get;

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
    /// The arguments as JSON text, `{}` for a call without any; must be valid
    /// JSON. OpenAI sends this text as is and every other backend writes it
    /// verbatim into its request, so history is never re-encoded. Text from a
    /// response is checked: arguments that aren't valid JSON become `{}`.
    arguments: []const u8 = "{}",
    /// Gemini thought signature — must be echoed back with the tool result.
    thought_signature: ?[]const u8 = null,

    /// Parse `arguments` (a `ToolCall.arguments` or the text a `ToolHandler`
    /// receives) into a tree allocated in `allocator`.
    pub fn parseArguments(allocator: std.mem.Allocator, arguments: []const u8) std.json.ParseError(std.json.Scanner)!std.json.Value {
        return std.json.parseFromSliceLeaky(std.json.Value, allocator, arguments, .{});
    }
};

pub const ToolResult = struct {
    id: []const u8,
    name: []const u8, // Required specifically by Gemini
    content: []const u8, // String representation of the result
    /// Rich content (text + images) for backends that can carry it in the
    /// result; `content` stays the text-only form. See `splitToolImages`.
    parts: ?[]const ContentPart = null,
    /// Signals that the tool call failed. Propagated to Anthropic's native
    /// `is_error` wire field; providers without an equivalent ignore it.
    is_error: bool = false,
    /// Gemini thought signature — echoed from the corresponding ToolCall.
    thought_signature: ?[]const u8 = null,
};

/// Encrypted reasoning from a Responses-API turn (OpenAI, Codex, xAI), echoed
/// back on the assistant message so the model keeps its chain of thought
/// across tool calls. The ciphertext is only valid for the provider and model
/// that produced it; other turns drop it.
pub const ReasoningItem = struct {
    id: []const u8,
    encrypted_content: []const u8,
    summary: []const []const u8 = &.{},
    provider: Tag,
    model: []const u8,
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
    /// Rich content parts (text + images). When set, takes precedence over
    /// `content` for building provider content blocks.
    parts: ?[]const ContentPart = null,
    /// Assistant turns only: reasoning to echo back. See `ReasoningItem`.
    reasoning: ?[]const ReasoningItem = null,
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
        .reasoning = if (msg.reasoning) |rs| try dupeReasoning(alloc, rs) else null,
    };
}

pub fn dupeReasoning(alloc: std.mem.Allocator, items: []const ReasoningItem) ![]const ReasoningItem {
    const out = try alloc.alloc(ReasoningItem, items.len);
    for (items, out) |r, *d| {
        const summary = try alloc.alloc([]const u8, r.summary.len);
        for (r.summary, summary) |t, *ds| ds.* = try alloc.dupe(u8, t);
        d.* = .{
            .id = try alloc.dupe(u8, r.id),
            .encrypted_content = try alloc.dupe(u8, r.encrypted_content),
            .summary = summary,
            .provider = r.provider,
            .model = try alloc.dupe(u8, r.model),
        };
    }
    return out;
}

pub fn dupeToolCalls(alloc: std.mem.Allocator, calls: []const ToolCall) ![]const ToolCall {
    const out = try alloc.alloc(ToolCall, calls.len);
    for (calls, 0..) |tc, i| {
        out[i] = .{
            .id = try alloc.dupe(u8, tc.id),
            .name = try alloc.dupe(u8, tc.name),
            .arguments = try alloc.dupe(u8, tc.arguments),
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
            .parts = if (tr.parts) |ps| try dupeParts(alloc, ps) else null,
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
/// A null `GenerationConfig.effort` sends no effort field, so the model
/// default applies.
pub const Effort = enum {
    /// Reasoning off, sent explicitly.
    none,
    minimal,
    low,
    medium,
    high,
    xhigh,
};

/// Minimal generation config — the intersection of what both providers support.
pub const GenerationConfig = struct {
    /// Ignored by Anthropic, whose API (2026-10-21) rejects sampling parameters.
    temperature: ?f32 = null,
    max_tokens: ?i32 = null,
    /// Ignored by Anthropic, like `temperature`.
    top_p: ?f32 = null,
    stop: ?[]const []const u8 = null,
    frequency_penalty: ?f32 = null,
    presence_penalty: ?f32 = null,
    seed: ?i32 = null,
    tools: ?[]const Tool = null,
    tool_choice: ?ToolChoice = null,
    response_format: ?ResponseFormat = null,
    /// Reasoning/thinking effort level; null sends none.
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
    /// Store on the assistant `Message` for this turn; see `ReasoningItem`.
    reasoning: ?[]const ReasoningItem = null,
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
    vertex: *gemini_mod,
    openai: *openai_mod,
    ollama: *ollama_mod,
    huggingface: *openai_mod,
    llama_cpp: *openai_mod,
    llmd: *openai_mod,
    /// Any OpenAI-compatible server configured via `OPENAI_BASE_URL`
    /// + `OPENAI_API_KEY`. Auto-detected when `OPENAI_BASE_URL` is set.
    openai_compatible: *openai_mod,
    vercel: *openai_mod,
    mistral: *openai_mod,
    openrouter: *openai_mod,
    xai: *openai_mod,
    /// The ChatGPT-subscription backend: the OpenAI client under
    /// `openai_mod.initChatGpt` auth.
    codex: *openai_mod,

    pub const Error = gemini_mod.ApiError || openai_mod.ApiError || anthropic_mod.ApiError;

    /// The backend client's HTTP plumbing.
    pub fn transport(self: Client) *http.Transport {
        return switch (self) {
            .ollama => |c| &c.openai.transport,
            inline else => |c| &c.transport,
        };
    }

    fn clientAllocator(self: Client) std.mem.Allocator {
        return self.transport().allocator;
    }

    /// Options every provider takes; what only one provider takes lives in
    /// its `Backend` variant.
    pub const InitOptions = struct {
        /// Overrides the provider's default base URL. Required for
        /// `.openai_compatible`, which has none (`fromEnv` reads it from
        /// OPENAI_BASE_URL).
        base_url: ?[:0]const u8 = null,
        retry_policy: retry.RetryPolicy = .{},
        /// Per-attempt wall-clock bound on non-streaming requests, from an
        /// established connection to the end of the body (connect excluded);
        /// exceeding it fails with `error.Timeout`. `null` waits indefinitely.
        request_timeout_ms: ?u32 = null,
        /// Borrowed `std.http.Client` to share a connection pool with other
        /// clients; see `http.Transport.Options.http_client`. Null owns one.
        http_client: ?*std.http.Client = null,
        /// Headers sent on every request, e.g. Hugging Face's `X-HF-Bill-To`.
        /// Borrowed: see `http.Transport.Options.extra_headers`.
        extra_headers: []const std.http.Header = &.{},
    };

    /// Construct the client for `backend`, authenticating with `key`; the
    /// caller owns it and must release it with `deinit`. Reads no environment
    /// variables: see `fromEnv` for the conventional ones.
    pub fn init(io: std.Io, allocator: std.mem.Allocator, backend: Backend, key: [:0]const u8, options: InitOptions) !Client {
        return switch (backend) {
            inline else => |settings, tag| blk: {
                const ClientPtr = @FieldType(Client, @tagName(tag));
                const Impl = @typeInfo(ClientPtr).pointer.child;
                // Never fall through to api.openai.com.
                const base_url: ?[:0]const u8 = options.base_url orelse info(tag).base_url orelse
                    if (tag == .openai_compatible) return error.MissingBaseUrl else null;
                var impl_opts: Impl.InitOptions = .{
                    .retry_policy = options.retry_policy,
                    .request_timeout_ms = options.request_timeout_ms,
                    .http_client = options.http_client,
                    .extra_headers = options.extra_headers,
                };
                if (base_url) |u| impl_opts.base_url = u;
                if (tag == .vertex) impl_opts.vertex = settings;
                const client = try allocator.create(Impl);
                errdefer allocator.destroy(client);
                client.* = if (tag == .codex)
                    try Impl.initChatGpt(io, allocator, key, impl_opts, settings)
                else
                    Impl.init(io, allocator, key, impl_opts);
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
        self.transport().interrupt = it;
    }

    /// Repoint a subscription client at a refreshed OAuth access token without
    /// tearing down connections. Only `.codex` refreshes tokens in place; other
    /// providers use a static key, so this is a no-op for them. The caller keeps
    /// ownership of `key` and frees the previous buffer only after this returns.
    pub fn setApiKey(self: Client, key: []const u8) void {
        switch (self) {
            .codex => |c| c.setApiKey(key),
            else => {},
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
        const detail = self.transport().last_error;
        return .{ .status = detail.status, .message = detail.message };
    }

    /// Generate content from a list of messages.
    pub fn generateContent(
        self: Client,
        model: []const u8,
        messages: []const Message,
        config: GenerationConfig,
    ) Error!GenerateResult {
        return self.generate(model, messages, config, null, null);
    }

    /// Streaming counterpart of `generateContent`: forwards each text delta to
    /// `on_text` as it arrives and returns the same reassembled `GenerateResult`
    /// (text, tool calls, reasoning, usage) the buffered call would.
    pub fn generateContentStream(
        self: Client,
        model: []const u8,
        messages: []const Message,
        config: GenerationConfig,
        on_text: TextDeltaHook,
    ) Error!GenerateResult {
        return self.generate(model, messages, config, on_text, null);
    }

    /// Build the backend's request once, then send it buffered (`on_text`
    /// null) or streamed into an accumulator that yields the same result.
    /// The result's text, tool calls and reasoning are copied into `out`, or
    /// into the result's own arena when `out` is null.
    fn generate(
        self: Client,
        model: []const u8,
        messages: []const Message,
        config_in: GenerationConfig,
        on_text: ?TextDeltaHook,
        out: ?std.mem.Allocator,
    ) Error!GenerateResult {
        const config = flooredConfig(config_in);
        var req_arena = std.heap.ArenaAllocator.init(self.clientAllocator());
        defer req_arena.deinit();
        const ra = req_arena.allocator();

        var result = GenerateResult.init(self.clientAllocator());
        errdefer result.deinit();
        // Points at this frame's `result.arena`; used only before `result` is returned.
        const oa = out orelse result.arena.allocator();

        switch (self) {
            .gemini, .vertex => |g| {
                const separated = try separateSystemMessages(ra, messages);
                const gen_config = mapGeminiGenerationConfig(model, config);
                const opts: gemini_mod.RequestOptions = .{
                    .systemInstruction = if (separated.system_text) |sys|
                        .{ .parts = try ra.dupe(gemini_types.Part, &.{.{ .text = sys }}) }
                    else
                        null,
                    .tools = if (config.tools) |t| try mapGeminiTools(ra, t) else null,
                    .toolConfig = mapToolChoiceToGemini(config.tool_choice),
                };
                const hook = on_text orelse {
                    var response = try g.generateContent(model, separated.contents, gen_config, opts);
                    defer response.deinit();
                    try geminiResult(&result, oa, response.value);
                    return result;
                };
                var acc = gemini_mod.StreamAccumulator.init(ra, hook.context, hook.onText);
                g.generateContentStream(model, separated.contents, gen_config, opts, &acc, gemini_mod.StreamAccumulator.onEvent) catch |err| return streamFailed(err);
                if (acc.err) |e| return e;
                try geminiResult(&result, oa, try acc.response());
                return result;
            },
            // OpenAI and xAI speak the Responses API (Chat Completions 400s on
            // function tools + reasoning for gpt-5; xAI treats it as legacy).
            // Codex is the same API behind ChatGPT auth, and stream-only.
            .openai, .xai, .codex => |o| {
                const input = try messagesToOpenAIResponsesInput(ra, messages, self.responsesSystemRole(), self, model);
                const tools = if (config.tools) |t| try mapOpenAIResponsesTools(ra, t) else null;
                const request = self.responsesRequest(model, input, tools, config);
                const hook = on_text orelse if (self == .codex) TextDeltaHook{ .context = undefined, .onText = noopOnText } else {
                    var response = try o.createResponse(request);
                    defer response.deinit();
                    try openAiResponsesResult(&result, oa, response.value, self, model);
                    return result;
                };
                var acc = openai_mod.ResponsesStreamAccumulator.init(ra, hook.context, hook.onText);
                o.createResponseStream(request, &acc, openai_mod.ResponsesStreamAccumulator.onEvent) catch |err| return streamFailed(err);
                if (acc.err) |e| return e;
                try openAiResponsesResult(&result, oa, try acc.response(), self, model);
                return result;
            },
            // Native `/api/chat` (NDJSON when streamed) so `num_ctx` can be
            // sized: the `/v1` shim defaults to 4096 and silently truncates.
            .ollama => |o| {
                const call = try mapOllamaCall(ra, messages, config);
                const hook = on_text orelse {
                    var response = try o.chat(model, call.messages, call.tools, call.think, call.format, call.options);
                    defer response.deinit();
                    try ollamaResult(&result, oa, response.value);
                    return result;
                };
                var acc = ollama_mod.StreamAccumulator.init(ra, hook.context, hook.onText);
                o.chatStream(model, call.messages, call.tools, call.think, call.format, call.options, &acc, ollama_mod.StreamAccumulator.onChunk) catch |err| return streamFailed(err);
                if (acc.err) |e| return e;
                try ollamaResult(&result, oa, acc.response());
                return result;
            },
            .huggingface, .llama_cpp, .llmd, .openai_compatible, .vercel, .mistral, .openrouter => |o| {
                const oai_messages = try messagesToOpenAIMessages(ra, messages);
                const tools = if (config.tools) |t| try mapOpenAITools(ra, t) else null;
                const cfg = self.openAiCompletionConfig(model, config, tools);
                const hook = on_text orelse {
                    var response = try o.chatCompletion(model, oai_messages, cfg);
                    defer response.deinit();
                    try openAiChatResult(&result, oa, response.value);
                    return result;
                };
                var acc = openai_mod.StreamAccumulator.init(ra, hook.context, hook.onText);
                o.chatCompletionStream(model, oai_messages, cfg, &acc, openai_mod.StreamAccumulator.onEvent) catch |err| return streamFailed(err);
                if (acc.err) |e| return e;
                try openAiChatResult(&result, oa, try acc.response());
                return result;
            },
            .anthropic => |a| {
                const system_text = try extractSystemText(ra, messages);
                const ant_messages = try messagesToAnthropicMessages(ra, messages);
                const reasoning = mapEffortToAnthropic(config.effort, config.max_tokens orelse 4096);
                // Anthropic has no native JSON mode; response_format is ignored.
                const opts: anthropic_mod.MessageConfig = .{
                    .system = if (system_text) |sys| try ra.dupe(anthropic_types.TextBlock, &.{.{ .text = sys }}) else null,
                    .stop_sequences = config.stop,
                    .tools = if (config.tools) |t| try mapAnthropicTools(ra, t) else null,
                    .tool_choice = mapToolChoiceToAnthropic(config.tool_choice),
                    .thinking = reasoning.thinking,
                    .output_config = reasoning.output_config,
                };
                const hook = on_text orelse {
                    var response = try a.createMessage(model, ant_messages, reasoning.max_tokens, opts);
                    defer response.deinit();
                    try anthropicResult(&result, oa, response.value);
                    return result;
                };
                var acc = anthropic_mod.StreamAccumulator.init(ra, hook.context, hook.onText);
                a.createMessageStream(model, ant_messages, reasoning.max_tokens, opts, &acc, anthropic_mod.StreamAccumulator.onEvent) catch |err| return streamFailed(err);
                if (acc.err) |e| return e;
                try anthropicResult(&result, oa, try acc.response());
                return result;
            },
        }
    }

    /// The OpenAI client behind an OpenAI-family tag.
    fn openAiClient(self: Client) *openai_mod {
        return switch (self) {
            .anthropic, .gemini, .vertex => unreachable,
            .ollama => |o| &o.openai,
            inline else => |o| o,
        };
    }

    /// Anthropic caches only behind explicit breakpoints, which each gateway
    /// takes in its own shape. OpenRouter's field is Anthropic-only; Vercel's
    /// `caching: "auto"` is a no-op for implicitly-caching providers.
    fn openAiCompletionConfig(self: Client, model: []const u8, config: GenerationConfig, tools: ?[]const openai_types.Tool) openai_mod.ChatCompletionConfig {
        var cfg = mapOpenAICompletionConfig(config, tools);
        switch (self) {
            .openrouter => if (std.mem.startsWith(u8, model, "anthropic/")) {
                cfg.cache_control = .{};
            },
            .vercel => cfg.providerOptions = .{ .gateway = .{ .caching = "auto" } },
            .mistral => if (config.effort) |tl| {
                cfg.reasoning_effort = mapEffortToMistral(tl);
            },
            .xai => if (config.effort) |tl| {
                cfg.reasoning_effort = mapEffortToXai(tl);
            },
            else => {},
        }
        return cfg;
    }

    /// Role for system messages in Responses `input`: the Codex backend 400s
    /// on `system`, and `developer` is its equivalent.
    fn responsesSystemRole(self: Client) []const u8 {
        return if (self == .codex) "developer" else "system";
    }

    /// Responses request for the `.openai`, `.xai` and `.codex` arms. xAI
    /// accepts fewer effort levels; Codex asks for reasoning summaries and,
    /// like codex-cli, sends no `max_output_tokens`.
    fn responsesRequest(self: Client, model: []const u8, input: []const openai_types.ResponseInputItem, tools: ?[]const openai_types.ResponseTool, config: GenerationConfig) openai_types.ResponsesRequest {
        return .{
            .model = model,
            .input = input,
            .tools = tools,
            .tool_choice = mapToolChoiceToOpenAI(config.tool_choice),
            .reasoning = if (config.effort) |tl| .{
                .effort = if (self == .xai) mapEffortToXai(tl) else mapEffortToOpenAI(tl),
                .summary = if (self == .codex) "auto" else null,
            } else null,
            .max_output_tokens = if (self == .codex) null else config.max_tokens,
            .temperature = config.temperature,
            // Stateless: the encrypted reasoning travels in `input` instead.
            .store = false,
            .include = &.{"reasoning.encrypted_content"},
        };
    }

    /// Generate an embedding for a text string.
    pub fn embed(
        self: Client,
        model: []const u8,
        text: []const u8,
    ) Error!EmbedResult {
        switch (self) {
            .gemini, .vertex => |g| {
                var response = try g.embedText(model, text);
                defer response.deinit();
                var result = EmbedResult.init(self.clientAllocator());
                if (response.value.embedding) |e| {
                    if (e.values) |v| {
                        result.values = try result.arena.allocator().dupe(f32, v);
                    }
                }
                return result;
            },
            .openai, .ollama, .huggingface, .openai_compatible, .llama_cpp, .llmd, .vercel, .mistral, .openrouter, .xai => {
                var response = try self.openAiClient().embedText(model, text);
                defer response.deinit();
                var result = EmbedResult.init(self.clientAllocator());
                if (response.value.data) |data| {
                    if (data.len > 0) {
                        if (data[0].embedding) |v| {
                            result.values = try result.arena.allocator().dupe(f32, v);
                        }
                    }
                }
                return result;
            },
            .anthropic, .codex => return error.UnsupportedByBackend,
        }
    }

    // --- Agentic tool-use loop ---

    /// Callback interface for executing tool calls.
    pub const ToolHandler = struct {
        context: *anyopaque,
        /// `arguments` is the call's JSON text (see `ToolCall.arguments`);
        /// `ToolCall.parseArguments` turns it into a tree. `allocator` is an
        /// arena freed once `runTools` has copied the result.
        callFn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tool_name: []const u8, arguments: []const u8) Result,

        /// What the tool callback returns: the string content the model should
        /// see, plus whether the call errored. `is_error` is the authoritative
        /// failure signal — the content may legitimately describe an error.
        /// `parts` adds images for backends that can carry them (see
        /// `ToolResult.parts`).
        pub const Result = struct {
            content: []const u8,
            is_error: bool = false,
            parts: ?[]const ContentPart = null,
        };

        pub fn call(self: ToolHandler, allocator: std.mem.Allocator, name: []const u8, args: []const u8) Result {
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
        /// When set, each turn streams the model response, forwarding text
        /// deltas to this hook as they arrive while still accumulating tool
        /// calls into the buffered per-turn result. Null runs the buffered path.
        stream: ?TextDeltaHook = null,
        /// Called before each model request; may rewrite earlier messages.
        before_request: ?BeforeRequestHook = null,
        /// Called with each model response, before its tool calls run.
        after_response: ?AfterResponseHook = null,

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

    pub const BeforeRequestHook = struct {
        context: *anyopaque,
        callFn: *const fn (context: *anyopaque) void,

        pub fn call(self: BeforeRequestHook) void {
            self.callFn(self.context);
        }
    };

    pub const AfterResponseHook = struct {
        context: *anyopaque,
        callFn: *const fn (context: *anyopaque, result: *const GenerateResult) void,

        pub fn call(self: AfterResponseHook, result: *const GenerateResult) void {
            self.callFn(self.context, result);
        }
    };

    /// Fired with each incremental assistant-text delta as it streams from the
    /// model. When set on `RunToolsConfig.stream`, `runTools` drives a streaming
    /// request per turn (reassembling tool calls internally) instead of the
    /// buffered path; when null, behavior is unchanged.
    pub const TextDeltaHook = struct {
        context: *anyopaque,
        onText: *const fn (context: *anyopaque, delta: []const u8) void,

        pub fn call(self: TextDeltaHook, delta: []const u8) void {
            self.onText(self.context, delta);
        }
    };

    /// Information about a tool call that was executed during the loop. The
    /// slices borrow the tool-call and tool-result messages `runTools`
    /// appended, so they live in `data_alloc` and stay valid as long as that
    /// memory does, independently of `RunToolsResult.deinit`.
    pub const ToolCallInfo = struct {
        name: []const u8,
        /// JSON text; see `ToolCall.arguments`.
        arguments: []const u8,
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
        /// Owns the `tool_calls_made` slice; `text` and the strings it points
        /// to live in `runTools`' `data_alloc`.
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
    /// ArrayList (must match the allocator used to create it). Each model
    /// reply is decoded straight into `data_alloc`, which also holds the
    /// copied tool results; the appended messages and `tool_calls_made`
    /// borrow it. Pass an arena: a reply dropped on cancel stays allocated.
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
            if (config.before_request) |hook| hook.call();
            const gen_config: GenerationConfig = .{
                .tools = config.tools,
                .max_tokens = config.max_tokens,
                .tool_choice = config.tool_choice,
                .temperature = config.temperature,
                .effort = config.effort,
            };
            // Decoded into `data_alloc`, where the messages below keep it.
            var gen_result = try self.generate(model, messages.items, gen_config, config.stream, data_alloc);
            defer gen_result.deinit();
            total_usage.add(gen_result.usage);
            if (config.after_response) |hook| hook.call(&gen_result);

            // A streamed turn returns partial data on mid-stream cancel rather
            // than erroring; catch it here so the loop stops instead of treating
            // the fragment as a finished answer.
            if (config.cancelRequested()) {
                cancelled = true;
                break;
            }

            if (gen_result.tool_calls) |tool_calls| {
                try messages.append(list_alloc, .{
                    .role = .assistant,
                    .content = gen_result.text,
                    .tool_calls = tool_calls,
                    .reasoning = gen_result.reasoning,
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

                    // The handler's arena dies with this iteration; copy what it
                    // returned. The rest borrows the assistant message above.
                    const content = try data_alloc.dupe(u8, handler_result.content);
                    try tool_results.append(data_alloc, .{
                        .id = tc.id,
                        .name = tc.name,
                        .content = content,
                        .parts = if (handler_result.parts) |p| try dupeParts(data_alloc, p) else null,
                        .is_error = handler_result.is_error,
                        .thought_signature = tc.thought_signature,
                    });

                    try all_tool_calls.append(ra, .{
                        .name = tc.name,
                        .arguments = tc.arguments,
                        .result = content,
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

            const text = gen_result.text;

            if (text != null) {
                try messages.append(list_alloc, .{
                    .role = .assistant,
                    .content = text,
                    .reasoning = gen_result.reasoning,
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

/// The provider `Client.init` builds, with the settings only that provider
/// takes. A tag with no settings coerces from its enum literal (`.anthropic`);
/// `Backend.of` converts a runtime `Tag`.
pub const Backend = union(Tag) {
    anthropic,
    gemini,
    /// Project/location mode when `project` is set, where the init key must be
    /// an OAuth access token; express mode (an API key) otherwise.
    vertex: gemini_mod.VertexConfig,
    openai,
    ollama,
    huggingface,
    llama_cpp,
    llmd,
    openai_compatible,
    vercel,
    mistral,
    openrouter,
    xai,
    /// The ChatGPT account and session ids; see `openai.Client.ChatGpt`.
    codex: openai_mod.ChatGpt,

    /// `tag` with default settings.
    pub fn of(tag: Tag) Backend {
        return switch (tag) {
            inline else => |t| @unionInit(Backend, @tagName(t), if (@FieldType(Backend, @tagName(t)) == void) {} else .{}),
        };
    }
};

/// What the conventional env vars configure for one provider: the pieces of a
/// `Client.init` call, which itself reads no environment.
pub const EnvConfig = struct {
    /// `envApiKey(environ, tag)`; null when none is set.
    key: ?[:0]const u8,
    /// `.vertex` takes its project from GOOGLE_CLOUD_PROJECT and its location
    /// from GOOGLE_CLOUD_LOCATION, then GOOGLE_CLOUD_REGION, then "global";
    /// every other tag gets `Backend.of(tag)`.
    backend: Backend,
    /// For `InitOptions.base_url`: OPENAI_BASE_URL for `.openai_compatible`,
    /// null for the others, which have their own default.
    base_url: ?[:0]const u8,
};

pub fn fromEnv(environ: std.process.Environ, tag: Tag) EnvConfig {
    return .{
        .key = envApiKey(environ, tag),
        .backend = switch (tag) {
            .vertex => .{ .vertex = .{
                .project = envGet(environ, "GOOGLE_CLOUD_PROJECT"),
                .location = envGet(environ, "GOOGLE_CLOUD_LOCATION") orelse
                    envGet(environ, "GOOGLE_CLOUD_REGION") orelse "global",
            } },
            else => .of(tag),
        },
        .base_url = if (tag == .openai_compatible) envGet(environ, "OPENAI_BASE_URL") else null,
    };
}

/// What differs per provider beyond its wire format, in one place.
pub const Info = struct {
    /// Default base URL; null when the client has its own default, or (for
    /// `.openai_compatible`) when it comes from OPENAI_BASE_URL.
    base_url: ?[:0]const u8 = null,
    /// Env var holding the key, for the providers whose lookup is just that.
    env_var: ?[:0]const u8 = null,
    /// Placeholder key for keyless local servers, so the client's key check passes.
    placeholder_key: ?[:0]const u8 = null,
    /// What `envVarName` reports when it isn't `env_var`.
    key_hint: ?[]const u8 = null,
    /// Recommended chat model; empty when the server decides (local servers).
    default_model: []const u8,
    /// Send no effort unless the caller picks one, instead of the caller's
    /// own default. See `omitsEffort`.
    omit_effort: bool = false,
    /// Local server: loopback-aware list retry, and excluded from auto-detection.
    local: bool = false,
};

pub fn info(tag: Tag) Info {
    return switch (tag) {
        .anthropic => .{ .env_var = "ANTHROPIC_API_KEY", .default_model = "claude-sonnet-5-5" },
        .gemini => .{ .key_hint = "GOOGLE_API_KEY/GEMINI_API_KEY", .default_model = "gemini-3.8-flash" },
        .vertex => .{ .key_hint = "VERTEX_API_KEY/GOOGLE_API_KEY", .default_model = "gemini-3.8-flash" },
        .openai => .{ .key_hint = "OPENAI_API_KEY", .default_model = "gpt-6-luna" },
        .codex => .{ .base_url = openai_mod.codex_base_url, .key_hint = "a ChatGPT subscription (OAuth)", .default_model = "gpt-5.5" },
        .ollama => .{ .base_url = "http://localhost:11434/v1", .placeholder_key = "ollama", .default_model = "qwen3.5:latest", .local = true },
        .huggingface => .{ .base_url = "https://router.huggingface.co/v1", .env_var = "HF_TOKEN", .default_model = "Qwen/Qwen3.5-122B-A10B" },
        // The base URL comes from OPENAI_BASE_URL; the model from the server.
        .openai_compatible => .{ .key_hint = "OPENAI_API_KEY", .default_model = "" },
        // The served model is whatever `llama-server` loaded.
        .llama_cpp => .{ .base_url = "http://localhost:8080/v1", .placeholder_key = "llama.cpp", .default_model = "", .local = true },
        // ZML's LLMD; the served model is whatever `--model` loaded.
        .llmd => .{ .base_url = "http://localhost:8000/v1", .placeholder_key = "llmd", .default_model = "", .local = true },
        .vercel => .{ .base_url = "https://ai-gateway.vercel.sh/v1", .env_var = "AI_GATEWAY_API_KEY", .default_model = "openai/gpt-5.5" },
        // codestral and ministral reject any `reasoning_effort`.
        .mistral => .{ .base_url = "https://api.mistral.ai/v1", .env_var = "MISTRAL_API_KEY", .default_model = "mistral-medium-3.5", .omit_effort = true },
        .openrouter => .{ .base_url = "https://openrouter.ai/api/v1", .env_var = "OPENROUTER_API_KEY", .default_model = "openai/gpt-5.5" },
        .xai => .{ .base_url = "https://api.x.ai/v1", .env_var = "XAI_API_KEY", .default_model = "grok-4.7" },
    };
}

/// Keyless servers on this machine, in enum order.
pub const local_providers: []const Tag = blk: {
    const all = std.enums.values(Tag);
    var arr: [all.len]Tag = undefined;
    var n: usize = 0;
    for (all) |t| if (info(t).local) {
        arr[n] = t;
        n += 1;
    };
    const out = arr[0..n].*;
    break :blk &out;
};

/// Look up the API key for `tag` from the conventional env var(s). Keyless local
/// servers return a placeholder so OpenAI-shaped clients clear their key check.
/// True when GOOGLE_GENAI_USE_VERTEXAI is "1" or "true" (case-insensitive),
/// matching go-genai's backend selection. Gates env detection so exactly one
/// of `.gemini`/`.vertex` can ever detect from GOOGLE_API_KEY.
pub fn useVertex(environ: std.process.Environ) bool {
    const v = envGet(environ, "GOOGLE_GENAI_USE_VERTEXAI") orelse return false;
    return std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "true");
}

pub fn envApiKey(environ: std.process.Environ, tag: Tag) ?[:0]const u8 {
    return switch (tag) {
        // OPENAI_BASE_URL gates OPENAI_API_KEY between `.openai` and
        // `.openai_compatible` so exactly one of them can ever detect it,
        // mirroring the `.gemini`/`.vertex` split on GOOGLE_API_KEY.
        .openai => if (envGet(environ, "OPENAI_BASE_URL") == null)
            envGet(environ, "OPENAI_API_KEY")
        else
            null,
        .openai_compatible => if (envGet(environ, "OPENAI_BASE_URL") != null)
            envGet(environ, "OPENAI_API_KEY")
        else
            null,
        .gemini => if (useVertex(environ))
            null
        else
            envGet(environ, "GOOGLE_API_KEY") orelse envGet(environ, "GEMINI_API_KEY"),
        // Express mode only: with GOOGLE_CLOUD_PROJECT set, the credential
        // must be an OAuth access token the caller supplies explicitly —
        // not detectable from env. VERTEX_API_KEY is unambiguous vertex
        // intent, so it skips the GOOGLE_GENAI_USE_VERTEXAI opt-in.
        .vertex => if (envGet(environ, "GOOGLE_CLOUD_PROJECT") != null)
            null
        else
            envGet(environ, "VERTEX_API_KEY") orelse
                (if (useVertex(environ)) envGet(environ, "GOOGLE_API_KEY") else null),
        // Codex authenticates with an OAuth subscription token supplied by the
        // caller, not an env var — never env-detected.
        .codex => null,
        else => {
            const i = info(tag);
            return if (i.env_var) |v| envGet(environ, v) else i.placeholder_key;
        },
    };
}

/// Human-readable env var name(s) read by `envApiKey`, for diagnostics. Gemini
/// joins its two names with `/`; keyless local servers give their tag name.
pub fn envVarName(tag: Tag) []const u8 {
    const i = info(tag);
    if (i.key_hint) |h| return h;
    return i.env_var orelse @tagName(tag);
}

/// The provider's default base URL, if it has one of its own.
pub fn defaultBaseUrl(tag: Tag) ?[:0]const u8 {
    return info(tag).base_url;
}

/// Recommended default chat model for `tag` when the user hasn't picked one.
pub fn defaultModel(tag: Tag) []const u8 {
    return info(tag).default_model;
}

/// True when `tag` should get no effort unless the user picks one, rather
/// than the caller's own default (Mistral: codestral and ministral reject any
/// `reasoning_effort`).
pub fn omitsEffort(tag: Tag) bool {
    return info(tag).omit_effort;
}

/// A provider whose key was found in the environment (`detectKeys`) or served
/// by a local probe — a selectable option, not yet the active credential.
pub const Candidate = struct {
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
        // openai_compatible is included; envApiKey gates it on OPENAI_BASE_URL.
        if (info(t).local) continue;
        arr[n] = t;
        n += 1;
    }
    const out = arr[0..n].*;
    break :blk &out;
};

/// Scan `candidates` and fill `buf` with a `Candidate` entry for each
/// provider that has a key in `environ`, preserving candidate order. Returns
/// the subslice of `buf` actually filled. `buf.len` must be >= `candidates.len`.
pub fn detectKeys(environ: std.process.Environ, buf: []Candidate, candidates: []const Tag) []Candidate {
    var n: usize = 0;
    for (candidates) |p| if (envApiKey(environ, p)) |key| {
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

/// Fetch chat-capable model IDs for `backend`, sorted and allocated in
/// `arena`. Builds a `Client` with `options` for the request; a local server
/// on loopback isn't retried, since a refused connection there is definitive.
pub fn listChatModelIds(
    io: std.Io,
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    backend: Backend,
    api_key: [:0]const u8,
    options: Client.InitOptions,
) ![][]const u8 {
    var ids: std.ArrayList([]const u8) = .empty;
    const tag = std.meta.activeTag(backend);
    // Codex has no listable models endpoint under OAuth; callers fall back to
    // the models.dev catalog.
    if (tag == .codex) return ids.toOwnedSlice(arena);

    var opts = options;
    if (info(tag).local and isLoopbackUrl(options.base_url orelse info(tag).base_url.?)) opts.retry_policy = .disabled;
    const client = try Client.init(io, allocator, backend, api_key, opts);
    defer client.deinit(allocator);

    switch (client) {
        .anthropic => |c| {
            var resp = try c.listModels();
            defer resp.deinit();
            for (resp.value.data orelse &.{}) |m| {
                if (!anthropic_mod.isChatModel(m)) continue;
                if (m.id) |id| try ids.append(arena, try arena.dupe(u8, id));
            }
        },
        .gemini => |c| {
            var resp = try c.listModels(.{});
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
        .vertex => |c| {
            var resp = try c.listModels(.{});
            defer resp.deinit();
            // Vertex publisher models carry no supportedGenerationMethods,
            // so the isChatModel filter can't apply; strip the resource
            // prefix so the output is pipe-ready into a `--model` flag.
            for (resp.value.publisherModels orelse &.{}) |m| {
                const name = m.name orelse continue;
                const prefix = "publishers/google/models/";
                const stripped = if (std.mem.startsWith(u8, name, prefix)) name[prefix.len..] else name;
                try ids.append(arena, try arena.dupe(u8, stripped));
            }
        },
        .codex => unreachable,
        inline else => |c, t| {
            const openai: *openai_mod = if (t == .ollama) &c.openai else c;
            var resp = try openai.listModels();
            defer resp.deinit();
            // Only OpenAI's own catalog follows isChatModel's naming.
            for (resp.value.data orelse &.{}) |m| {
                if (t == .openai and !openai_mod.isChatModel(m)) continue;
                if (m.id) |id| try ids.append(arena, try arena.dupe(u8, id));
            }
        },
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
    try std.testing.expectEqualStrings("ollama", envApiKey(.empty, .ollama).?);
}

test "isLoopbackUrl: loopback hosts disable retry, remote hosts keep it" {
    try std.testing.expect(isLoopbackUrl(defaultBaseUrl(.ollama).?));
    try std.testing.expect(isLoopbackUrl("http://127.0.0.1:11434/v1"));
    try std.testing.expect(isLoopbackUrl("http://[::1]:11434/v1"));
    try std.testing.expect(!isLoopbackUrl("http://ollama.example.com:11434/v1"));
    try std.testing.expect(!isLoopbackUrl("http://192.168.1.10:11434/v1"));
    try std.testing.expect(!isLoopbackUrl("not a url"));
}

test "vertex: gemini-backed, no OpenAI preset, gated auto-detection" {
    try std.testing.expect(defaultBaseUrl(.vertex) == null);
    try std.testing.expectEqualStrings(defaultModel(.gemini), defaultModel(.vertex));
    try std.testing.expect(envVarName(.vertex).len > 0);
    // Candidacy is runtime-gated by GOOGLE_GENAI_USE_VERTEXAI (useVertex),
    // not comptime-excluded, so the tag stays in the candidate list.
    var found = false;
    for (default_candidates) |t| {
        if (t == .vertex) found = true;
    }
    try std.testing.expect(found);
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
    try std.testing.expectEqualStrings("llama.cpp", envApiKey(.empty, .llama_cpp).?);
    try std.testing.expect(isLoopbackUrl(defaultBaseUrl(.llama_cpp).?));
    // Placeholder key, so it must be live-probed — never env-auto-detected.
    for (default_candidates) |t| {
        try std.testing.expect(t != .llama_cpp);
    }
}

test "llmd: placeholder key, loopback default, excluded from auto-detect" {
    try std.testing.expectEqualStrings("llmd", envApiKey(.empty, .llmd).?);
    try std.testing.expect(isLoopbackUrl(defaultBaseUrl(.llmd).?));
    for (default_candidates) |t| try std.testing.expect(t != .llmd);
}

test "cloud gateway presets: real key, default model, auto-detectable" {
    for ([_]Tag{ .vercel, .mistral, .openrouter, .xai }) |tag| {
        try std.testing.expect(info(tag).env_var != null);
        try std.testing.expect(defaultModel(tag).len > 0);
        // Real keys, so they join env auto-detection; the keyless local servers don't.
        try std.testing.expect(std.mem.findScalar(Tag, default_candidates, tag) != null);
    }
    for (default_candidates) |t| try std.testing.expect(!info(t).local);
    try std.testing.expectEqualSlices(Tag, &.{ .ollama, .llama_cpp, .llmd }, local_providers);
}

test "openai_compatible: no static preset, env-gated key detection" {
    try std.testing.expect(defaultBaseUrl(.openai_compatible) == null);
    // Without OPENAI_BASE_URL the key must not be detected — that's `.openai`.
    try std.testing.expect(envApiKey(.empty, .openai_compatible) == null);
}

test "openai/openai_compatible: OPENAI_BASE_URL splits key detection" {
    // A synthetic environ can only be built from a PosixBlock
    // slice, which does not exist on Windows; there getWindows
    // reads the live PEB environment (see the envGet test above),
    // so these POSIX-only assertions are comptime-skipped.
    if (comptime builtin.os.tag != .windows) {
        const with_url: std.process.Environ = .{ .block = .{ .slice = &[_:null]?[*:0]const u8{
            "OPENAI_BASE_URL=http://localhost:8000/v1",
            "OPENAI_API_KEY=sk-test",
        } } };
        try std.testing.expect(envApiKey(with_url, .openai) == null);
        try std.testing.expectEqualStrings("sk-test", envApiKey(with_url, .openai_compatible).?);

        const without_url: std.process.Environ = .{ .block = .{ .slice = &[_:null]?[*:0]const u8{
            "OPENAI_API_KEY=sk-test",
        } } };
        try std.testing.expectEqualStrings("sk-test", envApiKey(without_url, .openai).?);
        try std.testing.expect(envApiKey(without_url, .openai_compatible) == null);
    }
}

test "Client.init: each backend's settings reach its client, shared options reach every one" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const extra = [_]std.http.Header{.{ .name = "X-Extra", .value = "v" }};

    try std.testing.expectError(error.MissingBaseUrl, Client.init(io, a, .openai_compatible, "k", .{}));

    const vertex = try Client.init(io, a, .{ .vertex = .{ .project = "p", .location = "europe-west4" } }, "tok", .{ .extra_headers = &extra });
    defer vertex.deinit(a);
    try std.testing.expectEqualStrings("p", vertex.vertex.vertex.?.project.?);
    try std.testing.expectEqualStrings("europe-west4", vertex.vertex.vertex.?.location);
    try std.testing.expectEqualSlices(std.http.Header, &extra, vertex.transport().extra_headers);

    const codex = try Client.init(io, a, .{ .codex = .{ .account_id = "acct" } }, "tok", .{});
    defer codex.deinit(a);
    try std.testing.expectEqualStrings("acct", codex.codex.chatgpt.?.account_id.?);
    try std.testing.expectEqualStrings(openai_mod.codex_base_url, codex.codex.base_url);

    const claude = try Client.init(io, a, .anthropic, "k", .{ .extra_headers = &extra });
    defer claude.deinit(a);
    try std.testing.expectEqualSlices(std.http.Header, &extra, claude.transport().extra_headers);

    const gemini = try Client.init(io, a, Backend.of(.gemini), "k", .{});
    defer gemini.deinit(a);
    try std.testing.expect(gemini.gemini.vertex == null);
}

test "fromEnv: key, vertex settings and the openai_compatible base URL" {
    try std.testing.expectEqual(Backend.of(.anthropic), fromEnv(.empty, .anthropic).backend);
    try std.testing.expectEqualStrings("global", fromEnv(.empty, .vertex).backend.vertex.location);
    try std.testing.expect(fromEnv(.empty, .openai_compatible).base_url == null);
    try std.testing.expectEqualStrings("ollama", fromEnv(.empty, .ollama).key.?);
    // See the OPENAI_BASE_URL test above for why this is POSIX-only.
    if (comptime builtin.os.tag != .windows) {
        const environ: std.process.Environ = .{ .block = .{ .slice = &[_:null]?[*:0]const u8{
            "OPENAI_BASE_URL=http://localhost:8000/v1",
            "OPENAI_API_KEY=sk-test",
            "GOOGLE_CLOUD_PROJECT=proj",
            "GOOGLE_CLOUD_REGION=us-east5",
        } } };
        const compat = fromEnv(environ, .openai_compatible);
        try std.testing.expectEqualStrings("sk-test", compat.key.?);
        try std.testing.expectEqualStrings("http://localhost:8000/v1", compat.base_url.?);
        try std.testing.expect(fromEnv(environ, .openai).base_url == null);
        const vertex = fromEnv(environ, .vertex).backend.vertex;
        try std.testing.expectEqualStrings("proj", vertex.project.?);
        try std.testing.expectEqualStrings("us-east5", vertex.location);
    }
}

test "openai_compatible: envVarName and defaultModel" {
    try std.testing.expectEqualStrings("OPENAI_API_KEY", envVarName(.openai_compatible));
    try std.testing.expectEqualStrings("", defaultModel(.openai_compatible));
}

test "openai_compatible: present in default_candidates (runtime-gated)" {
    var found = false;
    for (default_candidates) |t| {
        if (t == .openai_compatible) found = true;
    }
    try std.testing.expect(found);
}

test "omitsEffort: Mistral only" {
    try std.testing.expect(omitsEffort(.mistral));
    try std.testing.expect(!omitsEffort(.anthropic));
    try std.testing.expect(!omitsEffort(.openai));
}

test "reasoning_effort: omitted for a null effort, sent otherwise" {
    const tools: ?[]const openai_types.Tool = null;
    try std.testing.expectEqual(@as(?openai_types.ReasoningEffort, null), mapOpenAICompletionConfig(.{ .effort = null }, tools).reasoning_effort);
    try std.testing.expectEqual(@as(?openai_types.ReasoningEffort, .none), mapOpenAICompletionConfig(.{ .effort = .none }, tools).reasoning_effort);
    try std.testing.expectEqual(@as(?openai_types.ReasoningEffort, .low), mapOpenAICompletionConfig(.{ .effort = .low }, tools).reasoning_effort);
}

test "responsesRequest: xAI never sends none, OpenAI does" {
    const xai: Client = .{ .xai = undefined };
    try std.testing.expect(xai.responsesRequest("grok-4.7", &.{}, null, .{}).reasoning == null);
    try std.testing.expectEqual(openai_types.ReasoningEffort.low, xai.responsesRequest("grok-4.7", &.{}, null, .{ .effort = .none }).reasoning.?.effort);
    try std.testing.expectEqual(openai_types.ReasoningEffort.xhigh, xai.responsesRequest("grok-4.7", &.{}, null, .{ .effort = .xhigh }).reasoning.?.effort);
    // Per-chunk streaming still goes through Chat Completions, with the same mapping.
    try std.testing.expectEqual(@as(?openai_types.ReasoningEffort, .low), xai.openAiCompletionConfig("grok-4.7", .{ .effort = .none }, null).reasoning_effort);

    const oai: Client = .{ .openai = undefined };
    const req = oai.responsesRequest("gpt-5.5", &.{}, null, .{ .effort = .none });
    try std.testing.expectEqual(openai_types.ReasoningEffort.none, req.reasoning.?.effort);
    // Stateless: encrypted reasoning is requested and carried in `input`.
    try std.testing.expectEqual(@as(?bool, false), req.store);
    try std.testing.expectEqualStrings("reasoning.encrypted_content", req.include.?[0]);
}

test "responsesRequest: codex body matches the former Codex client" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const codex: Client = .{ .codex = undefined };
    const msgs = [_]Message{ .{ .role = .system, .content = "be brief" }, .{ .role = .user, .content = "hi" } };
    const input = try messagesToOpenAIResponsesInput(a, &msgs, codex.responsesSystemRole(), codex, "gpt-5.5");
    const tools = try mapOpenAIResponsesTools(a, &.{.{ .name = "search", .description = "web", .parameters = .null }});
    const req = codex.responsesRequest("gpt-5.5", input, tools, .{ .effort = .high, .max_tokens = 1000, .temperature = 0.5, .tool_choice = .auto });
    // Captured from the standalone Codex client's request builder.
    try std.testing.expectEqualStrings(
        \\{"model":"gpt-5.5","input":[{"type":"message","role":"developer","content":"be brief"},{"type":"message","role":"user","content":"hi"}],"tools":[{"type":"function","name":"search","description":"web","parameters":null}],"tool_choice":"auto","reasoning":{"effort":"high","summary":"auto"},"temperature":0.5,"store":false,"include":["reasoning.encrypted_content"]}
    , try stringifyForTest(a, req));

    // OpenAI keeps max_output_tokens and the system role, and asks for no summary.
    const oai: Client = .{ .openai = undefined };
    const oai_req = oai.responsesRequest("gpt-5.5", input, tools, .{ .effort = .high, .max_tokens = 1000 });
    try std.testing.expectEqual(@as(?i32, 1000), oai_req.max_output_tokens);
    try std.testing.expectEqual(@as(?[]const u8, null), oai_req.reasoning.?.summary);
    try std.testing.expectEqualStrings("system", oai.responsesSystemRole());
}

test "openAiResponsesResult: keeps reasoning with its origin, skips ciphertext-less items" {
    const output = [_]openai_types.ResponseOutputItem{
        .{ .type = "reasoning", .id = "rs_1", .summary = &.{.{ .text = "plan" }}, .encrypted_content = "enc1" },
        .{ .type = "reasoning", .id = "rs_2" },
        .{ .type = "function_call", .call_id = "call_1", .name = "search", .arguments = "{}" },
    };
    var result: GenerateResult = .init(std.testing.allocator);
    defer result.deinit();
    try openAiResponsesResult(&result, result.arena.allocator(), .{ .output = &output }, .{ .xai = undefined }, "grok-4.7");

    const r = result.reasoning.?;
    try std.testing.expectEqual(@as(usize, 1), r.len);
    try std.testing.expectEqualStrings("rs_1", r[0].id);
    try std.testing.expectEqualStrings("enc1", r[0].encrypted_content);
    try std.testing.expectEqualStrings("plan", r[0].summary[0]);
    try std.testing.expectEqual(Tag.xai, r[0].provider);
    try std.testing.expectEqualStrings("grok-4.7", r[0].model);
    try std.testing.expectEqual(@as(usize, 1), result.tool_calls.?.len);
}

test "messagesToOpenAIResponsesInput: echoes reasoning only to its own provider and model" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const reasoning = [_]ReasoningItem{.{ .id = "rs_1", .encrypted_content = "enc1", .provider = .xai, .model = "grok-4.7" }};
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "search" }};
    const results = [_]ToolResult{.{ .id = "call_1", .name = "search", .content = "ok" }};
    const msgs = [_]Message{
        .{ .role = .user, .content = "hi" },
        .{ .role = .assistant, .tool_calls = &calls, .reasoning = &reasoning },
        .{ .role = .tool, .tool_results = &results },
    };

    const same = try messagesToOpenAIResponsesInput(a, &msgs, "system", .{ .xai = undefined }, "grok-4.7");
    try std.testing.expectEqual(@as(usize, 4), same.len);
    // Reasoning precedes the function call it produced.
    try std.testing.expectEqualStrings("reasoning", same[1].type);
    try std.testing.expectEqualStrings("function_call", same[2].type);
    try std.testing.expectEqualStrings("{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[],\"encrypted_content\":\"enc1\"}", try stringifyForTest(a, same[1]));

    for ([_]struct { Client, []const u8 }{
        .{ .{ .openai = undefined }, "grok-4.7" },
        .{ .{ .xai = undefined }, "grok-4.6" },
    }) |other| {
        const items = try messagesToOpenAIResponsesInput(a, &msgs, "system", other[0], other[1]);
        try std.testing.expectEqual(@as(usize, 3), items.len);
        for (items) |it| try std.testing.expect(!std.mem.eql(u8, it.type, "reasoning"));
    }
}

test "dupeMessage: deep-copies reasoning" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const summary = [_][]const u8{"plan"};
    const reasoning = [_]ReasoningItem{.{ .id = "rs_1", .encrypted_content = "enc1", .summary = &summary, .provider = .openai, .model = "gpt-5.5" }};
    const out = try dupeMessage(arena.allocator(), .{ .role = .assistant, .reasoning = &reasoning });
    const r = out.reasoning.?[0];
    try std.testing.expectEqualStrings("enc1", r.encrypted_content);
    try std.testing.expectEqualStrings("plan", r.summary[0]);
    try std.testing.expectEqual(Tag.openai, r.provider);
    try std.testing.expect(r.encrypted_content.ptr != reasoning[0].encrypted_content.ptr);
}

test "openAiCompletionConfig: Mistral sends only none or high" {
    const mistral: Client = .{ .mistral = undefined };
    try std.testing.expectEqual(@as(?openai_types.ReasoningEffort, null), mistral.openAiCompletionConfig("mistral-large-4", .{}, null).reasoning_effort);
    try std.testing.expectEqual(@as(?openai_types.ReasoningEffort, .none), mistral.openAiCompletionConfig("mistral-large-4", .{ .effort = .none }, null).reasoning_effort);
    try std.testing.expectEqual(@as(?openai_types.ReasoningEffort, .high), mistral.openAiCompletionConfig("mistral-large-4", .{ .effort = .low }, null).reasoning_effort);
    try std.testing.expectEqual(@as(?openai_types.ReasoningEffort, .high), mistral.openAiCompletionConfig("mistral-large-4", .{ .effort = .xhigh }, null).reasoning_effort);
}

test "openAiCompletionConfig: Anthropic cache hints per gateway" {
    const or_client: Client = .{ .openrouter = undefined };
    try std.testing.expect(or_client.openAiCompletionConfig("anthropic/claude-sonnet-5", .{}, null).cache_control != null);
    try std.testing.expect(or_client.openAiCompletionConfig("openai/gpt-5.5", .{}, null).cache_control == null);

    const vercel: Client = .{ .vercel = undefined };
    const cfg = vercel.openAiCompletionConfig("anthropic/claude-sonnet-5", .{}, null);
    try std.testing.expectEqualStrings("auto", cfg.providerOptions.?.gateway.caching);
    try std.testing.expect(cfg.cache_control == null);

    const compat: Client = .{ .openai_compatible = undefined };
    const plain = compat.openAiCompletionConfig("anthropic/claude-sonnet-5", .{}, null);
    try std.testing.expect(plain.cache_control == null and plain.providerOptions == null);
}

test "mapOpenAIUsage: cache writes come out of the fresh prompt count" {
    const openrouter = mapOpenAIUsage(.{ .usage = .{
        .prompt_tokens = 1000,
        .prompt_tokens_details = .{ .cached_tokens = 600, .cache_write_tokens = 300 },
    } });
    try std.testing.expectEqual(@as(?i32, 100), openrouter.prompt_tokens);
    try std.testing.expectEqual(@as(?i32, 600), openrouter.cached_tokens);
    try std.testing.expectEqual(@as(?i32, 300), openrouter.cache_creation_tokens);

    const vercel = mapOpenAIUsage(.{ .usage = .{
        .prompt_tokens = 8409,
        .prompt_tokens_details = .{ .cached_tokens = 0 },
        .cache_creation_input_tokens = 8406,
    } });
    try std.testing.expectEqual(@as(?i32, 3), vercel.prompt_tokens);
    try std.testing.expectEqual(@as(?i32, 8406), vercel.cache_creation_tokens);
    try std.testing.expectEqual(@as(i32, 8409), vercel.inputTokens());
}

test "mapOpenAIResponsesUsage: cache writes come out of the fresh input count" {
    const usage = mapOpenAIResponsesUsage(.{ .usage = .{
        .input_tokens = 1000,
        .input_tokens_details = .{ .cached_tokens = 600, .cache_write_tokens = 300 },
    } });
    try std.testing.expectEqual(@as(?i32, 100), usage.prompt_tokens);
    try std.testing.expectEqual(@as(?i32, 600), usage.cached_tokens);
    try std.testing.expectEqual(@as(?i32, 300), usage.cache_creation_tokens);
    try std.testing.expectEqual(@as(i32, 1000), usage.inputTokens());
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
                        .args = .{ .text = call.arguments },
                    },
                    .thoughtSignature = call.thought_signature,
                });
            }
        }

        if (msg.tool_results) |results| {
            for (results) |res| {
                try parts.append(allocator, .{
                    .functionResponse = .{
                        .id = res.id,
                        .name = res.name,
                        .response = if (res.content.len > 0) try gemini_types.ResponseBody.fromToolOutput(allocator, res.content) else null,
                    },
                    .thoughtSignature = res.thought_signature,
                });
                for (res.parts orelse &.{}) |cp| switch (cp) {
                    .image => |img| try parts.append(allocator, .{ .inlineData = .{ .data = img.data, .mimeType = img.mime_type } }),
                    .text => {},
                };
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

/// The OpenAI-style APIs accept only text in a tool message. Move each tool
/// turn's image parts into a user turn right after it, captioned with the
/// tool's name; the tool message keeps `content`. Returns the input when no
/// tool result carries an image.
fn splitToolImages(allocator: std.mem.Allocator, messages: []const Message) ![]const Message {
    var any = false;
    for (messages) |msg| {
        for (msg.tool_results orelse &.{}) |res| any = any or hasImage(res.parts orelse &.{});
    }
    if (!any) return messages;

    var out: std.ArrayList(Message) = .empty;
    for (messages) |msg| {
        const results = msg.tool_results orelse {
            try out.append(allocator, msg);
            continue;
        };
        const stripped = try allocator.alloc(ToolResult, results.len);
        var images: std.ArrayList(ContentPart) = .empty;
        for (results, 0..) |res, i| {
            stripped[i] = res;
            stripped[i].parts = null;
            if (!hasImage(res.parts orelse &.{})) continue;
            try images.append(allocator, .{ .text = try allocator.print("Image returned by {s}", .{res.name}) });
            for (res.parts.?) |cp| {
                if (cp == .image) try images.append(allocator, cp);
            }
        }
        var tool_msg = msg;
        tool_msg.tool_results = stripped;
        try out.append(allocator, tool_msg);
        if (images.items.len > 0) {
            try out.append(allocator, .{ .role = .user, .parts = try images.toOwnedSlice(allocator) });
        }
    }
    return out.toOwnedSlice(allocator);
}

pub fn hasImage(parts: []const ContentPart) bool {
    for (parts) |cp| if (cp == .image) return true;
    return false;
}

fn openAIContentParts(allocator: std.mem.Allocator, parts: []const ContentPart) ![]const openai_types.ContentPart {
    const out = try allocator.alloc(openai_types.ContentPart, parts.len);
    for (parts, 0..) |cp, i| out[i] = switch (cp) {
        .text => |t| .{ .type = "text", .text = t },
        .image => |img| .{ .type = "image_url", .image_url = .{ .url = .{ .mime_type = img.mime_type, .data = img.data } } },
    };
    return out;
}

fn responsesContentParts(allocator: std.mem.Allocator, parts: []const ContentPart) ![]const openai_types.ResponseContentPart {
    const out = try allocator.alloc(openai_types.ResponseContentPart, parts.len);
    for (parts, 0..) |cp, i| out[i] = switch (cp) {
        .text => |t| .{ .type = "input_text", .text = t },
        .image => |img| .{ .type = "input_image", .image_url = .{ .mime_type = img.mime_type, .data = img.data } },
    };
    return out;
}

fn anthropicBlocks(allocator: std.mem.Allocator, parts: []const ContentPart) ![]const anthropic_types.ContentBlockParam {
    const out = try allocator.alloc(anthropic_types.ContentBlockParam, parts.len);
    for (parts, 0..) |cp, i| out[i] = switch (cp) {
        .text => |t| .{ .type = "text", .text = t },
        .image => |img| .{ .type = "image", .source = .{ .media_type = img.mime_type, .data = img.data } },
    };
    return out;
}

/// Ollama carries images beside the text, as bare base64.
fn base64Images(allocator: std.mem.Allocator, parts: []const ContentPart) !?[]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (parts) |cp| switch (cp) {
        .image => |img| try out.append(allocator, img.data),
        .text => {},
    };
    if (out.items.len == 0) return null;
    return try out.toOwnedSlice(allocator);
}

fn messagesToOpenAIMessages(allocator: std.mem.Allocator, input: []const Message) ![]openai_types.Message {
    const messages = try splitToolImages(allocator, input);
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

        try out.append(allocator, .{
            .role = switch (msg.role) {
                .system => .system,
                .user => .user,
                .assistant => .assistant,
                .tool => unreachable,
            },
            .content = messageText(msg),
            .content_parts = if (msg.parts) |parts| try openAIContentParts(allocator, parts) else null,
            .tool_calls = oai_tool_calls,
        });
    }

    return out.toOwnedSlice(allocator);
}

/// Like `messagesToOpenAIMessages`, but writes tool-call arguments as objects
/// (not strings) and answers tool results with `tool_name` (Ollama matches
/// by name, not id).
fn messagesToOllamaMessages(allocator: std.mem.Allocator, input: []const Message) ![]ollama_mod.Message {
    const messages = try splitToolImages(allocator, input);
    var out: std.ArrayList(ollama_mod.Message) = .empty;

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

        var native_tool_calls: ?[]const ollama_mod.ToolCall = null;
        if (msg.tool_calls) |calls| {
            const arr = try allocator.alloc(ollama_mod.ToolCall, calls.len);
            for (calls, 0..) |call, i| {
                arr[i] = .{
                    .id = if (call.id.len > 0) call.id else null,
                    .type = "function",
                    .function = .{ .name = call.name, .arguments = .{ .text = call.arguments } },
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
            .images = if (msg.parts) |parts| try base64Images(allocator, parts) else null,
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
            try blocks.appendSlice(allocator, try anthropicBlocks(allocator, content_parts));
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
                    // no-arg call's arguments are `{}`.
                    .input = .{ .text = call.arguments },
                });
            }
        }

        if (msg.tool_results) |results| {
            for (results) |res| {
                try blocks.append(allocator, .{
                    .type = "tool_result",
                    .tool_use_id = res.id,
                    .content = try anthropicToolResultContent(allocator, res),
                    .is_error = if (res.is_error) true else null,
                });
            }
        }

        const role: anthropic_types.Role = if (msg.role == .assistant) .assistant else .user;
        try out.append(allocator, .{ .role = role, .content = try blocks.toOwnedSlice(allocator) });
    }
    return out.toOwnedSlice(allocator);
}

fn anthropicToolResultContent(allocator: std.mem.Allocator, res: ToolResult) !anthropic_types.ToolResultContent {
    const parts = res.parts orelse return .{ .text = res.content };
    return .{ .blocks = try anthropicBlocks(allocator, parts) };
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

fn noopOnText(_: *anyopaque, _: []const u8) void {}

/// A backend stream error, collapsed into `Client.Error`; the detail is in
/// the backend's `transport.last_error` (see `lastError`).
fn streamFailed(err: anyerror) Client.Error {
    return if (err == error.OutOfMemory) error.OutOfMemory else error.ApiError;
}

/// Flatten the normalized conversation into Responses-API `input` items: each
/// assistant tool call becomes a `function_call` item and each tool result a
/// `function_call_output` item, rather than the role-tagged messages chat
/// completions uses. Reasoning from `client`/`model` is echoed ahead of the
/// output it produced; reasoning from any other origin is dropped.
fn messagesToOpenAIResponsesInput(allocator: std.mem.Allocator, input: []const Message, system_role: []const u8, client: Client, model: []const u8) ![]openai_types.ResponseInputItem {
    const messages = try splitToolImages(allocator, input);
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

        if (msg.reasoning) |items| for (items) |r| {
            if (r.provider != client or !std.mem.eql(u8, r.model, model)) continue;
            const summary = try allocator.alloc(openai_types.ResponseSummaryText, r.summary.len);
            for (r.summary, summary) |t, *d| d.* = .{ .text = t };
            try out.append(allocator, .{
                .type = "reasoning",
                .id = r.id,
                .summary = summary,
                .encrypted_content = r.encrypted_content,
            });
        };

        const content: ?openai_types.ResponseContent = if (msg.parts) |parts|
            .{ .parts = try responsesContentParts(allocator, parts) }
        else if (msg.content) |c|
            .{ .text = c }
        else
            null;

        if (content) |c| {
            try out.append(allocator, .{
                .type = "message",
                .role = switch (msg.role) {
                    .system => system_role,
                    .user => "user",
                    .assistant => "assistant",
                    .tool => unreachable,
                },
                .content = c,
            });
        }

        if (msg.tool_calls) |calls| {
            for (calls) |call| {
                // Responses requires a valid JSON string; a no-arg call's is `{}`.
                try out.append(allocator, .{
                    .type = "function_call",
                    .call_id = call.id,
                    .name = call.name,
                    .arguments = call.arguments,
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
    const written = if (usage.input_tokens_details) |d| d.cache_write_tokens else null;
    return .{
        .prompt_tokens = freshPrompt(usage.input_tokens, addOpt(cached, written)),
        .completion_tokens = usage.output_tokens,
        .total_tokens = usage.total_tokens,
        .cached_tokens = cached,
        .cache_creation_tokens = written,
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
                    schema.properties = .init(prop_arr);
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
        .MAX_TOKENS, .CONTINUATION => .max_tokens,
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
    // Both gateways count cache writes in `prompt_tokens`.
    const written = (if (usage.prompt_tokens_details) |d| d.cache_write_tokens else null) orelse usage.cache_creation_input_tokens;
    return .{
        .prompt_tokens = freshPrompt(usage.prompt_tokens, addOpt(cached, written)),
        .completion_tokens = usage.completion_tokens,
        .total_tokens = usage.total_tokens,
        .cached_tokens = cached,
        .cache_creation_tokens = written,
    };
}

/// Ollama's native `think` is a boolean toggle; map any requested level to on,
/// `.none` to an explicit off, and an unset level to the model default (null).
fn mapOllamaThink(level: ?Effort) ?bool {
    const l = level orelse return null;
    return l != .none;
}

fn mapOllamaFinishReason(response: ollama_mod.ChatResponse) FinishReason {
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

fn mapOllamaUsage(response: ollama_mod.ChatResponse) Usage {
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

/// Native `/api/chat` request inputs mapped from a `GenerationConfig`. Shared by
/// the buffered and streaming `.ollama` arms.
const OllamaCall = struct {
    messages: []ollama_mod.Message,
    tools: ?[]const openai_types.Tool,
    think: ?bool,
    format: ?[]const u8,
    options: ollama_mod.Options,
};

fn mapOllamaCall(allocator: std.mem.Allocator, messages: []const Message, config: GenerationConfig) !OllamaCall {
    // Native `/api/chat` has no tool_choice; honor `.none` by withholding tools.
    const suppress_tools = if (config.tool_choice) |tc| tc == .none else false;
    return .{
        .messages = try messagesToOllamaMessages(allocator, messages),
        .tools = if (suppress_tools) null else if (config.tools) |t| try mapOpenAITools(allocator, t) else null,
        .think = mapOllamaThink(config.effort),
        .format = if (config.response_format) |rf| (if (rf == .json) "json" else null) else null,
        .options = .{
            .num_predict = config.max_tokens,
            .temperature = config.temperature,
            .top_p = config.top_p,
            .seed = config.seed,
            .stop = config.stop,
            .frequency_penalty = config.frequency_penalty,
            .presence_penalty = config.presence_penalty,
        },
    };
}

/// Collects a response's tool calls, copied into `out`.
const ToolCalls = struct {
    out: std.mem.Allocator,
    list: std.ArrayList(ToolCall) = .empty,

    /// Arguments arrive parsed (most APIs) or as a JSON string (OpenAI);
    /// either way they're stored as JSON text (see `ToolCall.arguments`).
    const Args = union(enum) { value: ?json.ValueOrText, text: ?[]const u8 };

    fn add(self: *ToolCalls, id: ?[]const u8, name: ?[]const u8, args: Args, thought_signature: ?[]const u8) error{OutOfMemory}!void {
        try self.list.append(self.out, .{
            .id = if (id) |v| try self.out.dupe(u8, v) else "",
            .name = if (name) |v| try self.out.dupe(u8, v) else "",
            .arguments = switch (args) {
                .value => |v| if (v) |val| switch (val) {
                    .value => |tree| try json.valueToString(self.out, tree),
                    .text => |t| try self.argumentsText(t),
                } else "{}",
                .text => |t| try self.argumentsText(t orelse ""),
            },
            .thought_signature = if (thought_signature) |v| try self.out.dupe(u8, v) else null,
        });
    }

    /// The model's text when it is valid JSON, else `{}`: other backends
    /// write it into their request verbatim.
    fn argumentsText(self: *ToolCalls, text: []const u8) error{OutOfMemory}![]const u8 {
        const trimmed = std.mem.trim(u8, text, " \t\r\n");
        if (trimmed.len == 0 or !try std.json.validate(self.out, trimmed)) return "{}";
        return self.out.dupe(u8, trimmed);
    }

    fn slice(self: *ToolCalls) error{OutOfMemory}!?[]const ToolCall {
        return if (self.list.items.len == 0) null else try self.list.toOwnedSlice(self.out);
    }
};

/// Map a native `/api/chat` `ChatResponse` (buffered or stream-reassembled) into
/// `result`, copying into `out`. Shared by the buffered and streaming arms.
fn ollamaResult(result: *GenerateResult, out: std.mem.Allocator, response: ollama_mod.ChatResponse) Client.Error!void {
    result.finish_reason = mapOllamaFinishReason(response);
    result.usage = mapOllamaUsage(response);

    if (response.message) |msg| {
        if (msg.content) |c| {
            if (c.len > 0) result.text = try out.dupe(u8, c);
        }
        if (msg.tool_calls) |calls| {
            var tool_calls: ToolCalls = .{ .out = out };
            for (calls) |call| {
                const f = call.function orelse continue;
                try tool_calls.add(call.id, f.name, .{ .value = f.arguments }, null);
            }
            result.tool_calls = try tool_calls.slice();
        }
    }
}

/// Map a Gemini `GenerateContentResponse` (buffered or stream-reassembled) into
/// `result`, copying into `out`. Shared by the buffered and streaming arms.
fn geminiResult(result: *GenerateResult, out: std.mem.Allocator, response: gemini_types.GenerateContentResponse) Client.Error!void {
    if (response.text()) |t| result.text = try out.dupe(u8, t);
    result.finish_reason = mapGeminiFinishReason(response);
    result.usage = mapGeminiUsage(response);
    if (response.candidates) |candidates| {
        if (candidates.len > 0) {
            if (candidates[0].content) |content| {
                var tool_calls: ToolCalls = .{ .out = out };
                for (content.parts) |p| {
                    const fc = p.functionCall orelse continue;
                    try tool_calls.add(fc.id, fc.name, .{ .value = fc.args }, p.thoughtSignature);
                }
                result.tool_calls = try tool_calls.slice();
            }
        }
    }
}

/// Map an Anthropic `MessageResponse` (buffered or stream-reassembled) into
/// `result`, copying into `out`. Shared by the buffered and streaming arms so
/// both extract text/usage/tool_use identically.
fn anthropicResult(result: *GenerateResult, out: std.mem.Allocator, response: anthropic_types.MessageResponse) Client.Error!void {
    if (response.text()) |t| result.text = try out.dupe(u8, t);
    result.finish_reason = mapAnthropicFinishReason(response);
    result.usage = mapAnthropicUsage(response);
    if (response.content) |blocks| {
        var tool_calls: ToolCalls = .{ .out = out };
        for (blocks) |block| {
            const t = block.type orelse continue;
            if (!std.mem.eql(u8, t, "tool_use")) continue;
            try tool_calls.add(block.id, block.name, .{ .value = if (block.input) |v| .{ .value = v } else null }, null);
        }
        result.tool_calls = try tool_calls.slice();
    }
}

/// Map an OpenAI-compatible `ChatCompletionResponse` (buffered or
/// stream-reassembled) into `result`, copying into `out`. Shared by the
/// buffered and streaming arms of the Chat Completions providers.
fn openAiChatResult(result: *GenerateResult, out: std.mem.Allocator, response: openai_types.ChatCompletionResponse) Client.Error!void {
    if (response.text()) |t| {
        if (t.len > 0) result.text = try out.dupe(u8, t);
    }
    result.finish_reason = mapOpenAIFinishReason(response);
    result.usage = mapOpenAIUsage(response);

    const choices = response.choices orelse return;
    if (choices.len == 0) return;
    const msg = choices[0].message orelse return;
    const calls = msg.tool_calls orelse return;

    var tool_calls: ToolCalls = .{ .out = out };
    for (calls) |call| {
        const f = call.function orelse continue;
        try tool_calls.add(call.id, f.name, .{ .text = f.arguments }, null);
    }
    result.tool_calls = try tool_calls.slice();
}

/// Map an OpenAI `ResponsesResponse` (buffered or stream-reassembled) into
/// `result`, copying into `out`. Shared by the buffered and streaming arms.
fn openAiResponsesResult(result: *GenerateResult, out: std.mem.Allocator, response: openai_types.ResponsesResponse, client: Client, model: []const u8) Client.Error!void {
    if (response.text()) |t| {
        if (t.len > 0) result.text = try out.dupe(u8, t);
    }
    result.finish_reason = mapOpenAIResponsesFinishReason(response);
    result.usage = mapOpenAIResponsesUsage(response);

    const items = response.output orelse return;
    var tool_calls: ToolCalls = .{ .out = out };
    var reasoning: std.ArrayList(ReasoningItem) = .empty;
    for (items) |item| {
        const item_type = item.type orelse continue;
        if (std.mem.eql(u8, item_type, "reasoning")) {
            // Without ciphertext (or an id to pair it with) there is nothing to echo.
            const id = item.id orelse continue;
            const enc = item.encrypted_content orelse continue;
            const src = item.summary orelse &.{};
            const summary = try out.alloc([]const u8, src.len);
            for (src, summary) |st, *d| d.* = try out.dupe(u8, st.text);
            try reasoning.append(out, .{
                .id = try out.dupe(u8, id),
                .encrypted_content = try out.dupe(u8, enc),
                .summary = summary,
                .provider = client,
                .model = try out.dupe(u8, model),
            });
            continue;
        }
        if (!std.mem.eql(u8, item_type, "function_call")) continue;
        try tool_calls.add(item.call_id, item.name, .{ .text = item.arguments }, null);
    }
    result.tool_calls = try tool_calls.slice();
    if (reasoning.items.len > 0) result.reasoning = try reasoning.toOwnedSlice(out);
}

fn mapAnthropicStopReason(reason: anthropic_types.StopReason) FinishReason {
    return switch (reason) {
        .end_turn, .stop_sequence => .stop,
        .max_tokens, .pause_turn, .model_context_window_exceeded => .max_tokens,
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
    return std.mem.find(u8, model, "gemini-2.5") != null;
}

fn mapEffortToGeminiConfig(model: []const u8, level: Effort) gemini_types.ThinkingConfig {
    if (geminiUsesThinkingBudget(model)) {
        return .{ .thinkingBudget = mapEffortToGeminiBudget(level) };
    }
    return .{ .thinkingLevel = mapEffortToGemini(level) };
}

/// `.none` does not turn reasoning off on Gemini 3: null serializes as
/// `"thinkingConfig": {}` (requests omit null fields), which is the model
/// default. Only Gemini 2.5's `thinkingBudget: 0` is a real off switch.
/// Some models also reject a level outright -- gemini-3.8-flash answers
/// `MINIMAL` with HTTP 400 where `.low` works.
fn mapEffortToGemini(level: Effort) ?gemini_types.ThinkingLevel {
    return switch (level) {
        .none => null,
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
        .reasoning_effort = if (config.effort) |tl| mapEffortToOpenAI(tl) else null,
    };
}

/// Mistral accepts only "none" and "high".
fn mapEffortToMistral(level: Effort) openai_types.ReasoningEffort {
    return switch (level) {
        .none => .none,
        .minimal, .low, .medium, .high, .xhigh => .high,
    };
}

/// Grok reasoning can't be disabled: it takes low, medium, high and xhigh.
fn mapEffortToXai(level: Effort) openai_types.ReasoningEffort {
    return switch (level) {
        .none, .minimal, .low => .low,
        .medium => .medium,
        .high => .high,
        .xhigh => .xhigh,
    };
}

const AnthropicReasoning = struct {
    thinking: ?anthropic_types.ThinkingConfig,
    output_config: ?anthropic_types.OutputConfig,
    max_tokens: i32,
};

/// Reasoning tokens share the output budget with the answer, so a tight cap
/// yields mostly thinking followed by a truncated answer.
fn floorMaxTokens(effort: ?Effort, max_tokens: i32) i32 {
    const answer_room = 4096;
    const floor: i32 = switch (effort orelse return max_tokens) {
        .none => return max_tokens,
        .minimal, .low => 1024 + answer_room,
        .medium => 4096 + answer_room,
        .high => 16384 + answer_room,
        .xhigh => 32768 + answer_room,
    };
    return @max(max_tokens, floor);
}

/// Null `max_tokens` means "provider default", which is generous, so it is
/// left alone rather than pinned to the floor.
fn flooredConfig(config: GenerationConfig) GenerationConfig {
    var floored = config;
    if (config.max_tokens) |requested| floored.max_tokens = floorMaxTokens(config.effort, requested);
    return floored;
}

/// Adaptive thinking + `output_config.effort` is the only on-mode on current
/// models (`enabled` + `budget_tokens` is HTTP 400 on Opus 4.7+/Sonnet 5);
/// legacy models (Sonnet 4.5, Haiku 4.5) reject adaptive/effort instead.
/// Deliberately no model sniffing: the API's error surfaces as-is, and
/// a null effort (omit both fields) works everywhere.
fn mapEffortToAnthropic(effort: ?Effort, max_tokens: i32) AnthropicReasoning {
    const off: AnthropicReasoning = .{ .thinking = null, .output_config = null, .max_tokens = max_tokens };
    // Omitting `thinking` runs adaptive thinking on Sonnet 5 / Opus 5, so an
    // explicit `.none` must say so; models that can't disable it return a 400.
    const disabled: AnthropicReasoning = .{ .thinking = .{ .type = "disabled" }, .output_config = null, .max_tokens = max_tokens };
    const wire: []const u8 = switch (effort orelse return off) {
        .none => return disabled,
        .minimal, .low => "low",
        .medium => "medium",
        .high => "high",
        .xhigh => "xhigh",
    };
    return .{
        .thinking = .{ .type = "adaptive" },
        .output_config = .{ .effort = wire },
        .max_tokens = floorMaxTokens(effort, max_tokens),
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

test "floorMaxTokens: a tight cap gets reasoning headroom, a generous one is left alone" {
    const cases = .{
        .{ .effort = @as(?Effort, .low), .in = @as(i32, 256), .want = @as(i32, 5120) },
        .{ .effort = @as(?Effort, .medium), .in = @as(i32, 256), .want = @as(i32, 8192) },
        .{ .effort = @as(?Effort, .high), .in = @as(i32, 256), .want = @as(i32, 20480) },
        .{ .effort = @as(?Effort, .xhigh), .in = @as(i32, 256), .want = @as(i32, 36864) },
        .{ .effort = @as(?Effort, .medium), .in = @as(i32, 64000), .want = @as(i32, 64000) },
        .{ .effort = @as(?Effort, .none), .in = @as(i32, 256), .want = @as(i32, 256) },
        .{ .effort = @as(?Effort, null), .in = @as(i32, 256), .want = @as(i32, 256) },
    };
    inline for (cases) |c| try std.testing.expectEqual(c.want, floorMaxTokens(c.effort, c.in));
}

test "flooredConfig: null max_tokens stays the provider default" {
    // Pinning it to the floor would *lower* the budget where the default is larger.
    try std.testing.expectEqual(@as(?i32, null), flooredConfig(.{ .effort = .high }).max_tokens);
    try std.testing.expectEqual(@as(?i32, 20480), flooredConfig(.{ .effort = .high, .max_tokens = 256 }).max_tokens);
}

test "mapEffortToAnthropic: null omits thinking, none disables it" {
    const off = mapEffortToAnthropic(null, 4096);
    try std.testing.expect(off.thinking == null);
    try std.testing.expect(off.output_config == null);
    try std.testing.expectEqual(@as(i32, 4096), off.max_tokens);

    const none = mapEffortToAnthropic(.none, 4096);
    try std.testing.expectEqualStrings("disabled", none.thinking.?.type);
    try std.testing.expect(none.output_config == null);
    try std.testing.expectEqual(@as(i32, 4096), none.max_tokens);
}

test "mapEffortToAnthropic pairs adaptive thinking with effort and headroom" {
    const medium = mapEffortToAnthropic(.medium, 4096);
    try std.testing.expectEqualStrings("adaptive", medium.thinking.?.type);
    try std.testing.expectEqualStrings("medium", medium.output_config.?.effort.?);
    try std.testing.expectEqual(@as(i32, 8192), medium.max_tokens);

    try std.testing.expectEqualStrings("low", mapEffortToAnthropic(.minimal, 4096).output_config.?.effort.?);
    try std.testing.expectEqualStrings("low", mapEffortToAnthropic(.low, 4096).output_config.?.effort.?);
    try std.testing.expectEqualStrings("high", mapEffortToAnthropic(.high, 4096).output_config.?.effort.?);
    try std.testing.expectEqual(@as(i32, 36864), mapEffortToAnthropic(.xhigh, 4096).max_tokens);
}

test "mapEffortToAnthropic respects caller-supplied max_tokens when already large" {
    try std.testing.expectEqual(@as(i32, 64000), mapEffortToAnthropic(.medium, 64000).max_tokens);
}

fn stringifyForTest(allocator: std.mem.Allocator, value: anytype) ![]const u8 {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    try std.json.Stringify.value(value, .{ .emit_null_optional_fields = false }, &buf.writer);
    return buf.written();
}

test "tool-call arguments round-trip on every wire format" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const calls = [_]ToolCall{
        .{ .id = "c1", .name = "search", .arguments = "{\"q\":\"cats\"}" },
        .{ .id = "c2", .name = "list" },
    };
    const msgs = [_]Message{
        .{ .role = .user, .content = "go" },
        .{ .role = .assistant, .tool_calls = &calls },
    };
    const has = struct {
        fn f(haystack: []const u8, needle: []const u8) !void {
            if (std.mem.find(u8, haystack, needle) == null) {
                std.debug.print("missing {s} in {s}\n", .{ needle, haystack });
                return error.TestUnexpectedResult;
            }
        }
    }.f;

    // Object-valued APIs get the text written in place.
    const gem = try stringifyForTest(a, (try separateSystemMessages(a, &msgs)).contents);
    try has(gem, "\"functionCall\":{\"id\":\"c1\",\"name\":\"search\",\"args\":{\"q\":\"cats\"}}");
    try has(gem, "\"name\":\"list\",\"args\":{}");
    const anth = try stringifyForTest(a, try messagesToAnthropicMessages(a, &msgs));
    try has(anth, "\"name\":\"search\",\"input\":{\"q\":\"cats\"}");
    try has(anth, "\"name\":\"list\",\"input\":{}");
    const oll = try stringifyForTest(a, try messagesToOllamaMessages(a, &msgs));
    try has(oll, "\"name\":\"search\",\"arguments\":{\"q\":\"cats\"}");
    try has(oll, "\"name\":\"list\",\"arguments\":{}");

    // String-valued APIs send it as a string.
    const chat = try stringifyForTest(a, try messagesToOpenAIMessages(a, &msgs));
    try has(chat, "\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\\\"cats\\\"}\"");
    try has(chat, "\"name\":\"list\",\"arguments\":\"{}\"");
    const resp = try stringifyForTest(a, try messagesToOpenAIResponsesInput(a, &msgs, "system", .{ .openai = undefined }, "m"));
    try has(resp, "\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\\\"cats\\\"}\"");
}

test "response tool-call arguments become JSON text" {
    var result: GenerateResult = .init(std.testing.allocator);
    defer result.deinit();
    const ra = result.arena.allocator();

    // OpenAI text is kept (trimmed); empty or invalid text becomes `{}`.
    try openAiChatResult(&result, ra, .{ .choices = &.{.{ .index = 0, .message = .{ .tool_calls = &.{
        .{ .id = "a", .function = .{ .name = "t", .arguments = " {\"x\": 1}\n" } },
        .{ .id = "b", .function = .{ .name = "t", .arguments = "" } },
        .{ .id = "c", .function = .{ .name = "t", .arguments = "{\"x\": " } },
        .{ .id = "d", .function = .{ .name = "t" } },
    } } }} });
    const oai = result.tool_calls.?;
    try std.testing.expectEqualStrings("{\"x\": 1}", oai[0].arguments);
    for (oai[1..]) |tc| try std.testing.expectEqualStrings("{}", tc.arguments);

    // A parsed object is encoded once.
    const input = try ToolCall.parseArguments(ra, "{\"url\":\"https://x.test/\"}");
    try anthropicResult(&result, ra, .{ .content = &.{
        .{ .type = "tool_use", .id = "e", .name = "goto", .input = input },
        .{ .type = "tool_use", .id = "f", .name = "back" },
    } });
    const ant = result.tool_calls.?;
    try std.testing.expectEqualStrings("{\"url\":\"https://x.test/\"}", ant[0].arguments);
    try std.testing.expectEqualStrings("{}", ant[1].arguments);
}

test "tool result parts land where each API accepts an image" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // The model calls `screenshot`; the tool answers with text plus one image.
    const msgs = [_]Message{
        .{ .role = .user, .content = "look" },
        .{ .role = .assistant, .tool_calls = &.{.{ .id = "c1", .name = "screenshot" }} },
        .{ .role = .tool, .tool_results = &.{.{
            .id = "c1",
            .name = "screenshot",
            .content = "PNG, 1920px wide",
            .parts = &.{ .{ .text = "PNG, 1920px wide" }, .{ .image = .{ .data = "iVBORw0KGgo=", .mime_type = "image/png" } } },
        }} },
    };

    // Anthropic: inside the tool_result.
    const anth = try stringifyForTest(a, try messagesToAnthropicMessages(a, &msgs));
    try std.testing.expect(std.mem.find(u8, anth, "\"type\":\"tool_result\",\"tool_use_id\":\"c1\",\"content\":[{\"type\":\"text\",\"text\":\"PNG, 1920px wide\"},{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"iVBORw0KGgo=\"}}]") != null);

    // Gemini: an inlineData part after the functionResponse.
    const gem = try stringifyForTest(a, (try separateSystemMessages(a, &msgs)).contents);
    const fr = std.mem.find(u8, gem, "\"functionResponse\"") orelse return error.TestUnexpectedResult;
    const inline_data = std.mem.find(u8, gem, "\"inlineData\":{\"data\":\"iVBORw0KGgo=\",\"mimeType\":\"image/png\"}") orelse return error.TestUnexpectedResult;
    try std.testing.expect(inline_data > fr);

    // OpenAI chat, Responses, Ollama: a captioned user turn after the tool message.
    const chat = try messagesToOpenAIMessages(a, &msgs);
    try std.testing.expectEqual(4, chat.len);
    try std.testing.expectEqualStrings("PNG, 1920px wide", chat[2].content.?);
    const chat_json = try stringifyForTest(a, chat[3]);
    try std.testing.expect(std.mem.find(u8, chat_json, "\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Image returned by screenshot\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,iVBORw0KGgo=\"}}]") != null);

    const responses = try messagesToOpenAIResponsesInput(a, &msgs, "developer", .{ .codex = undefined }, "gpt-5.5");
    try std.testing.expectEqual(4, responses.len);
    const responses_json = try stringifyForTest(a, responses[3]);
    try std.testing.expect(std.mem.find(u8, responses_json, "\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"Image returned by screenshot\"},{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,iVBORw0KGgo=\"}]") != null);

    const ollama = try messagesToOllamaMessages(a, &msgs);
    try std.testing.expectEqual(4, ollama.len);
    const ollama_json = try stringifyForTest(a, ollama[3]);
    try std.testing.expect(std.mem.find(u8, ollama_json, "\"role\":\"user\",\"content\":\"Image returned by screenshot\",\"images\":[\"iVBORw0KGgo=\"]") != null);
}

test "user image parts reach every backend" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const parts = &[_]ContentPart{ .{ .text = "what is this?" }, .{ .image = .{ .data = "AAAA", .mime_type = "image/jpeg" } } };
    const msgs = [_]Message{.{ .role = .user, .parts = parts }};

    const anth = try stringifyForTest(a, try messagesToAnthropicMessages(a, &msgs));
    try std.testing.expect(std.mem.find(u8, anth, "{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/jpeg\",\"data\":\"AAAA\"}}") != null);

    const chat = try stringifyForTest(a, try messagesToOpenAIMessages(a, &msgs));
    try std.testing.expect(std.mem.find(u8, chat, "\"content\":[{\"type\":\"text\",\"text\":\"what is this?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,AAAA\"}}]") != null);
    try std.testing.expect(std.mem.find(u8, chat, "content_parts") == null);

    const responses = try stringifyForTest(a, try messagesToOpenAIResponsesInput(a, &msgs, "developer", .{ .codex = undefined }, "gpt-5.5"));
    try std.testing.expect(std.mem.find(u8, responses, "{\"type\":\"input_image\",\"image_url\":\"data:image/jpeg;base64,AAAA\"}") != null);

    const ollama = try stringifyForTest(a, try messagesToOllamaMessages(a, &msgs));
    try std.testing.expect(std.mem.find(u8, ollama, "\"images\":[\"AAAA\"]") != null);
}
