//! Ollama client. Chat goes to the native `/api/chat`: unlike the
//! OpenAI-compatible `/v1` shim it honors `options.num_ctx`, so we can size the
//! context window to the request; the shim's small default (4096) silently
//! truncates prompts that exceed it, dropping tool calls. Listing and
//! embeddings use `/v1` through the embedded `openai` client.

const std = @import("std");
const http = @import("../http.zig");
const json = @import("../json.zig");
const retry = @import("../retry.zig");
const openai_types = @import("../openai/types.zig");
const OpenAIClient = @import("../openai/Client.zig");

const Client = @This();

/// The `/v1` client: model listing, embeddings, and the transport and headers
/// the native calls share.
openai: OpenAIClient,
/// The last model's context window from `/api/show`, so the lookup runs once
/// per model. `model` is owned by the transport's allocator; `len` is null
/// when the lookup missed (cached so it isn't retried).
ctx_cache: ?struct { model: []const u8, len: ?i32 } = null,

pub const InitOptions = struct {
    /// The OpenAI-compatible base URL; the native `/api/*` endpoints hang off
    /// its origin.
    base_url: []const u8 = "http://localhost:11434/v1",
    /// Sent on every request; borrowed, see `openai.Client.InitOptions`.
    extra_headers: []const std.http.Header = &.{},
    retry_policy: retry.RetryPolicy = .{},
    /// See `openai.Client.InitOptions.request_timeout_ms`.
    request_timeout_ms: ?u32 = null,
    /// Borrowed `std.http.Client` to share a connection pool with other
    /// clients; see `http.Transport.Options.http_client`. Null owns one.
    http_client: ?*std.http.Client = null,
};

/// Local Ollama ignores `api_key`; it is sent for an authenticating proxy.
pub fn init(io: std.Io, allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    var openai: OpenAIClient = .init(io, allocator, api_key, .{
        .base_url = options.base_url,
        .extra_headers = options.extra_headers,
        .retry_policy = options.retry_policy,
        .request_timeout_ms = options.request_timeout_ms,
        .http_client = options.http_client,
    });
    openai.transport.log_tag = "Ollama";
    return .{ .openai = openai };
}

pub fn deinit(self: *Client) void {
    if (self.ctx_cache) |c| self.openai.transport.allocator.free(c.model);
    self.openai.deinit();
}

pub const ApiError = OpenAIClient.ApiError;
pub const StreamError = OpenAIClient.StreamError;

/// Ollama delivers arguments as a JSON object both ways, mapping to `Value`.
pub const FunctionCall = struct {
    name: ?[]const u8 = null,
    arguments: ?std.json.Value = null,
};

/// `id`/`type` are echoed when present; Ollama matches tool results by name.
pub const ToolCall = struct {
    id: ?[]const u8 = null,
    type: ?[]const u8 = null,
    function: ?FunctionCall = null,
};

/// Serves both request and response. Unknown fields are ignored on parse.
pub const Message = struct {
    role: ?[]const u8 = null,
    content: ?[]const u8 = null,
    /// Reasoning trace from a thinking model (response only).
    thinking: ?[]const u8 = null,
    /// Names the tool a `role: "tool"` message answers (request only).
    tool_name: ?[]const u8 = null,
    tool_calls: ?[]const ToolCall = null,
    /// Base64 images attached to the message (request only).
    images: ?[]const []const u8 = null,
};

/// Request `options`. `num_ctx` is the field the `/v1` shim lacks.
pub const Options = struct {
    /// Context window; filled by `chat` from the request size when null.
    num_ctx: ?i32 = null,
    /// Maps to the unified `max_tokens`.
    num_predict: ?i32 = null,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    seed: ?i32 = null,
    stop: ?[]const []const u8 = null,
    frequency_penalty: ?f32 = null,
    presence_penalty: ?f32 = null,
};

/// `POST /api/chat` body. `openai_types.Tool` shares Ollama's tool schema shape.
pub const ChatRequest = struct {
    model: []const u8,
    messages: []const Message,
    tools: ?[]const openai_types.Tool = null,
    /// Toggle a thinking model's reasoning; top-level, not inside `options`.
    think: ?bool = null,
    /// `"json"` for JSON mode (or a JSON schema); top-level, not in `options`.
    format: ?[]const u8 = null,
    stream: bool = false,
    options: Options,
};

/// Response body for a non-streaming `POST /api/chat`.
pub const ChatResponse = struct {
    model: ?[]const u8 = null,
    message: ?Message = null,
    done: ?bool = null,
    /// "stop", "length", "load", … — Ollama's analogue of finish_reason.
    done_reason: ?[]const u8 = null,
    prompt_eval_count: ?i32 = null,
    eval_count: ?i32 = null,
};

/// `model_info` carries `<arch>.context_length` (e.g. `qwen35.context_length`);
/// the rest of `/api/show` (tensors, modelfile, …) is ignored on parse.
pub const ShowResponse = struct {
    model_info: ?std.json.Value = null,
};

// Floor gives small follow-up turns headroom to reuse the loaded model rather
// than forcing a reload; the real cap is the model's own context length.
const min_num_ctx: usize = 8192;
const ctx_round: usize = 2048;

/// Context window fitting the request plus output budget. Tokens are estimated
/// at ~3 bytes each (an overestimate, so we reserve rather than truncate),
/// padded, rounded up, floored at `min_num_ctx`, and capped at `model_max`.
pub fn computeNumCtx(prompt_bytes: usize, output_budget: i32, model_max: ?i32) i32 {
    const prompt_tokens = prompt_bytes / 3;
    const out: usize = @intCast(@max(output_budget, 0));
    const needed = prompt_tokens + out + 1024;
    const rounded = (needed + ctx_round - 1) / ctx_round * ctx_round;
    var ctx = @max(rounded, min_num_ctx);
    if (model_max) |m| {
        if (m > 0) ctx = @min(ctx, @as(usize, @intCast(m)));
    }
    return @intCast(ctx);
}

/// Model's context window via `/api/show`, cached on the client (once per
/// model), or null if the lookup fails or the field is absent.
fn modelContextLength(self: *Client, model: []const u8) ?i32 {
    if (self.ctx_cache) |cached| {
        if (std.mem.eql(u8, cached.model, model)) return cached.len;
    }

    const len = self.fetchModelContextLength(model) catch null;

    // Cache even a null miss, keyed by model name.
    const allocator = self.openai.transport.allocator;
    if (self.ctx_cache) |old| allocator.free(old.model);
    self.ctx_cache = if (allocator.dupe(u8, model)) |owned|
        .{ .model = owned, .len = len }
    else |_|
        null;
    return len;
}

fn fetchModelContextLength(self: *Client, model: []const u8) !?i32 {
    const url = try showUrl(self.openai.transport.allocator, self.openai.base_url);
    defer self.openai.transport.allocator.free(url);

    const auth = [_]std.http.Header{self.authHeader()};
    const headers = try self.openai.withExtraHeaders(&auth);
    defer self.openai.freeHeaders(&auth, headers);
    var response = try self.openai.transport.postJson(url, headers, .{ .model = model }, ShowResponse);
    defer response.deinit();

    const info = response.value.model_info orelse return null;
    if (info != .object) return null;
    // Match `<arch>.context_length` by suffix, architecture-agnostic.
    var it = info.object.iterator();
    while (it.next()) |entry| {
        if (!std.mem.endsWith(u8, entry.key_ptr.*, ".context_length")) continue;
        if (entry.value_ptr.* == .integer) {
            const n = entry.value_ptr.*.integer;
            if (n > 0 and n <= std.math.maxInt(i32)) return @intCast(n);
        }
        // A non-integer (or out-of-range) match isn't the field we want; keep
        // scanning in case another key carries a usable value.
    }
    return null;
}

/// Local Ollama ignores auth; harmless, and supports an authenticating proxy.
fn authHeader(self: *const Client) std.http.Header {
    return .{ .name = "Authorization", .value = self.openai.api_key };
}

fn showUrl(allocator: std.mem.Allocator, base_url: []const u8) ![]u8 {
    const origin = stripV1(base_url);
    return allocator.print("{s}/api/show", .{origin});
}

/// Strip the OpenAI-compatible `/v1` suffix (or a trailing slash) to recover
/// the server origin the native `/api/*` endpoints hang off.
fn stripV1(base_url: []const u8) []const u8 {
    const trimmed = std.mem.trimEnd(u8, base_url, "/");
    if (std.mem.endsWith(u8, trimmed, "/v1"))
        return trimmed[0 .. trimmed.len - "/v1".len];
    return trimmed;
}

/// Derive the native `/api/chat` URL from the OpenAI-compatible base URL the
/// client was built with (e.g. `http://localhost:11434/v1` → `…/api/chat`).
fn chatUrl(allocator: std.mem.Allocator, base_url: []const u8) ![]u8 {
    return allocator.print("{s}/api/chat", .{stripV1(base_url)});
}

/// Build the `/api/chat` body, sizing a null `options.num_ctx` to the serialized
/// request (capped at the model's limit). Shared by the buffered and streaming
/// paths so both get the same context-window fitting.
fn buildChatRequest(
    self: *Client,
    model: []const u8,
    messages: []const Message,
    tools: ?[]const openai_types.Tool,
    think: ?bool,
    format: ?[]const u8,
    options: Options,
    stream: bool,
) std.mem.Allocator.Error!ChatRequest {
    var req = ChatRequest{
        .model = model,
        .messages = messages,
        .tools = tools,
        .think = think,
        .format = format,
        .stream = stream,
        .options = options,
    };

    if (req.options.num_ctx == null) {
        // Only the encoded size matters: count it without keeping the bytes.
        var buf: [256]u8 = undefined;
        var measure: std.Io.Writer.Discarding = .init(&buf);
        std.json.Stringify.value(req, .{ .emit_null_optional_fields = false }, &measure.writer) catch unreachable;
        const model_max = self.modelContextLength(model);
        req.options.num_ctx = computeNumCtx(@intCast(measure.fullCount()), req.options.num_predict orelse 4096, model_max);
    }
    return req;
}

/// Non-streaming chat via the native endpoint. A null `options.num_ctx` is
/// sized to the request.
pub fn chat(
    self: *Client,
    model: []const u8,
    messages: []const Message,
    tools: ?[]const openai_types.Tool,
    think: ?bool,
    format: ?[]const u8,
    options: Options,
) ApiError!http.Response(ChatResponse) {
    if (self.openai.api_key.len == 0) return error.MissingApiKey;

    const url = try chatUrl(self.openai.transport.allocator, self.openai.base_url);
    defer self.openai.transport.allocator.free(url);

    const req = try self.buildChatRequest(model, messages, tools, think, format, options, false);
    const auth = [_]std.http.Header{self.authHeader()};
    const headers = try self.openai.withExtraHeaders(&auth);
    defer self.openai.freeHeaders(&auth, headers);
    return self.openai.transport.postJson(url, headers, req, ChatResponse);
}

/// Streaming counterpart of `chat`: sets `stream: true` and invokes `callback`
/// with each NDJSON chunk as it arrives (each carries an incremental
/// `message.content` delta). No retry — a mid-stream failure can't be replayed
/// without re-emitting already-delivered deltas.
pub fn chatStream(
    self: *Client,
    model: []const u8,
    messages: []const Message,
    tools: ?[]const openai_types.Tool,
    think: ?bool,
    format: ?[]const u8,
    options: Options,
    context: anytype,
    callback: *const fn (@TypeOf(context), ChatResponse) void,
) StreamError!void {
    if (self.openai.api_key.len == 0) return error.MissingApiKey;

    const url = try chatUrl(self.openai.transport.allocator, self.openai.base_url);
    defer self.openai.transport.allocator.free(url);

    const req = try self.buildChatRequest(model, messages, tools, think, format, options, true);
    const auth = [_]std.http.Header{self.authHeader()};
    const headers = try self.openai.withExtraHeaders(&auth);
    defer self.openai.freeHeaders(&auth, headers);
    return self.openai.transport.streamNdjson(url, headers, req, ChatResponse, context, callback);
}

/// Reassembles a streamed `/api/chat` response — forwarding each text delta to
/// `on_text` as it arrives — into a buffered `ChatResponse` shaped like the
/// non-streaming path's, so callers can reuse the same result mapping. All kept
/// data is duped into `arena`, which must outlive the final `response()`.
pub const StreamAccumulator = struct {
    arena: std.mem.Allocator,
    on_text_ctx: *anyopaque,
    on_text_fn: *const fn (*anyopaque, []const u8) void,
    content: std.ArrayList(u8) = .empty,
    tool_calls: std.ArrayList(ToolCall) = .empty,
    done_reason: ?[]const u8 = null,
    prompt_eval_count: ?i32 = null,
    eval_count: ?i32 = null,
    err: ?error{OutOfMemory} = null,

    pub fn init(
        arena: std.mem.Allocator,
        on_text_ctx: *anyopaque,
        on_text_fn: *const fn (*anyopaque, []const u8) void,
    ) StreamAccumulator {
        return .{ .arena = arena, .on_text_ctx = on_text_ctx, .on_text_fn = on_text_fn };
    }

    pub fn onChunk(self: *StreamAccumulator, chunk: ChatResponse) void {
        self.handle(chunk) catch |e| {
            self.err = e;
        };
    }

    fn handle(self: *StreamAccumulator, chunk: ChatResponse) error{OutOfMemory}!void {
        if (chunk.message) |m| {
            if (m.content) |c| if (c.len > 0) {
                self.on_text_fn(self.on_text_ctx, c);
                try self.content.appendSlice(self.arena, c);
            };
            // Ollama emits each tool call whole (arguments as a complete object),
            // not fragmented across chunks like OpenAI — just collect them.
            if (m.tool_calls) |calls| for (calls) |call| {
                if (call.function) |f| try self.tool_calls.append(self.arena, .{
                    .id = if (call.id) |id| try self.arena.dupe(u8, id) else null,
                    .function = .{
                        .name = if (f.name) |n| try self.arena.dupe(u8, n) else null,
                        .arguments = if (f.arguments) |v| try json.dupeValue(self.arena, v) else null,
                    },
                });
            };
        }
        if (chunk.done orelse false) {
            if (chunk.done_reason) |dr| self.done_reason = try self.arena.dupe(u8, dr);
            self.prompt_eval_count = chunk.prompt_eval_count;
            self.eval_count = chunk.eval_count;
        }
    }

    /// Assemble the accumulated deltas into a buffered-shaped `ChatResponse`.
    /// Call once, after the stream completes and `err` is clear.
    pub fn response(self: *StreamAccumulator) ChatResponse {
        return .{
            .message = .{
                .role = "assistant",
                .content = self.content.items,
                .tool_calls = if (self.tool_calls.items.len > 0) self.tool_calls.items else null,
            },
            .done = true,
            .done_reason = self.done_reason,
            .prompt_eval_count = self.prompt_eval_count,
            .eval_count = self.eval_count,
        };
    }
};

test "chatUrl derives native endpoint from the v1 base" {
    const a = std.testing.allocator;
    {
        const u = try chatUrl(a, "http://localhost:11434/v1");
        defer a.free(u);
        try std.testing.expectEqualStrings("http://localhost:11434/api/chat", u);
    }
    {
        const u = try chatUrl(a, "http://example.com:1234/");
        defer a.free(u);
        try std.testing.expectEqualStrings("http://example.com:1234/api/chat", u);
    }
    {
        // A trailing slash after `/v1` must still strip the suffix.
        const u = try chatUrl(a, "http://localhost:11434/v1/");
        defer a.free(u);
        try std.testing.expectEqualStrings("http://localhost:11434/api/chat", u);
    }
}

test "the context-window cache answers a repeat lookup and is freed on deinit" {
    var client = Client.init(std.testing.io, std.testing.allocator, "ollama", .{});
    defer client.deinit();
    client.ctx_cache = .{ .model = try std.testing.allocator.dupe(u8, "qwen3.5"), .len = 32768 };
    try std.testing.expectEqual(@as(?i32, 32768), client.modelContextLength("qwen3.5"));
}

test "computeNumCtx floors, rounds, and caps at the model limit" {
    // Small prompt floors at min_num_ctx.
    try std.testing.expectEqual(@as(i32, @intCast(min_num_ctx)), computeNumCtx(100, 256, null));
    // Mid-range rounds up to a multiple of ctx_round and clears the prompt.
    const mid = computeNumCtx(30_000, 4096, null); // ~10k prompt + 4k + 1k slack
    try std.testing.expect(mid >= 14336);
    try std.testing.expect(@rem(@as(usize, @intCast(mid)), ctx_round) == 0);
    // A generous model limit doesn't shrink a need-based window.
    try std.testing.expectEqual(mid, computeNumCtx(30_000, 4096, 262144));
    // A model smaller than the need caps the request (no silent over-ask).
    try std.testing.expectEqual(@as(i32, 8192), computeNumCtx(1_000_000, 4096, 8192));
}

test "ChatResponse parses tool calls with object arguments" {
    const body =
        \\{"model":"qwen3.5","message":{"role":"assistant","content":"","thinking":"hmm","tool_calls":[{"id":"call_1","function":{"name":"goto","arguments":{"url":"https://news.ycombinator.com/"}}}]},"done":true,"done_reason":"stop","prompt_eval_count":301,"eval_count":71}
    ;
    const parsed = try std.json.parseFromSlice(ChatResponse, std.testing.allocator, body, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const msg = parsed.value.message.?;
    const call = msg.tool_calls.?[0];
    try std.testing.expectEqualStrings("goto", call.function.?.name.?);
    try std.testing.expectEqualStrings("https://news.ycombinator.com/", call.function.?.arguments.?.object.get("url").?.string);
    try std.testing.expectEqual(@as(i32, 71), parsed.value.eval_count.?);
}

test "StreamAccumulator reassembles content, forwards deltas, captures usage" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const Sink = struct {
        buf: std.ArrayList(u8) = .empty,
        a: std.mem.Allocator,
        fn onText(ctx: *anyopaque, delta: []const u8) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.buf.appendSlice(self.a, delta) catch {};
        }
    };
    var sink = Sink{ .a = arena };

    var acc = StreamAccumulator.init(arena, &sink, Sink.onText);
    acc.onChunk(.{ .message = .{ .role = "assistant", .content = "Hel" }, .done = false });
    acc.onChunk(.{ .message = .{ .role = "assistant", .content = "lo" }, .done = false });
    acc.onChunk(.{ .message = .{ .role = "assistant", .content = "" }, .done = true, .done_reason = "stop", .prompt_eval_count = 12, .eval_count = 3 });

    try std.testing.expect(acc.err == null);
    try std.testing.expectEqualStrings("Hello", sink.buf.items);
    const resp = acc.response();
    try std.testing.expectEqualStrings("Hello", resp.message.?.content.?);
    try std.testing.expectEqualStrings("stop", resp.done_reason.?);
    try std.testing.expectEqual(@as(i32, 12), resp.prompt_eval_count.?);
    try std.testing.expectEqual(@as(i32, 3), resp.eval_count.?);
}

test "StreamAccumulator dupes a tool call so it survives chunk free" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const noop = struct {
        fn onText(_: *anyopaque, _: []const u8) void {}
    };
    var dummy: u8 = 0;
    var acc = StreamAccumulator.init(arena, &dummy, noop.onText);

    // Feed the tool call from an owned parse, then free that parse — the
    // accumulator must have duped name + arguments into its own arena.
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"url\":\"https://x.test/\"}", .{});
    acc.onChunk(.{ .message = .{ .role = "assistant", .content = "", .tool_calls = &.{.{
        .id = "call_1",
        .function = .{ .name = "goto", .arguments = parsed.value },
    }} }, .done = false });
    parsed.deinit();
    acc.onChunk(.{ .done = true, .done_reason = "stop" });

    try std.testing.expect(acc.err == null);
    const calls = acc.response().message.?.tool_calls.?;
    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("goto", calls[0].function.?.name.?);
    try std.testing.expectEqualStrings("https://x.test/", calls[0].function.?.arguments.?.object.get("url").?.string);
}
