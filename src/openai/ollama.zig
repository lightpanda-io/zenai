//! Native Ollama `/api/chat`. Unlike the OpenAI-compatible `/v1` shim it
//! honors `options.num_ctx`, so we can size the context window to the request;
//! the shim's small default (4096) silently truncates prompts that exceed it,
//! dropping tool calls. Only chat routes here — listing/embeddings use `/v1`.

const std = @import("std");
const http = @import("../http.zig");
const openai_types = @import("types.zig");
const Client = @import("Client.zig");

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
fn modelContextLength(client: *Client, model: []const u8) ?i32 {
    if (client.ollama_ctx) |cached| {
        if (std.mem.eql(u8, cached.model, model)) return cached.len;
    }

    const len = fetchModelContextLength(client, model) catch null;

    // Cache even a null miss, keyed by model name.
    if (client.ollama_ctx) |old| client.allocator.free(old.model);
    client.ollama_ctx = if (client.allocator.dupe(u8, model)) |owned|
        .{ .model = owned, .len = len }
    else |_|
        null;
    return len;
}

fn fetchModelContextLength(client: *Client, model: []const u8) !?i32 {
    const url = try showUrl(client.allocator, client.base_url);
    defer client.allocator.free(url);

    var payload_buf: std.Io.Writer.Allocating = .init(client.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(.{ .model = model }, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    const auth = [_]std.http.Header{.{ .name = "Authorization", .value = client.api_key }};
    var response = try http.fetchJsonWithRetry(client.allocator, &client.http_client, client.retry_policy, .{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload_buf.written(),
        .extra_headers = &auth,
        .headers = .{ .content_type = .{ .override = "application/json" } },
    }, ShowResponse, client);
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

fn showUrl(allocator: std.mem.Allocator, base_url: []const u8) ![]u8 {
    const origin = stripV1(base_url);
    return std.fmt.allocPrint(allocator, "{s}/api/show", .{origin});
}

/// Strip the OpenAI-compatible `/v1` suffix (or a trailing slash) to recover
/// the server origin the native `/api/*` endpoints hang off.
fn stripV1(base_url: []const u8) []const u8 {
    const trimmed = std.mem.trimRight(u8, base_url, "/");
    if (std.mem.endsWith(u8, trimmed, "/v1"))
        return trimmed[0 .. trimmed.len - "/v1".len];
    return trimmed;
}

/// Derive the native `/api/chat` URL from the OpenAI-compatible base URL the
/// client was built with (e.g. `http://localhost:11434/v1` → `…/api/chat`).
fn chatUrl(allocator: std.mem.Allocator, base_url: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/api/chat", .{stripV1(base_url)});
}

/// Non-streaming chat via the native endpoint, reusing `client`'s HTTP client,
/// retry policy, and interrupt. A null `options.num_ctx` is sized to the request.
pub fn chat(
    client: *Client,
    model: []const u8,
    messages: []const Message,
    tools: ?[]const openai_types.Tool,
    think: ?bool,
    format: ?[]const u8,
    options: Options,
) Client.ApiError!http.Response(ChatResponse) {
    if (client.api_key.len == 0) return error.MissingApiKey;

    const url = try chatUrl(client.allocator, client.base_url);
    defer client.allocator.free(url);

    var req = ChatRequest{
        .model = model,
        .messages = messages,
        .tools = tools,
        .think = think,
        .format = format,
        .stream = false,
        .options = options,
    };

    // Size num_ctx to the serialized request, capped at the model's limit.
    if (req.options.num_ctx == null) {
        var measure: std.Io.Writer.Allocating = .init(client.allocator);
        defer measure.deinit();
        std.json.Stringify.value(req, .{ .emit_null_optional_fields = false }, &measure.writer) catch
            return error.OutOfMemory;
        const model_max = modelContextLength(client, model);
        req.options.num_ctx = computeNumCtx(measure.written().len, req.options.num_predict orelse 4096, model_max);
    }

    var payload_buf: std.Io.Writer.Allocating = .init(client.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(req, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    // Local Ollama ignores auth; harmless, and supports an authenticating proxy.
    const auth = [_]std.http.Header{.{ .name = "Authorization", .value = client.api_key }};
    return http.fetchJsonWithRetry(client.allocator, &client.http_client, client.retry_policy, .{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload_buf.written(),
        .extra_headers = &auth,
        .headers = .{ .content_type = .{ .override = "application/json" } },
    }, ChatResponse, client);
}

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
    const json =
        \\{"model":"qwen3.5","message":{"role":"assistant","content":"","thinking":"hmm","tool_calls":[{"id":"call_1","function":{"name":"goto","arguments":{"url":"https://news.ycombinator.com/"}}}]},"done":true,"done_reason":"stop","prompt_eval_count":301,"eval_count":71}
    ;
    const parsed = try std.json.parseFromSlice(ChatResponse, std.testing.allocator, json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const msg = parsed.value.message.?;
    const call = msg.tool_calls.?[0];
    try std.testing.expectEqualStrings("goto", call.function.?.name.?);
    try std.testing.expectEqualStrings("https://news.ycombinator.com/", call.function.?.arguments.?.object.get("url").?.string);
    try std.testing.expectEqual(@as(i32, 71), parsed.value.eval_count.?);
}
