//! A multi-turn conversation on top of `provider.Client`, for any backend.
//!
//! The chat owns its history: every message (user input, model replies, tool
//! calls and results) is copied into the chat's arena, so callers may pass
//! temporary buffers. A failed turn rolls the history back to where it was,
//! so the conversation stays valid; its copies stay in the arena until
//! `deinit` or `reset`.

const std = @import("std");
const provider = @import("provider.zig");

const Client = provider.Client;
const Message = provider.Message;
const ContentPart = provider.ContentPart;

const Chat = @This();

client: Client,
model: []const u8,
options: Options,
allocator: std.mem.Allocator,
arena: std.heap.ArenaAllocator,
history: std.ArrayListUnmanaged(Message) = .empty,

pub const Options = struct {
    /// Sent as the first message of every request.
    system: ?[]const u8 = null,
    max_tokens: ?i32 = 4096,
    temperature: ?f32 = null,
    effort: ?provider.Effort = null,
};

/// What one turn produced. Strings live in the chat's arena: valid until
/// `reset` or `deinit`.
pub const Reply = struct {
    text: ?[]const u8,
    finish_reason: provider.FinishReason,
    usage: provider.Usage,
};

/// The chat borrows `client`; it must outlive the chat.
pub fn init(allocator: std.mem.Allocator, client: Client, model: []const u8, options: Options) Chat {
    return .{
        .client = client,
        .model = model,
        .options = options,
        .allocator = allocator,
        .arena = .init(allocator),
    };
}

pub fn deinit(self: *Chat) void {
    self.history.deinit(self.allocator);
    self.arena.deinit();
}

/// Forget the conversation and free everything it held.
pub fn reset(self: *Chat) void {
    self.history.clearRetainingCapacity();
    _ = self.arena.reset(.retain_capacity);
}

/// The conversation so far, without the system prompt.
pub fn messages(self: *const Chat) []const Message {
    return self.history.items;
}

/// Send a text prompt and wait for the reply.
pub fn send(self: *Chat, prompt: []const u8) Client.Error!Reply {
    return self.sendMessage(.{ .role = .user, .content = prompt }, null);
}

/// Send a text prompt, forwarding the reply's text to `on_text` as it streams.
pub fn sendStream(self: *Chat, prompt: []const u8, on_text: Client.TextDeltaHook) Client.Error!Reply {
    return self.sendMessage(.{ .role = .user, .content = prompt }, on_text);
}

/// Send text and images (`ContentPart`s) as one user message.
pub fn sendParts(self: *Chat, parts: []const ContentPart, on_text: ?Client.TextDeltaHook) Client.Error!Reply {
    return self.sendMessage(.{ .role = .user, .parts = parts }, on_text);
}

/// Send any user message; streams when `on_text` is set. The model gets no
/// tools here, so the reply is text. Use `sendWithTools` for tool use.
pub fn sendMessage(self: *Chat, message: Message, on_text: ?Client.TextDeltaHook) Client.Error!Reply {
    const mark = self.history.items.len;
    errdefer self.history.shrinkRetainingCapacity(mark);
    const a = self.arena.allocator();
    try self.history.append(self.allocator, try provider.dupeMessage(a, message));

    const request = try self.withSystem();
    defer self.allocator.free(request);
    const config: provider.GenerationConfig = .{
        .max_tokens = self.options.max_tokens,
        .temperature = self.options.temperature,
        .effort = self.options.effort,
    };
    var result = if (on_text) |hook|
        try self.client.generateContentStream(self.model, request, config, hook)
    else
        try self.client.generateContent(self.model, request, config);
    defer result.deinit();

    const text = if (result.text) |t| try a.dupe(u8, t) else null;
    try self.history.append(self.allocator, .{
        .role = .assistant,
        .content = text,
        .reasoning = if (result.reasoning) |r| try provider.dupeReasoning(a, r) else null,
    });
    return .{ .text = text, .finish_reason = result.finish_reason, .usage = result.usage };
}

/// Send a text prompt and run the tool loop (`Client.runTools`) until the
/// model answers. Tool calls and results join the history. `config.max_tokens`,
/// `temperature` and `effort` default to the chat's options when left null.
/// The caller must `deinit` the result.
pub fn sendWithTools(
    self: *Chat,
    prompt: []const u8,
    handler: Client.ToolHandler,
    config_in: Client.RunToolsConfig,
) Client.Error!Client.RunToolsResult {
    const mark = self.history.items.len;
    errdefer self.history.shrinkRetainingCapacity(mark);
    const a = self.arena.allocator();
    try self.history.append(self.allocator, .{ .role = .user, .content = try a.dupe(u8, prompt) });

    var config = config_in;
    config.max_tokens = config.max_tokens orelse self.options.max_tokens;
    config.temperature = config.temperature orelse self.options.temperature;
    config.effort = config.effort orelse self.options.effort;

    // runTools appends to the list it is given; run it on the full request
    // (system prompt first) and copy the new turns back into the history.
    var list: std.ArrayListUnmanaged(Message) = .empty;
    defer list.deinit(self.allocator);
    if (self.options.system) |s| try list.append(self.allocator, .{ .role = .system, .content = s });
    const head = list.items.len;
    try list.appendSlice(self.allocator, self.history.items);
    const before = list.items.len;

    var result = try self.client.runTools(self.model, &list, self.allocator, a, handler, config);
    errdefer result.deinit();
    try self.history.appendSlice(self.allocator, list.items[before..]);
    std.debug.assert(self.history.items.len == list.items.len - head);
    return result;
}

/// The system prompt followed by the history; caller frees the slice.
fn withSystem(self: *Chat) error{OutOfMemory}![]Message {
    const system = self.options.system orelse return self.allocator.dupe(Message, self.history.items);
    const out = try self.allocator.alloc(Message, self.history.items.len + 1);
    out[0] = .{ .role = .system, .content = system };
    @memcpy(out[1..], self.history.items);
    return out;
}

// --- Tests ---

/// Answers each request on one keep-alive connection with the next canned
/// Chat Completions body, recording the request bodies.
const CannedServer = struct {
    server: std.Io.net.Server,
    replies: []const []const u8,
    requests: [4][]u8 = undefined,
    served: usize = 0,
    /// The client borrows its base URL, so it lives here.
    url_buf: [64]u8 = undefined,

    fn run(self: *CannedServer) void {
        const io = std.testing.io;
        const stream = self.server.accept(io) catch return;
        defer stream.close(io);
        var read_buf: [8192]u8 = undefined;
        var write_buf: [1024]u8 = undefined;
        var reader = stream.reader(io, &read_buf);
        var writer = stream.writer(io, &write_buf);
        var http_server: std.http.Server = .init(&reader.interface, &writer.interface);
        while (self.served < self.replies.len) {
            var req = http_server.receiveHead() catch return;
            var body_buf: [8192]u8 = undefined;
            const body = req.readerExpectNone(&body_buf).allocRemaining(std.testing.allocator, .unlimited) catch return;
            self.requests[self.served] = body;
            const out = self.replies[self.served];
            self.served += 1;
            req.respond(out, .{
                .keep_alive = true,
                .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
            }) catch return;
        }
    }

    /// An OpenAI-compatible client pointed at this server.
    fn client(self: *CannedServer) !Client {
        const url = try std.fmt.bufPrintSentinel(&self.url_buf, "http://127.0.0.1:{d}", .{self.server.socket.address.getPort()}, 0);
        return Client.init(std.testing.io, std.testing.allocator, .openai_compatible, "key", .{
            .base_url = url,
            .retry_policy = .disabled,
            .request_timeout_ms = 2000,
        });
    }

    fn deinit(self: *CannedServer) void {
        for (self.requests[0..self.served]) |r| std.testing.allocator.free(r);
        self.server.deinit(std.testing.io);
    }
};

fn completion(comptime text: []const u8) []const u8 {
    return "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" ++ text ++
        "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}";
}

test "send keeps the history and sends it back with the system prompt" {
    const loopback: std.Io.net.IpAddress = try .parse("127.0.0.1", 0);
    var server: CannedServer = .{
        .server = try loopback.listen(std.testing.io, .{}),
        .replies = &.{ completion("Hi Ada."), completion("Your name is Ada.") },
    };
    defer server.deinit();
    const thread = try std.Thread.spawn(.{}, CannedServer.run, .{&server});
    defer thread.join();

    const client = try server.client();
    defer client.deinit(std.testing.allocator);

    var chat: Chat = .init(std.testing.allocator, client, "m", .{ .system = "Be brief." });
    defer chat.deinit();

    var name_buf = "I am Ada.".*;
    const first = try chat.send(&name_buf);
    @memset(&name_buf, 'x'); // the chat keeps its own copy
    try std.testing.expectEqualStrings("Hi Ada.", first.text.?);

    const second = try chat.send("What is my name?");
    try std.testing.expectEqualStrings("Your name is Ada.", second.text.?);
    try std.testing.expectEqual(.stop, second.finish_reason);

    try std.testing.expectEqual(4, chat.messages().len);
    try std.testing.expectEqualStrings("I am Ada.", chat.messages()[0].content.?);
    try std.testing.expectEqual(.assistant, chat.messages()[3].role);

    // The second request carries the system prompt and the first exchange.
    const sent = server.requests[1];
    for ([_][]const u8{ "Be brief.", "I am Ada.", "Hi Ada.", "What is my name?" }) |want| {
        try std.testing.expect(std.mem.indexOf(u8, sent, want) != null);
    }
}

test "a failed turn leaves the history as it was" {
    // A port nobody listens on: the request fails before any reply.
    const loopback: std.Io.net.IpAddress = try .parse("127.0.0.1", 0);
    var server: CannedServer = .{ .server = try loopback.listen(std.testing.io, .{}), .replies = &.{} };
    const client = try server.client();
    defer client.deinit(std.testing.allocator);
    server.deinit();

    var chat: Chat = .init(std.testing.allocator, client, "m", .{ .system = "s" });
    defer chat.deinit();

    try std.testing.expect(std.meta.isError(chat.send("hello")));
    try std.testing.expectEqual(0, chat.messages().len);
}
