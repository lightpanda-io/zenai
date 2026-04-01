const std = @import("std");
const types = @import("types.zig");
const Client = @import("Client.zig");

const Message = types.Message;
const ChatCompletionResponse = types.ChatCompletionResponse;

/// A multi-turn chat session that automatically tracks conversation history.
/// All messages are deep-copied into an arena, so temporary buffers are safe to use.
/// Call `deinit()` when done to free all resources.
const Chat = @This();

client: *Client,
model: []const u8,
config: Client.ChatCompletionConfig,
history: std.ArrayListUnmanaged(Message),
responses: std.ArrayListUnmanaged(Client.Response(ChatCompletionResponse)),
arena: std.heap.ArenaAllocator,

/// Create a new chat session with the given model and configuration.
pub fn init(
    client: *Client,
    model: []const u8,
    config: Client.ChatCompletionConfig,
) Chat {
    return .{
        .client = client,
        .model = model,
        .config = config,
        .history = .empty,
        .responses = .empty,
        .arena = std.heap.ArenaAllocator.init(client.allocator),
    };
}

/// Release all resources: conversation history, responses, and owned strings.
pub fn deinit(self: *Chat) void {
    self.history.deinit(self.client.allocator);

    for (self.responses.items) |*resp| resp.deinit();
    self.responses.deinit(self.client.allocator);

    self.arena.deinit();
}

/// Send a message with the given role and content.
pub fn send(self: *Chat, user_messages: []const Message) Client.ApiError!ChatCompletionResponse {
    for (user_messages) |msg| {
        const owned = try self.dupeMessage(msg);
        try self.history.append(self.client.allocator, owned);
    }

    var response = self.client.chatCompletion(
        self.model,
        self.history.items,
        self.config,
    ) catch |err| {
        // Roll back added messages
        self.history.shrinkRetainingCapacity(self.history.items.len - user_messages.len);
        return err;
    };

    errdefer {
        response.deinit();
        self.history.shrinkRetainingCapacity(self.history.items.len - user_messages.len);
    }

    try self.responses.append(self.client.allocator, response);

    if (validateResponse(response.value)) {
        try self.history.append(self.client.allocator, response.value.choices.?[0].message.?);
    } else {
        // Invalid response: remove user messages so they don't pollute future requests.
        self.history.shrinkRetainingCapacity(self.history.items.len - user_messages.len);
    }

    return response.value;
}

/// Send a text-only message (convenience wrapper around `send`).
pub fn sendMessage(self: *Chat, prompt: []const u8) Client.ApiError!ChatCompletionResponse {
    const messages = [_]Message{.{ .role = .user, .content = prompt }};
    return self.send(&messages);
}

/// Send messages and stream the response.
/// The callback receives each chunk. History is updated after streaming completes.
pub fn sendStream(
    self: *Chat,
    user_messages: []const Message,
    context: anytype,
    callback: *const fn (@TypeOf(context), ChatCompletionResponse) void,
) Client.StreamError!void {
    const arena_alloc = self.arena.allocator();

    for (user_messages) |msg| {
        const owned = self.dupeMessage(msg) catch return error.OutOfMemory;
        self.history.append(self.client.allocator, owned) catch return error.OutOfMemory;
    }

    var collected_content = std.ArrayListUnmanaged(u8).empty;
    var is_valid = true;

    errdefer self.history.shrinkRetainingCapacity(self.history.items.len - user_messages.len);

    const StreamCtx = struct {
        user_ctx: @TypeOf(context),
        user_cb: *const fn (@TypeOf(context), ChatCompletionResponse) void,
        content: *std.ArrayListUnmanaged(u8),
        valid: *bool,
        alloc: std.mem.Allocator,

        fn handle(s: *const @This(), response: ChatCompletionResponse) void {
            if (!validateResponse(response)) {
                s.valid.* = false;
            }
            if (response.choices) |choices| {
                if (choices.len > 0) {
                    if (choices[0].delta) |delta| {
                        if (delta.content) |c| {
                            s.content.appendSlice(s.alloc, c) catch {};
                        }
                    }
                }
            }
            s.user_cb(s.user_ctx, response);
        }
    };
    const stream_ctx = StreamCtx{
        .user_ctx = context,
        .user_cb = callback,
        .content = &collected_content,
        .valid = &is_valid,
        .alloc = arena_alloc,
    };

    self.client.chatCompletionStream(
        self.model,
        self.history.items,
        self.config,
        &stream_ctx,
        &StreamCtx.handle,
    ) catch |err| {
        self.history.shrinkRetainingCapacity(self.history.items.len - user_messages.len);
        return err;
    };

    if (is_valid and collected_content.items.len > 0) {
        self.history.append(self.client.allocator, Message{
            .role = .assistant,
            .content = collected_content.items,
        }) catch return error.OutOfMemory;
    } else {
        self.history.shrinkRetainingCapacity(self.history.items.len - user_messages.len);
    }
}

/// Send a text message and stream the response (convenience wrapper).
pub fn sendMessageStream(
    self: *Chat,
    prompt: []const u8,
    context: anytype,
    callback: *const fn (@TypeOf(context), ChatCompletionResponse) void,
) Client.StreamError!void {
    const messages = [_]Message{.{ .role = .user, .content = prompt }};
    return self.sendStream(&messages, context, callback);
}

/// Return the full conversation history.
pub fn getHistory(self: *const Chat) []const Message {
    return self.history.items;
}

/// Check whether a response contains valid content suitable for history.
fn validateResponse(response: ChatCompletionResponse) bool {
    const choices = response.choices orelse return false;
    if (choices.len == 0) return false;
    // Non-streaming: check message
    if (choices[0].message) |msg| {
        if (msg.content != null or msg.tool_calls != null) return true;
    }
    // Streaming: check delta
    if (choices[0].delta) |delta| {
        if (delta.content != null) return true;
    }
    return false;
}

fn dupeMessage(self: *Chat, msg: Message) std.mem.Allocator.Error!Message {
    const a = self.arena.allocator();
    var duped = msg;
    if (msg.content) |c| duped.content = try a.dupe(u8, c);
    if (msg.name) |n| duped.name = try a.dupe(u8, n);
    if (msg.tool_call_id) |id| duped.tool_call_id = try a.dupe(u8, id);
    if (msg.refusal) |r| duped.refusal = try a.dupe(u8, r);
    if (msg.tool_calls) |tcs| {
        const duped_tcs = try a.alloc(types.ToolCall, tcs.len);
        for (tcs, 0..) |tc, i| {
            duped_tcs[i] = tc;
            if (tc.id) |v| duped_tcs[i].id = try a.dupe(u8, v);
            if (tc.type) |v| duped_tcs[i].type = try a.dupe(u8, v);
            if (tc.function) |f| {
                if (f.name) |v| duped_tcs[i].function.?.name = try a.dupe(u8, v);
                if (f.arguments) |v| duped_tcs[i].function.?.arguments = try a.dupe(u8, v);
            }
        }
        duped.tool_calls = duped_tcs;
    }
    return duped;
}

test "Chat init and deinit" {
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    var chat = Chat.init(&client, "gpt-4o", .{});
    defer chat.deinit();
    try std.testing.expectEqualStrings("gpt-4o", chat.model);
    try std.testing.expectEqual(@as(usize, 0), chat.getHistory().len);
}
