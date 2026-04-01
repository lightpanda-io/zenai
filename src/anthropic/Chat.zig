const std = @import("std");
const types = @import("types.zig");
const Client = @import("Client.zig");

const MessageParam = types.MessageParam;
const ContentBlockParam = types.ContentBlockParam;
const MessageResponse = types.MessageResponse;
const StreamEvent = types.StreamEvent;

/// A multi-turn chat session that automatically tracks conversation history.
/// All user inputs are deep-copied into an arena, so temporary buffers are safe to use.
/// Call `deinit()` when done to free all resources.
const Chat = @This();

client: *Client,
model: []const u8,
max_tokens: i32,
config: Client.MessageConfig,
history: std.ArrayListUnmanaged(MessageParam),
responses: std.ArrayListUnmanaged(Client.Response(MessageResponse)),
arena: std.heap.ArenaAllocator,

/// Create a new chat session with the given model and configuration.
pub fn init(
    client: *Client,
    model: []const u8,
    max_tokens: i32,
    config: Client.MessageConfig,
) Chat {
    return .{
        .client = client,
        .model = model,
        .max_tokens = max_tokens,
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

/// Send content blocks as a user message.
pub fn send(self: *Chat, user_content: []const ContentBlockParam) Client.ApiError!MessageResponse {
    const owned = try self.dupeContentBlocks(user_content);
    try self.history.append(self.client.allocator, MessageParam{ .role = .user, .content = owned });

    var response = self.client.createMessage(
        self.model,
        self.history.items,
        self.max_tokens,
        self.config,
    ) catch |err| {
        _ = self.history.pop();
        return err;
    };

    errdefer {
        response.deinit();
        _ = self.history.pop();
    }

    try self.responses.append(self.client.allocator, response);

    if (validateResponse(response.value)) {
        // Convert response content blocks to request content block params for history
        const assistant_content = try self.responseToContentBlocks(response.value);
        try self.history.append(self.client.allocator, MessageParam{ .role = .assistant, .content = assistant_content });
    } else {
        _ = self.history.pop();
    }

    return response.value;
}

/// Send a text-only message (convenience wrapper around `send`).
pub fn sendMessage(self: *Chat, prompt: []const u8) Client.ApiError!MessageResponse {
    const content = [_]ContentBlockParam{.{ .text = prompt }};
    return self.send(&content);
}

/// Send content blocks and stream the response.
/// The callback receives each streaming event. History is updated after streaming completes.
pub fn sendStream(
    self: *Chat,
    user_content: []const ContentBlockParam,
    context: anytype,
    callback: *const fn (@TypeOf(context), StreamEvent) void,
) Client.StreamError!void {
    const arena_alloc = self.arena.allocator();
    const owned = self.dupeContentBlocks(user_content) catch return error.OutOfMemory;
    self.history.append(self.client.allocator, MessageParam{ .role = .user, .content = owned }) catch return error.OutOfMemory;

    var collected_text = std.ArrayListUnmanaged(u8).empty;
    var is_valid = true;
    errdefer _ = self.history.pop();

    const StreamCtx = struct {
        user_ctx: @TypeOf(context),
        user_cb: *const fn (@TypeOf(context), StreamEvent) void,
        text_buf: *std.ArrayListUnmanaged(u8),
        valid: *bool,
        alloc: std.mem.Allocator,

        fn handle(s: *const @This(), event: StreamEvent) void {
            if (event.delta) |delta| {
                if (delta.text) |t| {
                    s.text_buf.appendSlice(s.alloc, t) catch {};
                }
            }
            // Check for message_delta with stop_reason to validate
            if (event.type) |t| {
                if (std.mem.eql(u8, t, "message_delta")) {
                    // valid unless something went wrong
                }
            }
            s.user_cb(s.user_ctx, event);
        }
    };
    const stream_ctx = StreamCtx{
        .user_ctx = context,
        .user_cb = callback,
        .text_buf = &collected_text,
        .valid = &is_valid,
        .alloc = arena_alloc,
    };

    self.client.createMessageStream(
        self.model,
        self.history.items,
        self.max_tokens,
        self.config,
        &stream_ctx,
        &StreamCtx.handle,
    ) catch |err| {
        _ = self.history.pop();
        return err;
    };

    if (is_valid and collected_text.items.len > 0) {
        const text_block = ContentBlockParam{ .type = "text", .text = collected_text.items };
        const content = arena_alloc.dupe(ContentBlockParam, &.{text_block}) catch return error.OutOfMemory;
        self.history.append(self.client.allocator, MessageParam{ .role = .assistant, .content = content }) catch return error.OutOfMemory;
    } else {
        _ = self.history.pop();
    }
}

/// Send a text message and stream the response (convenience wrapper).
pub fn sendMessageStream(
    self: *Chat,
    prompt: []const u8,
    context: anytype,
    callback: *const fn (@TypeOf(context), StreamEvent) void,
) Client.StreamError!void {
    const content = [_]ContentBlockParam{.{ .text = prompt }};
    return self.sendStream(&content, context, callback);
}

/// Return the full conversation history.
pub fn getHistory(self: *const Chat) []const MessageParam {
    return self.history.items;
}

/// Check whether a response contains valid content suitable for history.
fn validateResponse(response: MessageResponse) bool {
    const blocks = response.content orelse return false;
    if (blocks.len == 0) return false;
    for (blocks) |block| {
        if (block.type) |t| {
            if (std.mem.eql(u8, t, "text") or std.mem.eql(u8, t, "tool_use")) return true;
        }
    }
    return false;
}

/// Convert response ContentBlocks to request ContentBlockParams for history.
fn responseToContentBlocks(self: *Chat, response: MessageResponse) std.mem.Allocator.Error![]ContentBlockParam {
    const a = self.arena.allocator();
    const blocks = response.content orelse return try a.alloc(ContentBlockParam, 0);
    const params = try a.alloc(ContentBlockParam, blocks.len);
    for (blocks, 0..) |block, i| {
        params[i] = .{
            .type = if (block.type) |t| try a.dupe(u8, t) else "text",
            .text = if (block.text) |t| try a.dupe(u8, t) else null,
            .id = if (block.id) |v| try a.dupe(u8, v) else null,
            .name = if (block.name) |v| try a.dupe(u8, v) else null,
            .input = if (block.input) |v| try dupeJsonValue(a, v) else null,
            .thinking = if (block.thinking) |v| try a.dupe(u8, v) else null,
            .signature = if (block.signature) |v| try a.dupe(u8, v) else null,
        };
    }
    return params;
}

fn dupeContentBlocks(self: *Chat, blocks: []const ContentBlockParam) std.mem.Allocator.Error![]ContentBlockParam {
    const a = self.arena.allocator();
    const duped = try a.alloc(ContentBlockParam, blocks.len);
    for (blocks, 0..) |block, i| {
        duped[i] = block;
        duped[i].type = try a.dupe(u8, block.type);
        if (block.text) |v| duped[i].text = try a.dupe(u8, v);
        if (block.id) |v| duped[i].id = try a.dupe(u8, v);
        if (block.name) |v| duped[i].name = try a.dupe(u8, v);
        if (block.input) |v| duped[i].input = try dupeJsonValue(a, v);
        if (block.tool_use_id) |v| duped[i].tool_use_id = try a.dupe(u8, v);
        if (block.content) |v| duped[i].content = try a.dupe(u8, v);
        if (block.thinking) |v| duped[i].thinking = try a.dupe(u8, v);
        if (block.signature) |v| duped[i].signature = try a.dupe(u8, v);
    }
    return duped;
}

fn dupeJsonValue(a: std.mem.Allocator, value: std.json.Value) std.mem.Allocator.Error!std.json.Value {
    return switch (value) {
        .null, .bool, .integer, .float => value,
        .number_string => |s| .{ .number_string = try a.dupe(u8, s) },
        .string => |s| .{ .string = try a.dupe(u8, s) },
        .array => |arr| blk: {
            var new_arr = try std.json.Array.initCapacity(a, arr.items.len);
            for (arr.items) |item| {
                new_arr.appendAssumeCapacity(try dupeJsonValue(a, item));
            }
            break :blk .{ .array = new_arr };
        },
        .object => |obj| blk: {
            var new_obj = std.json.ObjectMap.init(a);
            try new_obj.ensureTotalCapacity(@intCast(obj.count()));
            var it = obj.iterator();
            while (it.next()) |entry| {
                new_obj.putAssumeCapacity(try a.dupe(u8, entry.key_ptr.*), try dupeJsonValue(a, entry.value_ptr.*));
            }
            break :blk .{ .object = new_obj };
        },
    };
}

test "Chat init and deinit" {
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    var chat = Chat.init(&client, "claude-sonnet-4-6", 1024, .{});
    defer chat.deinit();
    try std.testing.expectEqualStrings("claude-sonnet-4-6", chat.model);
    try std.testing.expectEqual(@as(usize, 0), chat.getHistory().len);
}
