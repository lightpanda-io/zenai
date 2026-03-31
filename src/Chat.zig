const std = @import("std");
const types = @import("types.zig");
const Client = @import("Client.zig");

const Content = types.Content;
const Part = types.Part;
const GenerationConfig = types.GenerationConfig;
const GenerateContentResponse = types.GenerateContentResponse;

const Chat = @This();

client: *Client,
model: []const u8,
config: ?GenerationConfig,
options: Client.RequestOptions,
history: std.ArrayListUnmanaged(Content),
responses: std.ArrayListUnmanaged(Client.Response(GenerateContentResponse)),
arena: std.heap.ArenaAllocator,

pub fn init(
    client: *Client,
    model: []const u8,
    config: ?GenerationConfig,
    options: Client.RequestOptions,
) Chat {
    return .{
        .client = client,
        .model = model,
        .config = config,
        .options = options,
        .history = .empty,
        .responses = .empty,
        .arena = std.heap.ArenaAllocator.init(client.allocator),
    };
}

pub fn deinit(self: *Chat) void {
    self.history.deinit(self.client.allocator);

    for (self.responses.items) |*resp| resp.deinit();
    self.responses.deinit(self.client.allocator);

    self.arena.deinit();
}

/// Send a multimodal message (text, images, files, function responses, etc.).
pub fn send(self: *Chat, user_parts: []const Part) Client.ApiError!GenerateContentResponse {
    const owned = try self.dupeParts(user_parts);
    try self.history.append(self.client.allocator, Content{ .role = "user", .parts = owned });

    var response = self.client.generateContent(
        self.model,
        self.history.items,
        self.config,
        self.options,
    ) catch |err| {
        _ = self.history.pop();
        return err;
    };

    errdefer {
        response.deinit();
        _ = self.history.pop();
    }

    try self.responses.append(self.client.allocator, response);

    if (response.value.candidates) |candidates| {
        if (candidates.len > 0) {
            if (candidates[0].content) |content| {
                try self.history.append(self.client.allocator, content);
            }
        }
    }

    return response.value;
}

/// Send a text-only message (convenience wrapper around `send`).
pub fn sendMessage(self: *Chat, prompt: []const u8) Client.ApiError!GenerateContentResponse {
    const parts = [_]Part{.{ .text = prompt }};
    return self.send(&parts);
}

/// Send multimodal parts and stream the response.
/// The callback receives each chunk. History is updated after streaming completes.
pub fn sendStream(
    self: *Chat,
    user_parts: []const Part,
    context: anytype,
    callback: *const fn (@TypeOf(context), GenerateContentResponse) void,
) Client.StreamError!void {
    const arena_alloc = self.arena.allocator();
    const owned = self.dupeParts(user_parts) catch return error.OutOfMemory;
    self.history.append(self.client.allocator, Content{ .role = "user", .parts = owned }) catch return error.OutOfMemory;

    var text_buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer _ = self.history.pop();

    const StreamCtx = struct {
        user_ctx: @TypeOf(context),
        user_cb: *const fn (@TypeOf(context), GenerateContentResponse) void,
        buf: *std.ArrayListUnmanaged(u8),
        alloc: std.mem.Allocator,

        fn handle(s: *const @This(), response: GenerateContentResponse) void {
            if (response.text()) |t| {
                s.buf.appendSlice(s.alloc, t) catch {};
            }
            s.user_cb(s.user_ctx, response);
        }
    };
    const stream_ctx = StreamCtx{
        .user_ctx = context,
        .user_cb = callback,
        .buf = &text_buf,
        .alloc = arena_alloc,
    };

    self.client.generateContentStream(
        self.model,
        self.history.items,
        self.config,
        self.options,
        &stream_ctx,
        &StreamCtx.handle,
    ) catch |err| {
        _ = self.history.pop();
        return err;
    };

    const model_text = text_buf.items;
    const model_parts = arena_alloc.alloc(Part, 1) catch return error.OutOfMemory;
    model_parts[0] = .{ .text = model_text };
    self.history.append(self.client.allocator, Content{ .role = "model", .parts = model_parts }) catch return error.OutOfMemory;
}

/// Send a text message and stream the response (convenience wrapper).
pub fn sendMessageStream(
    self: *Chat,
    prompt: []const u8,
    context: anytype,
    callback: *const fn (@TypeOf(context), GenerateContentResponse) void,
) Client.StreamError!void {
    const parts = [_]Part{.{ .text = prompt }};
    return self.sendStream(&parts, context, callback);
}

pub fn getHistory(self: *const Chat) []const Content {
    return self.history.items;
}

fn dupeParts(self: *Chat, parts: []const Part) std.mem.Allocator.Error![]Part {
    const a = self.arena.allocator();
    const duped = try a.alloc(Part, parts.len);
    for (parts, 0..) |part, i| {
        duped[i] = part;
        if (part.text) |txt| duped[i].text = try a.dupe(u8, txt);
        if (part.inlineData) |blob| {
            if (blob.data) |d| duped[i].inlineData.?.data = try a.dupe(u8, d);
            if (blob.mimeType) |m| duped[i].inlineData.?.mimeType = try a.dupe(u8, m);
            if (blob.displayName) |d| duped[i].inlineData.?.displayName = try a.dupe(u8, d);
        }
        if (part.fileData) |fd| {
            if (fd.fileUri) |u| duped[i].fileData.?.fileUri = try a.dupe(u8, u);
            if (fd.mimeType) |m| duped[i].fileData.?.mimeType = try a.dupe(u8, m);
        }
    }
    return duped;
}

test "Chat init and deinit" {
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    var chat = Chat.init(&client, "gemini-2.5-flash", null, .{});
    defer chat.deinit();
    try std.testing.expectEqualStrings("gemini-2.5-flash", chat.model);
    try std.testing.expectEqual(@as(usize, 0), chat.getHistory().len);
}
