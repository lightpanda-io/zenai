const std = @import("std");
const types = @import("types.zig");
const Client = @import("Client.zig");
const json = @import("../json.zig");

const Content = types.Content;
const Part = types.Part;
const GenerationConfig = types.GenerationConfig;
const GenerateContentResponse = types.GenerateContentResponse;

/// A multi-turn chat session that automatically tracks conversation history.
/// All user inputs are deep-copied into an arena, so temporary buffers are safe to use.
/// Call `deinit()` when done to free all resources.
const Chat = @This();

client: *Client,
model: []const u8,
config: ?GenerationConfig,
options: Client.RequestOptions,
history: std.ArrayListUnmanaged(Content),
last_response: ?Client.Response(GenerateContentResponse) = null,
arena: std.heap.ArenaAllocator,

/// Create a new chat session with the given model and configuration.
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
        .arena = std.heap.ArenaAllocator.init(client.allocator),
    };
}

/// Release all resources: conversation history, response arenas, and owned strings.
pub fn deinit(self: *Chat) void {
    self.history.deinit(self.client.allocator);
    if (self.last_response) |*r| r.deinit();
    self.arena.deinit();
}

/// Send a multimodal message (text, images, files, function responses, etc.).
/// The returned response is valid until the next `send()` or `deinit()` call.
pub fn send(self: *Chat, user_parts: []const Part) Client.ApiError!GenerateContentResponse {
    if (self.last_response) |*r| r.deinit();
    self.last_response = null;

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

    if (validateResponse(response.value)) {
        const model_content = response.value.candidates.?[0].content.?;
        const owned_parts = try self.dupeParts(model_content.parts);
        try self.history.append(self.client.allocator, Content{ .role = model_content.role, .parts = owned_parts });
    } else {
        _ = self.history.pop();
    }

    self.last_response = response;
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

    var collected_parts: std.ArrayListUnmanaged(Part) = .empty;
    errdefer _ = self.history.pop();

    const StreamCtx = struct {
        user_ctx: @TypeOf(context),
        user_cb: *const fn (@TypeOf(context), GenerateContentResponse) void,
        parts: *std.ArrayListUnmanaged(Part),
        alloc: std.mem.Allocator,

        fn handle(s: *const @This(), response: GenerateContentResponse) void {
            if (response.candidates) |candidates| {
                if (candidates.len > 0) {
                    if (candidates[0].content) |content| {
                        s.parts.appendSlice(s.alloc, content.parts) catch {};
                    }
                }
            }
            s.user_cb(s.user_ctx, response);
        }
    };
    const stream_ctx = StreamCtx{
        .user_ctx = context,
        .user_cb = callback,
        .parts = &collected_parts,
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

    if (collected_parts.items.len > 0) {
        const model_parts = arena_alloc.dupe(Part, collected_parts.items) catch return error.OutOfMemory;
        self.history.append(self.client.allocator, Content{ .role = "model", .parts = model_parts }) catch return error.OutOfMemory;
    } else {
        // Invalid response: remove the user turn so it doesn't pollute future requests.
        _ = self.history.pop();
    }
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

/// Return the full conversation history (user and model turns).
pub fn getHistory(self: *const Chat) []const Content {
    return self.history.items;
}

/// Check whether a response contains valid content suitable for history.
fn validateResponse(response: GenerateContentResponse) bool {
    const candidates = response.candidates orelse return false;
    if (candidates.len == 0) return false;
    const content = candidates[0].content orelse return false;
    if (content.parts.len == 0) return false;
    for (content.parts) |part| {
        if (part.text != null or
            part.inlineData != null or
            part.fileData != null or
            part.functionCall != null or
            part.functionResponse != null or
            part.executableCode != null or
            part.codeExecutionResult != null) return true;
    }
    return false;
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
        }
        if (part.fileData) |fd| {
            if (fd.fileUri) |u| duped[i].fileData.?.fileUri = try a.dupe(u8, u);
            if (fd.mimeType) |m| duped[i].fileData.?.mimeType = try a.dupe(u8, m);
        }
        if (part.functionCall) |fc| {
            if (fc.id) |v| duped[i].functionCall.?.id = try a.dupe(u8, v);
            if (fc.name) |v| duped[i].functionCall.?.name = try a.dupe(u8, v);
            if (fc.args) |v| duped[i].functionCall.?.args = try json.dupeValue(a, v);
        }
        if (part.functionResponse) |fr| {
            if (fr.id) |v| duped[i].functionResponse.?.id = try a.dupe(u8, v);
            if (fr.name) |v| duped[i].functionResponse.?.name = try a.dupe(u8, v);
            if (fr.response) |v| duped[i].functionResponse.?.response = try json.dupeValue(a, v);
        }
        if (part.executableCode) |ec| {
            if (ec.code) |v| duped[i].executableCode.?.code = try a.dupe(u8, v);
        }
        if (part.codeExecutionResult) |cr| {
            if (cr.output) |v| duped[i].codeExecutionResult.?.output = try a.dupe(u8, v);
        }
        if (part.thoughtSignature) |v| duped[i].thoughtSignature = try a.dupe(u8, v);
        if (part.partMetadata) |v| duped[i].partMetadata = try json.dupeValue(a, v);
    }
    return duped;
}

test "Chat init and deinit" {
    var client = Client.init(std.testing.io, std.testing.allocator, "test-key", .{});
    defer client.deinit();
    var chat = Chat.init(&client, "gemini-2.5-flash", null, .{});
    defer chat.deinit();
    try std.testing.expectEqualStrings("gemini-2.5-flash", chat.model);
    try std.testing.expectEqual(@as(usize, 0), chat.getHistory().len);
}
