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
/// Owns the Content structs (role strings, parts arrays).
history: std.ArrayListUnmanaged(Content),
/// Owns the response objects whose parts are referenced by history.
responses: std.ArrayListUnmanaged(Client.Response),
allocator: std.mem.Allocator,

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
        .allocator = client.allocator,
    };
}

pub fn deinit(self: *Chat) void {
    // Free user messages (we duped them)
    for (self.history.items) |entry| {
        if (std.mem.eql(u8, entry.role orelse "", "user")) {
            self.allocator.free(entry.parts);
        }
    }
    self.history.deinit(self.allocator);

    // Free response objects (owns model messages' backing memory)
    for (self.responses.items) |*resp| {
        resp.deinit();
    }
    self.responses.deinit(self.allocator);
}

pub fn sendMessage(self: *Chat, prompt: []const u8) Client.GenerateContentError!GenerateContentResponse {
    const parts = try self.allocator.alloc(Part, 1);
    parts[0] = .{ .text = prompt };
    const user_content = Content{ .role = "user", .parts = parts };
    try self.history.append(self.allocator, user_content);

    // Send with full history
    const response = try self.client.generateContent(
        self.model,
        self.history.items,
        self.config,
        self.options,
    );

    // Append model response to history
    if (response.value.candidates) |candidates| {
        if (candidates.len > 0) {
            if (candidates[0].content) |content| {
                try self.history.append(self.allocator, content);
            }
        }
    }

    // Keep response alive (it owns the model content's memory)
    try self.responses.append(self.allocator, response);

    return response.value;
}

pub fn getHistory(self: *const Chat) []const Content {
    return self.history.items;
}

test "Chat init and deinit" {
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    var chat = Chat.init(&client, "gemini-2.5-flash", null, .{});
    defer chat.deinit();
    try std.testing.expectEqualStrings("gemini-2.5-flash", chat.model);
    try std.testing.expectEqual(@as(usize, 0), chat.getHistory().len);
}
