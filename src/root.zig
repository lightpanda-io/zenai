pub const gemini = struct {
    pub const Client = @import("gemini/Client.zig");
    pub const Chat = @import("gemini/Chat.zig");
    pub const types = @import("gemini/types.zig");
};

pub const openai = struct {
    pub const Client = @import("openai/Client.zig");
    pub const Chat = @import("openai/Chat.zig");
    pub const types = @import("openai/types.zig");
};

pub const provider = @import("provider.zig");

test {
    _ = gemini.Client;
    _ = gemini.Chat;
    _ = gemini.types;
    _ = openai.Client;
    _ = openai.Chat;
    _ = openai.types;
    _ = provider;
}
