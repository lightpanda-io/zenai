//! Where a System One request can be sent: TypeSafe directly, or Vercel AI
//! Gateway, which serves the same protocol under its own key, URL and model id.

const std = @import("std");
const Client = @import("Client.zig");
const types = @import("types.zig");

pub const Channel = struct {
    name: []const u8,
    api_key_env: [:0]const u8,
    base_url: []const u8,
    model: []const u8,
};

/// Ordered by precedence: a TypeSafe key is the more specific signal.
pub const all = [_]Channel{
    .{
        .name = "typesafe",
        .api_key_env = "TYPESAFE_API_KEY",
        .base_url = Client.default_base_url,
        .model = types.default_model,
    },
    .{
        .name = "vercel",
        .api_key_env = "AI_GATEWAY_API_KEY",
        .base_url = "https://ai-gateway.vercel.sh/typesafe",
        .model = "typesafe-ai/jev",
    },
};

pub const api_keys_hint = "TYPESAFE_API_KEY, or AI_GATEWAY_API_KEY to reach Jev through Vercel AI Gateway";

pub const Credential = struct {
    channel: Channel,
    api_key: [:0]const u8,
};

/// The first channel with a key in `environ`.
pub fn detect(environ: std.process.Environ) ?Credential {
    for (all) |channel| {
        if (environ.getPosix(channel.api_key_env)) |api_key| {
            if (api_key.len > 0) return .{ .channel = channel, .api_key = api_key };
        }
    }
    return null;
}

test "the documented base URLs, in precedence order" {
    try std.testing.expectEqualStrings("https://api.typesafe.ai", all[0].base_url);
    try std.testing.expectEqualStrings("https://ai-gateway.vercel.sh/typesafe", all[1].base_url);
    try std.testing.expectEqualStrings("jev-latest", all[0].model);
    try std.testing.expectEqualStrings("typesafe-ai/jev", all[1].model);

    for (all) |channel| {
        try std.testing.expect(std.mem.indexOf(u8, api_keys_hint, channel.api_key_env) != null);
    }
}

test "detect: the first channel with a key wins" {
    const both: [:null]const ?[*:0]const u8 = &.{ "AI_GATEWAY_API_KEY=gw", "TYPESAFE_API_KEY=ts" };
    const gateway_only: [:null]const ?[*:0]const u8 = &.{"AI_GATEWAY_API_KEY=gw"};
    const unrelated: [:null]const ?[*:0]const u8 = &.{"PATH=/usr/bin"};

    // A TypeSafe key is the more specific signal, whatever the order.
    try std.testing.expectEqualStrings("typesafe", detect(.{ .block = .{ .slice = both } }).?.channel.name);
    try std.testing.expectEqualStrings("ts", detect(.{ .block = .{ .slice = both } }).?.api_key);
    try std.testing.expectEqualStrings("vercel", detect(.{ .block = .{ .slice = gateway_only } }).?.channel.name);
    try std.testing.expect(detect(.{ .block = .{ .slice = unrelated } }) == null);
}
