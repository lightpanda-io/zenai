//! Where a System One request can be sent. TypeSafe serves the protocol
//! directly; Vercel AI Gateway serves the same request and response shapes at
//! its own base URL, so reaching Jev through it is a different key, base URL
//! and model id rather than a different client.
//!
//! The chat side of the same question lives in `provider.openAiPreset` /
//! `provider.envApiKey`; this is that table for the judgement side.

const std = @import("std");
const Client = @import("Client.zig");
const types = @import("types.zig");

pub const Channel = struct {
    name: []const u8,
    /// Read from the environment only: a remembered-settings file holds a
    /// provider and a model, never a secret.
    api_key_env: [:0]const u8,
    base_url: []const u8,
    model: []const u8,
};

/// Ordered by precedence: a TypeSafe key is the more specific signal, since a
/// gateway key also serves a hundred chat models.
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
    // `Client` appends `/v1/systemone` and `/v1/models`, so these are the base
    // URLs each service documents for a TypeSafe client.
    try std.testing.expectEqualStrings("https://api.typesafe.ai", all[0].base_url);
    try std.testing.expectEqualStrings("https://ai-gateway.vercel.sh/typesafe", all[1].base_url);
    // The gateway routes by a namespaced id; TypeSafe's own API does not know it.
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
