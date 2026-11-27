//! A gateway: one URL and key in front of several model providers. Its config,
//! a JSON document whose shape the caller defines, names a `Route` per use;
//! `Gateway` fetches that config and turns a route into what
//! `provider.Client.init` or the decision pool takes.

const std = @import("std");
const http = @import("http.zig");
const json = @import("json.zig");
const retry = @import("retry.zig");
const provider = @import("provider.zig");
const Pool = @import("decisions/Pool.zig");

const log = std.log.scoped(.zenai);

/// Where one use of the gateway goes: a zenai provider or decision provider
/// name as the wire protocol, under `path` on the gateway.
pub const Route = struct {
    protocol: []const u8,
    path: []const u8,
    /// The only model the route serves; null lets the caller pick.
    model: ?[]const u8 = null,
};

pub const RouteError = error{
    /// The route's protocol isn't one this version speaks at another URL.
    UnsupportedProtocol,
} || std.mem.Allocator.Error;

pub const ConfigError = error{
    /// Neither the gateway nor the cache has a config; the cause is logged.
    GatewayUnavailable,
} || std.mem.Allocator.Error;

pub const ConfigOptions = struct {
    /// Under the gateway's URL, fetched with a GET and the key as a bearer
    /// token.
    path: []const u8,
    /// The JSON cache file; null skips caching. A copy fetched from another
    /// URL or with other `extra_headers` is ignored.
    cache_path: ?[]const u8 = null,
    /// How long a cached copy is used without asking the gateway.
    ttl_ms: i64 = 6 * std.time.ms_per_hour,
    request_timeout_ms: u32 = 10_000,
    /// One attempt by default: the cache is the fallback.
    retry_policy: retry.RetryPolicy = .disabled,
};

pub const Gateway = struct {
    /// No trailing slash; paths start with one.
    url: []const u8,
    /// Sent as each protocol's own API key.
    key: [:0]const u8,
    /// Sent on every request, e.g. a client version header. Borrowed.
    extra_headers: []const std.http.Header = &.{},
    /// Response headers every endpoint keeps, e.g. the gateway's own error
    /// code. Borrowed.
    capture_headers: []const []const u8 = &.{},

    /// The gateway's config as `T`, allocated in `arena`: a cached copy
    /// younger than `ttl_ms`, else a fresh one (which refreshes the cache),
    /// else an older cached copy.
    pub fn config(self: Gateway, comptime T: type, io: std.Io, arena: std.mem.Allocator, options: ConfigOptions) ConfigError!T {
        const Cache = struct { fetched_ms: i64, url: []const u8, headers: []const std.http.Header, config: T };
        const url = try std.mem.concat(arena, u8, &.{ self.url, options.path });
        const now = std.Io.Clock.real.now(io).toMilliseconds();
        const cached: ?Cache = if (options.cache_path) |path| blk: {
            const c = json.readFile(Cache, io, arena, path, 1024 * 1024) orelse break :blk null;
            break :blk if (std.mem.eql(u8, c.url, url) and sameHeaders(c.headers, self.extra_headers)) c else null;
        } else null;
        if (cached) |c| if (now - c.fetched_ms <= options.ttl_ms) return c.config;

        const fresh = self.fetch(T, io, arena, url, options) catch |err| {
            if (cached) |c| {
                log.warn("gateway: config fetch from {s} failed ({t}); using the cached copy", .{ url, err });
                return c.config;
            }
            log.warn("gateway: config fetch from {s} failed ({t})", .{ url, err });
            return error.GatewayUnavailable;
        };
        if (options.cache_path) |path| {
            json.writeFileAtomic(io, path, Cache{ .fetched_ms = now, .url = url, .headers = self.extra_headers, .config = fresh }, .default_file) catch |err| {
                log.warn("gateway: config cache write to {s} failed ({t})", .{ path, err });
            };
        }
        return fresh;
    }

    /// The client endpoint for `route`, allocated in `arena`.
    pub fn endpoint(self: Gateway, arena: std.mem.Allocator, route: Route) RouteError!provider.Endpoint {
        const tag = std.meta.stringToEnum(provider.Tag, route.protocol) orelse return error.UnsupportedProtocol;
        switch (tag) {
            .gemini, .anthropic, .openai, .openai_compatible => {},
            else => return error.UnsupportedProtocol,
        }
        return .{
            .backend = .of(tag),
            .key = self.key,
            .base_url = try self.routeUrl(arena, route),
            .default_model = route.model,
            .models = if (route.model) |m| try arena.dupe([]const u8, &.{m}) else null,
            .extra_headers = self.extra_headers,
            .capture_headers = self.capture_headers,
        };
    }

    /// The decision-pool credentials for `route`, allocated in `arena`.
    pub fn decisionCredentials(self: Gateway, arena: std.mem.Allocator, route: Route) RouteError!Pool.Credentials {
        return .{
            .provider = std.meta.stringToEnum(Pool.Provider, route.protocol) orelse return error.UnsupportedProtocol,
            .api_key = self.key,
            .base_url = try self.routeUrl(arena, route),
            .extra_headers = self.extra_headers,
        };
    }

    fn routeUrl(self: Gateway, arena: std.mem.Allocator, route: Route) std.mem.Allocator.Error![:0]u8 {
        return std.mem.concatWithSentinel(arena, u8, &.{ self.url, route.path }, 0);
    }

    fn fetch(self: Gateway, comptime T: type, io: std.Io, arena: std.mem.Allocator, url: []const u8, options: ConfigOptions) !T {
        var transport: http.Transport = .init(io, arena, .{
            .request_timeout_ms = options.request_timeout_ms,
            .retry_policy = options.retry_policy,
            .extra_headers = self.extra_headers,
            .log_tag = "gateway",
        });
        defer transport.deinit();
        var bearer: http.BearerAuth = .{};
        const response = try transport.getJson(url, &.{try bearer.header(arena, self.key)}, T);
        // Not deinit'ed: its strings live in `arena`.
        return response.value;
    }
};

fn sameHeaders(a: []const std.http.Header, b: []const std.http.Header) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (!std.mem.eql(u8, x.name, y.name) or !std.mem.eql(u8, x.value, y.value)) return false;
    }
    return true;
}

const testing = std.testing;

const test_gateway: Gateway = .{ .url = "https://gw", .key = "tok", .capture_headers = &.{"x-gw-error"} };

test "Gateway.endpoint: the route's protocol at the gateway's URL, serving only the route's model" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const ep = try test_gateway.endpoint(a, .{ .protocol = "gemini", .path = "/gemini", .model = "default" });
    try testing.expectEqual(provider.Tag.gemini, ep.tag());
    try testing.expectEqualStrings("https://gw/gemini", ep.base_url.?);
    try testing.expectEqualStrings("tok", ep.key);
    try testing.expectEqualStrings("default", ep.defaultModel());
    try testing.expectEqualDeep(@as([]const []const u8, &.{"default"}), ep.models.?);
    try testing.expectEqualDeep(@as([]const []const u8, &.{"x-gw-error"}), ep.capture_headers);

    const any_model = try test_gateway.endpoint(a, .{ .protocol = "anthropic", .path = "/anthropic" });
    try testing.expectEqual(null, any_model.models);
    try testing.expectEqualStrings(provider.defaultModel(.anthropic), any_model.defaultModel());

    for ([_][]const u8{ "brand-new", "ollama", "codex" }) |protocol| {
        try testing.expectError(error.UnsupportedProtocol, test_gateway.endpoint(a, .{ .protocol = protocol, .path = "/x" }));
    }
}

test "Gateway.decisionCredentials: decision providers only" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const c = try test_gateway.decisionCredentials(arena.allocator(), .{ .protocol = "typesafe", .path = "/typesafe" });
    try testing.expectEqual(Pool.Provider.typesafe, c.provider);
    try testing.expectEqualStrings("https://gw/typesafe", c.base_url.?);
    try testing.expectEqualStrings("tok", c.api_key);
    try testing.expectError(error.UnsupportedProtocol, test_gateway.decisionCredentials(arena.allocator(), .{ .protocol = "anthropic", .path = "/a" }));
}

test "Gateway.config: a fresh cached copy skips the fetch unless the request differs" {
    const Config = struct { agent: Route };
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "gw.json" });

    const v1 = [_]std.http.Header{.{ .name = "X-Version", .value = "1" }};
    const gateway: Gateway = .{ .url = "http://127.0.0.1:1", .key = "tok", .extra_headers = &v1 };
    const now = std.Io.Clock.real.now(testing.io).toMilliseconds();
    try json.writeFileAtomic(testing.io, path, .{
        .fetched_ms = now,
        .url = "http://127.0.0.1:1/config",
        .headers = &v1,
        .config = Config{ .agent = .{ .protocol = "gemini", .path = "/g" } },
    }, .default_file);
    const config = try gateway.config(Config, testing.io, a, .{ .path = "/config", .cache_path = path });
    try testing.expectEqualStrings("/g", config.agent.path);

    // Another client version asks again, and the closed port fails it.
    const v2 = [_]std.http.Header{.{ .name = "X-Version", .value = "2" }};
    const upgraded: Gateway = .{ .url = gateway.url, .key = "tok", .extra_headers = &v2 };
    try testing.expectError(error.GatewayUnavailable, upgraded.config(Config, testing.io, a, .{ .path = "/config", .cache_path = path }));
}
