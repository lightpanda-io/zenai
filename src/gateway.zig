//! A gateway: one URL and token in front of several model providers, whose
//! config (a JSON shape the caller defines) names a `Route` per use.
//! `Gateway` turns a route into what `provider.Client.init` or the decision
//! pool takes; `loadConfig` fetches the config through a disk cache.

const std = @import("std");
const http = @import("http.zig");
const json = @import("json.zig");
const retry = @import("retry.zig");
const provider = @import("provider.zig");
const Pool = @import("decisions/Pool.zig");

const log = std.log.scoped(.zenai);

/// Where one use of the gateway goes: the wire protocol, a zenai provider or
/// decision provider name, under `path` on the gateway.
pub const Route = struct {
    protocol: []const u8,
    path: []const u8,
    /// The only model the route serves; null lets the caller pick.
    model: ?[]const u8 = null,
};

/// The response headers of RFC 9745 and RFC 8594, which every endpoint a
/// `Gateway` builds captures; see `deprecation`.
pub const deprecation_headers = [_][]const u8{ "deprecation", "sunset" };

/// A route the gateway marks deprecated, from a response's captured headers.
pub const Deprecation = struct {
    /// The `Sunset` date as sent (an HTTP date); null when the gateway gives
    /// none.
    sunset: ?[]const u8,
};

/// Null when the response doesn't mark its route deprecated.
pub fn deprecation(headers: http.CapturedHeaders) ?Deprecation {
    if (headers.get("deprecation") == null) return null;
    return .{ .sunset = headers.get("sunset") };
}

pub const Gateway = struct {
    /// No trailing slash; route paths start with one.
    url: []const u8,
    /// Sent as each protocol's own API key.
    key: [:0]const u8,
    /// Sent on every request, e.g. a client version header. Borrowed.
    extra_headers: []const std.http.Header = &.{},
    /// Response headers to keep besides `deprecation_headers`, e.g. the
    /// gateway's own error code. Borrowed.
    capture_headers: []const []const u8 = &.{},

    /// The endpoint for a chat `route`, allocated in `arena`; null when its
    /// protocol isn't one zenai can speak at another URL.
    pub fn endpoint(self: Gateway, arena: std.mem.Allocator, route: Route) std.mem.Allocator.Error!?provider.Endpoint {
        const tag = std.meta.stringToEnum(provider.Tag, route.protocol) orelse return null;
        switch (tag) {
            .gemini, .anthropic, .openai, .openai_compatible => {},
            else => return null,
        }
        return .{
            .backend = .of(tag),
            .key = self.key,
            .base_url = try self.routeUrl(arena, route),
            .default_model = route.model,
            .models = if (route.model) |m| try arena.dupe([]const u8, &.{m}) else null,
            .extra_headers = self.extra_headers,
            .capture_headers = try std.mem.concat(arena, []const u8, &.{ self.capture_headers, &deprecation_headers }),
        };
    }

    /// The decision-pool credentials for `route`, allocated in `arena`; null
    /// when its protocol isn't a decision provider.
    pub fn decisionCredentials(self: Gateway, arena: std.mem.Allocator, route: Route) std.mem.Allocator.Error!?Pool.Credentials {
        return .{
            .provider = std.meta.stringToEnum(Pool.Provider, route.protocol) orelse return null,
            .api_key = self.key,
            .base_url = try self.routeUrl(arena, route),
            .extra_headers = self.extra_headers,
        };
    }

    fn routeUrl(self: Gateway, arena: std.mem.Allocator, route: Route) std.mem.Allocator.Error![:0]u8 {
        return std.mem.concatWithSentinel(arena, u8, &.{ self.url, route.path }, 0);
    }
};

pub const LoadOptions = struct {
    /// Fetched with a GET and the gateway's key as a bearer token.
    url: []const u8,
    key: []const u8,
    extra_headers: []const std.http.Header = &.{},
    /// The JSON cache file; null skips caching. Keyed by `url`, so a copy
    /// from another gateway is ignored.
    cache_path: ?[]const u8 = null,
    /// How long a cached copy is used without asking the gateway.
    ttl_ms: i64 = 6 * std.time.ms_per_hour,
    request_timeout_ms: u32 = 10_000,
    /// One attempt by default: a startup shouldn't wait out backoff when the
    /// cache is there to fall back on.
    retry_policy: retry.RetryPolicy = .disabled,
    log_tag: []const u8 = "gateway",
};

pub const LoadError = error{
    /// Neither the gateway nor the cache has a config; the fetch error is logged.
    GatewayUnavailable,
};

/// The gateway's config of type `T`, allocated in `arena`: a cached copy
/// younger than `ttl_ms`, else a fresh one (which refreshes the cache), else
/// an older cached copy.
pub fn loadConfig(comptime T: type, io: std.Io, arena: std.mem.Allocator, options: LoadOptions) LoadError!T {
    const Cache = struct { fetched_ms: i64, url: []const u8, config: T };
    const now = std.Io.Clock.real.now(io).toMilliseconds();
    const cached: ?Cache = if (options.cache_path) |path| blk: {
        const c = json.readFile(Cache, io, arena, path, 1024 * 1024) orelse break :blk null;
        break :blk if (std.mem.eql(u8, c.url, options.url)) c else null;
    } else null;
    if (cached) |c| if (now - c.fetched_ms <= options.ttl_ms) return c.config;

    const config = fetchConfig(T, io, arena, options) catch |err| {
        if (cached) |c| {
            log.warn("{s}: config fetch failed ({t}); using the cached copy", .{ options.log_tag, err });
            return c.config;
        }
        log.err("{s}: config fetch from {s} failed ({t})", .{ options.log_tag, options.url, err });
        return error.GatewayUnavailable;
    };
    if (options.cache_path) |path| json.writeFileAtomic(io, path, Cache{ .fetched_ms = now, .url = options.url, .config = config }, .default_file) catch |err| {
        log.warn("{s}: config cache write failed ({t})", .{ options.log_tag, err });
    };
    return config;
}

fn fetchConfig(comptime T: type, io: std.Io, arena: std.mem.Allocator, options: LoadOptions) !T {
    var transport: http.Transport = .init(io, arena, .{
        .request_timeout_ms = options.request_timeout_ms,
        .retry_policy = options.retry_policy,
        .extra_headers = options.extra_headers,
        .log_tag = options.log_tag,
    });
    defer transport.deinit();
    var bearer: http.BearerAuth = .{};
    const response = try transport.getJson(options.url, &.{try bearer.header(arena, options.key)}, T);
    // Not deinit'ed: its strings live in `arena`.
    return response.value;
}

const testing = std.testing;

const test_gateway: Gateway = .{ .url = "https://gw", .key = "tok", .capture_headers = &.{"x-gw-error"} };

test "Gateway.endpoint: the route's protocol at the gateway's URL, serving only the route's model" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const ep = (try test_gateway.endpoint(a, .{ .protocol = "gemini", .path = "/gemini", .model = "default" })).?;
    try testing.expectEqual(provider.Tag.gemini, ep.tag());
    try testing.expectEqualStrings("https://gw/gemini", ep.base_url.?);
    try testing.expectEqualStrings("tok", ep.key);
    try testing.expectEqualStrings("default", ep.defaultModel());
    try testing.expectEqualDeep(@as([]const []const u8, &.{"default"}), ep.models.?);
    try testing.expectEqualDeep(@as([]const []const u8, &.{ "x-gw-error", "deprecation", "sunset" }), ep.capture_headers);

    const any_model = (try test_gateway.endpoint(a, .{ .protocol = "anthropic", .path = "/anthropic" })).?;
    try testing.expectEqual(null, any_model.models);
    try testing.expectEqualStrings(provider.defaultModel(.anthropic), any_model.defaultModel());

    for ([_][]const u8{ "brand-new", "ollama", "codex" }) |protocol| {
        try testing.expectEqual(null, try test_gateway.endpoint(a, .{ .protocol = protocol, .path = "/x" }));
    }
}

test "Gateway.decisionCredentials: decision providers only" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const c = (try test_gateway.decisionCredentials(arena.allocator(), .{ .protocol = "typesafe", .path = "/typesafe" })).?;
    try testing.expectEqual(Pool.Provider.typesafe, c.provider);
    try testing.expectEqualStrings("https://gw/typesafe", c.base_url.?);
    try testing.expectEqualStrings("tok", c.api_key);
    try testing.expectEqual(null, try test_gateway.decisionCredentials(arena.allocator(), .{ .protocol = "anthropic", .path = "/a" }));
}

test "deprecation: the sunset date when the gateway gives one" {
    var none = [_]std.http.Header{};
    try testing.expectEqual(null, deprecation(.{ .items = &none }));
    var plain = [_]std.http.Header{.{ .name = "Deprecation", .value = "@1764547200" }};
    try testing.expectEqual(null, deprecation(.{ .items = &plain }).?.sunset);
    var dated = [_]std.http.Header{ .{ .name = "deprecation", .value = "@1764547200" }, .{ .name = "sunset", .value = "Tue, 01 Dec 2026 00:00:00 GMT" } };
    try testing.expectEqualStrings("Tue, 01 Dec 2026 00:00:00 GMT", deprecation(.{ .items = &dated }).?.sunset.?);
}

test "loadConfig: a fresh cached copy for the same gateway skips the fetch" {
    const Config = struct { agent: Route };
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "gw.json" });

    const now = std.Io.Clock.real.now(testing.io).toMilliseconds();
    try json.writeFileAtomic(testing.io, path, .{ .fetched_ms = now, .url = "http://127.0.0.1:1/v1/config", .config = Config{ .agent = .{ .protocol = "gemini", .path = "/g" } } }, .default_file);
    const config = try loadConfig(Config, testing.io, a, .{ .url = "http://127.0.0.1:1/v1/config", .key = "tok", .cache_path = path });
    try testing.expectEqualStrings("/g", config.agent.path);
}
