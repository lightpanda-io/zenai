//! A pool of idle decision-model clients: an ask takes one, or makes one, and
//! puts it back, so concurrent asks each get their own and keep its connection
//! warm.
//!
//! Which provider answers is read from the environment on every ask, and so
//! are its credentials. TypeSafe answers unless the provider variable names
//! Gemini: a Google key is set for many other reasons, so it is no consent to
//! send content to a decision model. A client borrows its API key and base
//! URL rather than copying them, and the environment they came from can
//! change under it, so the pool keeps its own copies and retires any client
//! that no longer matches.

const std = @import("std");
const types = @import("types.zig");
const http = @import("../http.zig");
const retry = @import("../retry.zig");
const provider = @import("../provider.zig");
const TypesafeClient = @import("../typesafe/Client.zig");
const typesafe_types = @import("../typesafe/types.zig");
const GeminiClient = @import("../gemini/Client.zig");
const gemini_decisions = @import("../gemini/decisions.zig");
const typesafe_adapter = @import("typesafe.zig");
const gemini_adapter = @import("gemini.zig");

const Pool = @This();

io: std.Io,
allocator: std.mem.Allocator,
/// Called whenever a client has to be made, rather than held as a snapshot:
/// `setenv` can move the block a `std.process.Environ` points into, and
/// exporting a key part-way through a run should take effect without a
/// restart.
environ: *const fn () std.process.Environ,
options: Options,
idle: std.ArrayList(Pooled) = .empty,
mutex: std.Io.Mutex = .init,

pub const Provider = enum {
    gemini,
    typesafe,

    pub fn defaultModel(self: Provider) []const u8 {
        return switch (self) {
            .gemini => gemini_decisions.default_model,
            .typesafe => typesafe_types.default_model,
        };
    }
};

/// The variable that points the Gemini client somewhere other than Google, as
/// the google-genai SDKs name it.
pub const gemini_base_url_env_var_name = "GOOGLE_GEMINI_BASE_URL";

/// What to set for an ask to work, for a message telling someone to set it.
pub fn envHint(comptime provider_env_var_name: []const u8) []const u8 {
    return TypesafeClient.env_var_name ++ ", or " ++ provider.envVarName(.gemini) ++ " with " ++ provider_env_var_name ++ "=gemini";
}

pub const Options = struct {
    retry_policy: retry.RetryPolicy = .{},
    /// Worth setting when the caller is waiting on the ask: see
    /// `InitOptions.request_timeout_ms` on either client.
    request_timeout_ms: ?u32 = null,
    /// Borrowed `std.http.Client` for every client the pool makes, so they
    /// share one connection pool; see `http.Transport.Options.http_client`,
    /// including what sharing it across threads asks of its allocator. Null
    /// gives each client its own.
    http_client: ?*std.http.Client = null,
    /// A variable naming the provider to use, "gemini" or "typesafe". Unset
    /// (or null), TypeSafe answers: Gemini is only ever asked when named here.
    provider_env_var_name: ?[]const u8 = null,
};

pub const AskOptions = struct {
    /// Null asks the provider's default model.
    model: ?[]const u8 = null,
};

/// An idle client and copies of what it borrows.
const Pooled = struct {
    client: union(Provider) {
        gemini: *GeminiClient,
        typesafe: *TypesafeClient,
    },
    api_key: []const u8,
    base_url: ?[]const u8,

    fn matches(self: Pooled, credentials: Credentials) bool {
        if (std.meta.activeTag(self.client) != credentials.provider) return false;
        if (!std.mem.eql(u8, self.api_key, credentials.api_key)) return false;
        const a = self.base_url orelse return credentials.base_url == null;
        const b = credentials.base_url orelse return false;
        return std.mem.eql(u8, a, b);
    }
};

pub const Credentials = struct {
    provider: Provider,
    api_key: []const u8,
    /// Null is the provider's default.
    base_url: ?[]const u8,
};

pub const CredentialsError = error{
    MissingApiKey,
    /// The provider variable names neither provider.
    UnknownProvider,
};

/// The provider an ask would go to, and what it would authenticate with.
pub fn envCredentials(environ: std.process.Environ, provider_env_var_name: ?[]const u8) CredentialsError!Credentials {
    const forced: ?Provider = if (provider_env_var_name) |name| blk: {
        const value = provider.envGet(environ, name) orelse break :blk null;
        break :blk std.meta.stringToEnum(Provider, value) orelse return error.UnknownProvider;
    } else null;

    if (forced == .gemini) {
        const gemini = provider.fromEnv(environ, .gemini);
        const key = gemini.key orelse return error.MissingApiKey;
        return .{
            .provider = .gemini,
            .api_key = key,
            .base_url = gemini.base_url orelse provider.envGet(environ, gemini_base_url_env_var_name),
        };
    }
    const key = TypesafeClient.envApiKey(environ) orelse return error.MissingApiKey;
    return .{ .provider = .typesafe, .api_key = key, .base_url = TypesafeClient.envBaseUrl(environ) };
}

/// The provider an ask would go to, or null when it would fail for want of a
/// key.
pub fn envProvider(environ: std.process.Environ, provider_env_var_name: ?[]const u8) ?Provider {
    const credentials = envCredentials(environ, provider_env_var_name) catch return null;
    return credentials.provider;
}

pub fn init(io: std.Io, allocator: std.mem.Allocator, environ: *const fn () std.process.Environ, options: Options) Pool {
    return .{ .io = io, .allocator = allocator, .environ = environ, .options = options };
}

pub fn deinit(self: *Pool) void {
    self.mutex.lockUncancelable(self.io);
    defer self.mutex.unlock(self.io);
    for (self.idle.items) |pooled| self.destroy(pooled);
    self.idle.deinit(self.allocator);
}

/// One ask, on a client of its own. On `error.ApiError`, `failure` carries the
/// API's own message, duped into `arena` so it outlives the client going back
/// in the pool; on `error.InputBlocked`, the reason the input was blocked.
///
/// Caller owns the returned `Response` and must call `deinit()`.
pub fn ask(
    self: *Pool,
    arena: std.mem.Allocator,
    state: types.Content,
    questions: types.Questions,
    options: AskOptions,
    failure: *http.ErrorDetail,
) !types.Response {
    const pooled = try self.take();
    defer self.put(pooled);

    switch (pooled.client) {
        .typesafe => |client| {
            const model = options.model orelse Provider.typesafe.defaultModel();
            const request = try typesafe_adapter.toTypesafe(arena, state, questions);
            var response = client.ask(request.state, request.questions, .{ .model = model }) catch |err| {
                if (err == error.ApiError) failure.* = try client.transport.last_error.dupe(arena);
                return err;
            };
            errdefer response.deinit();
            return typesafe_adapter.fromTypesafe(&response);
        },
        .gemini => |client| {
            const model = options.model orelse Provider.gemini.defaultModel();
            const request = try gemini_adapter.toGemini(arena, state, questions);
            var response = client.createDecision(model, request.input, request.questions) catch |err| {
                if (err == error.ApiError) failure.* = try client.transport.last_error.dupe(arena);
                return err;
            };
            errdefer response.deinit();
            return gemini_adapter.fromGemini(arena, &response, model, failure);
        },
    }
}

fn take(self: *Pool) !Pooled {
    const credentials = try envCredentials(self.environ(), self.options.provider_env_var_name);

    {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        while (self.idle.pop()) |pooled| {
            if (pooled.matches(credentials)) return pooled;
            self.destroy(pooled);
        }
    }

    const key_copy = try self.allocator.dupe(u8, credentials.api_key);
    errdefer self.allocator.free(key_copy);
    const url_copy: ?[]const u8 = if (credentials.base_url) |url| try self.allocator.dupe(u8, url) else null;
    errdefer if (url_copy) |url| self.allocator.free(url);

    return .{
        .api_key = key_copy,
        .base_url = url_copy,
        .client = switch (credentials.provider) {
            .gemini => .{ .gemini = try self.create(GeminiClient, key_copy, .{
                .base_url = url_copy,
                .retry_policy = self.options.retry_policy,
                .request_timeout_ms = self.options.request_timeout_ms,
                .http_client = self.options.http_client,
            }) },
            .typesafe => .{ .typesafe = try self.create(TypesafeClient, key_copy, .{
                .base_url = url_copy.?,
                .retry_policy = self.options.retry_policy,
                .request_timeout_ms = self.options.request_timeout_ms,
                .http_client = self.options.http_client,
            }) },
        },
    };
}

fn create(self: *Pool, comptime Client: type, api_key: []const u8, options: Client.InitOptions) !*Client {
    const client = try self.allocator.create(Client);
    client.* = .init(self.io, self.allocator, api_key, options);
    return client;
}

fn put(self: *Pool, pooled: Pooled) void {
    self.mutex.lockUncancelable(self.io);
    defer self.mutex.unlock(self.io);
    self.idle.append(self.allocator, pooled) catch self.destroy(pooled);
}

fn destroy(self: *Pool, pooled: Pooled) void {
    switch (pooled.client) {
        inline else => |client| {
            client.deinit();
            self.allocator.destroy(client);
        },
    }
    self.allocator.free(pooled.api_key);
    if (pooled.base_url) |url| self.allocator.free(url);
}

/// An environment of our own, so a test can change what the pool reads without
/// libc's `setenv` — zenai's tests do not link libc. Windows reads the
/// environment from a global rather than a block a caller can hand over, so
/// the tests that supply one of their own skip there.
const can_fake_environ = @hasField(std.process.Environ.Block, "slice");

fn fakeEnviron(comptime pairs: anytype) std.process.Environ {
    return if (comptime can_fake_environ) .{ .block = .{ .slice = pairs } } else .empty;
}

fn firstKey() std.process.Environ {
    return fakeEnviron(&[_:null]?[*:0]const u8{"TYPESAFE_API_KEY=first-key"});
}

fn secondKey() std.process.Environ {
    return fakeEnviron(&[_:null]?[*:0]const u8{"TYPESAFE_API_KEY=second-key"});
}

fn firstKeyAndBaseUrl() std.process.Environ {
    return fakeEnviron(&[_:null]?[*:0]const u8{ "TYPESAFE_API_KEY=first-key", "TYPESAFE_BASE_URL=http://127.0.0.1:9999" });
}

fn bothKeys() std.process.Environ {
    return fakeEnviron(&[_:null]?[*:0]const u8{ "TYPESAFE_API_KEY=ts-key", "GEMINI_API_KEY=gemini-key" });
}

fn bothKeysGeminiForced() std.process.Environ {
    return fakeEnviron(&[_:null]?[*:0]const u8{ "TYPESAFE_API_KEY=ts-key", "GEMINI_API_KEY=gemini-key", "DECISION_PROVIDER=gemini" });
}

fn geminiKeyOnly() std.process.Environ {
    return fakeEnviron(&[_:null]?[*:0]const u8{"GEMINI_API_KEY=gemini-key"});
}

fn geminiForcedWithoutKey() std.process.Environ {
    return fakeEnviron(&[_:null]?[*:0]const u8{ "TYPESAFE_API_KEY=ts-key", "DECISION_PROVIDER=gemini" });
}

fn unknownProvider() std.process.Environ {
    return fakeEnviron(&[_:null]?[*:0]const u8{ "TYPESAFE_API_KEY=ts-key", "DECISION_PROVIDER=jev" });
}

fn geminiOnVertex() std.process.Environ {
    return fakeEnviron(&[_:null]?[*:0]const u8{ "GOOGLE_API_KEY=vertex-key", "GOOGLE_GENAI_USE_VERTEXAI=1", "DECISION_PROVIDER=gemini" });
}

fn noKey() std.process.Environ {
    return .empty;
}

test "decisions.Pool: a client is reused, and retired when its credentials change" {
    if (comptime !can_fake_environ) return error.SkipZigTest;

    var pool: Pool = .init(std.testing.io, std.testing.allocator, firstKey, .{});
    defer pool.deinit();

    const first = try pool.take();
    try std.testing.expectEqualStrings("first-key", first.client.typesafe.api_key);
    try std.testing.expectEqualStrings(TypesafeClient.default_base_url, first.client.typesafe.base_url);
    pool.put(first);
    try std.testing.expectEqual(1, pool.idle.items.len);

    // Same environment, so the warm client comes back rather than a new one.
    const again = try pool.take();
    try std.testing.expectEqual(first.client.typesafe, again.client.typesafe);
    pool.put(again);

    // A client borrows its key, so one built from an environment that has
    // since changed cannot be handed out again.
    pool.environ = secondKey;
    const replaced = try pool.take();
    try std.testing.expect(replaced.client.typesafe != first.client.typesafe);
    try std.testing.expectEqualStrings("second-key", replaced.client.typesafe.api_key);
    pool.put(replaced);
    try std.testing.expectEqual(1, pool.idle.items.len);
}

test "decisions.Pool: a changed base URL retires a client too" {
    if (comptime !can_fake_environ) return error.SkipZigTest;

    var pool: Pool = .init(std.testing.io, std.testing.allocator, firstKey, .{});
    defer pool.deinit();

    const first = try pool.take();
    pool.put(first);
    pool.environ = firstKeyAndBaseUrl;
    const second = try pool.take();
    try std.testing.expect(second.client.typesafe != first.client.typesafe);
    try std.testing.expectEqualStrings("http://127.0.0.1:9999", second.client.typesafe.base_url);
    pool.put(second);
}

test "decisions.Pool: TypeSafe answers unless the provider variable names Gemini" {
    if (comptime !can_fake_environ) return error.SkipZigTest;

    var pool: Pool = .init(std.testing.io, std.testing.allocator, bothKeys, .{ .provider_env_var_name = "DECISION_PROVIDER" });
    defer pool.deinit();

    const typesafe = try pool.take();
    try std.testing.expectEqualStrings("ts-key", typesafe.client.typesafe.api_key);
    pool.put(typesafe);

    // Switching provider retires the idle TypeSafe client.
    pool.environ = bothKeysGeminiForced;
    const gemini = try pool.take();
    try std.testing.expectEqualStrings("gemini-key", gemini.client.gemini.api_key);
    pool.put(gemini);
    try std.testing.expectEqual(1, pool.idle.items.len);
}

test "decisions.envCredentials: a forced provider needs its own key" {
    if (comptime !can_fake_environ) return error.SkipZigTest;

    try std.testing.expectError(error.MissingApiKey, envCredentials(geminiForcedWithoutKey(), "DECISION_PROVIDER"));
    try std.testing.expectError(error.UnknownProvider, envCredentials(unknownProvider(), "DECISION_PROVIDER"));
    // Not read unless the caller names it.
    try std.testing.expectEqual(Provider.typesafe, envProvider(unknownProvider(), null).?);
    // A Google key alone is not a choice of Gemini.
    try std.testing.expectEqual(@as(?Provider, null), envProvider(geminiKeyOnly(), "DECISION_PROVIDER"));
    try std.testing.expectEqual(Provider.typesafe, envProvider(bothKeysGeminiForced(), null).?);
    try std.testing.expectEqual(Provider.typesafe, envProvider(bothKeys(), "DECISION_PROVIDER").?);
    // The Decisions API is Developer-API-only, so a Vertex key is no key.
    try std.testing.expectError(error.MissingApiKey, envCredentials(geminiOnVertex(), "DECISION_PROVIDER"));
}

test "decisions.Pool: no key is an error, not a client that cannot ask" {
    var pool: Pool = .init(std.testing.io, std.testing.allocator, noKey, .{});
    defer pool.deinit();
    try std.testing.expectError(error.MissingApiKey, pool.take());
}
