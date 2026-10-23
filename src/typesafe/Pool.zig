//! A pool of idle `Client`s: an ask takes one, or makes one, and puts it back,
//! so concurrent asks each get their own and keep its connection warm.
//!
//! A `Client` borrows its API key and base URL rather than copying them, and
//! the environment they came from can change under it, so the pool keeps its
//! own copies and retires any client whose credentials no longer match.

const std = @import("std");
const Client = @import("Client.zig");
const types = @import("types.zig");
const http = @import("../http.zig");

const Pool = @This();

io: std.Io,
allocator: std.mem.Allocator,
/// Called whenever a client has to be made, rather than held as a snapshot:
/// `setenv` can move the block a `std.process.Environ` points into, and
/// exporting a key part-way through a run should take effect without a
/// restart.
environ: *const fn () std.process.Environ,
options: Options,
idle: std.ArrayList(*Client) = .empty,
mutex: std.Io.Mutex = .init,

pub const Options = struct {
    retry_policy: Client.RetryPolicy = .{},
    /// Worth setting when the caller is waiting on the ask: see
    /// `Client.InitOptions.request_timeout_ms`.
    request_timeout_ms: ?u32 = null,
};

pub fn init(io: std.Io, allocator: std.mem.Allocator, environ: *const fn () std.process.Environ, options: Options) Pool {
    return .{ .io = io, .allocator = allocator, .environ = environ, .options = options };
}

pub fn deinit(self: *Pool) void {
    self.mutex.lockUncancelable(self.io);
    defer self.mutex.unlock(self.io);
    for (self.idle.items) |client| self.destroy(client);
    self.idle.deinit(self.allocator);
}

/// One ask, on a client of its own. On `error.ApiError`, `failure` carries the
/// API's own message, duped into `arena` so it outlives the client going back
/// in the pool.
pub fn ask(
    self: *Pool,
    arena: std.mem.Allocator,
    state: types.Content,
    questions: types.Questions,
    options: types.AskOptions,
    failure: *http.ErrorDetail,
) !Client.Response(types.AskResponse) {
    const client = try self.take();
    defer self.put(client);

    return client.ask(state, questions, options) catch |err| {
        if (err == error.ApiError) failure.* = try client.last_error.dupe(arena);
        return err;
    };
}

fn take(self: *Pool) !*Client {
    const environ = self.environ();
    const api_key = Client.envApiKey(environ) orelse return error.MissingApiKey;
    const base_url = Client.envBaseUrl(environ);

    {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        while (self.idle.pop()) |client| {
            if (std.mem.eql(u8, client.api_key, api_key) and std.mem.eql(u8, client.base_url, base_url)) {
                return client;
            }
            self.destroy(client);
        }
    }

    const key_copy = try self.allocator.dupe(u8, api_key);
    errdefer self.allocator.free(key_copy);
    const url_copy = try self.allocator.dupe(u8, base_url);
    errdefer self.allocator.free(url_copy);

    const client = try self.allocator.create(Client);
    client.* = .init(self.io, self.allocator, key_copy, .{
        .base_url = url_copy,
        .retry_policy = self.options.retry_policy,
        .request_timeout_ms = self.options.request_timeout_ms,
    });
    return client;
}

fn put(self: *Pool, client: *Client) void {
    self.mutex.lockUncancelable(self.io);
    defer self.mutex.unlock(self.io);
    self.idle.append(self.allocator, client) catch self.destroy(client);
}

fn destroy(self: *Pool, client: *Client) void {
    const api_key = client.api_key;
    const base_url = client.base_url;
    client.deinit();
    self.allocator.destroy(client);
    self.allocator.free(api_key);
    self.allocator.free(base_url);
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

fn noKey() std.process.Environ {
    return .empty;
}

test "typesafe.Pool: a client is reused, and retired when its credentials change" {
    if (comptime !can_fake_environ) return error.SkipZigTest;

    var pool: Pool = .init(std.testing.io, std.testing.allocator, firstKey, .{});
    defer pool.deinit();

    const first = try pool.take();
    try std.testing.expectEqualStrings("first-key", first.api_key);
    try std.testing.expectEqualStrings(Client.default_base_url, first.base_url);
    pool.put(first);
    try std.testing.expectEqual(1, pool.idle.items.len);

    // Same environment, so the warm client comes back rather than a new one.
    const again = try pool.take();
    try std.testing.expectEqual(first, again);
    pool.put(again);

    // A client borrows its key, so one built from an environment that has
    // since changed cannot be handed out again.
    pool.environ = secondKey;
    const replaced = try pool.take();
    try std.testing.expect(replaced != first);
    try std.testing.expectEqualStrings("second-key", replaced.api_key);
    pool.put(replaced);
    try std.testing.expectEqual(1, pool.idle.items.len);
}

test "typesafe.Pool: a changed base URL retires a client too" {
    if (comptime !can_fake_environ) return error.SkipZigTest;

    var pool: Pool = .init(std.testing.io, std.testing.allocator, firstKey, .{});
    defer pool.deinit();

    const first = try pool.take();
    pool.put(first);
    pool.environ = firstKeyAndBaseUrl;
    const second = try pool.take();
    try std.testing.expect(second != first);
    try std.testing.expectEqualStrings("http://127.0.0.1:9999", second.base_url);
    pool.put(second);
}

test "typesafe.Pool: no key is an error, not a client that cannot ask" {
    var pool: Pool = .init(std.testing.io, std.testing.allocator, noKey, .{});
    defer pool.deinit();
    try std.testing.expectError(error.MissingApiKey, pool.take());
}
