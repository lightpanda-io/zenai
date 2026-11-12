//! Exa Search API client. https://exa.ai/docs/reference/search
//!
//! An embeddings-based search API aimed at agents. Snippet text is opt-in:
//! a bare search returns metadata only — request `contents`
//! (highlights/text/summary) to get any.

const std = @import("std");
const types = @import("types.zig");
const http = @import("../../http.zig");
const retry = @import("../../retry.zig");
const env = @import("../../env.zig");

pub const RetryPolicy = retry.RetryPolicy;

const SearchOptions = types.SearchOptions;
const SearchResponse = types.SearchResponse;

const Client = @This();

transport: http.Transport,
api_key: []const u8,
base_url: []const u8,

/// The environment variable `envApiKey` reads.
pub const env_var_name = "EXA_API_KEY";

pub fn envApiKey(environ: std.process.Environ) ?[:0]const u8 {
    return env.get(environ, env_var_name);
}

pub const InitOptions = struct {
    base_url: []const u8 = "https://api.exa.ai",
    retry_policy: RetryPolicy = .{},
    /// Per-attempt wall-clock bound on non-streaming requests, from an
    /// established connection to the end of the body (connect excluded);
    /// exceeding it fails with `error.Timeout`. `null` waits indefinitely.
    request_timeout_ms: ?u32 = null,
    /// Borrowed `std.http.Client` to share a connection pool with other
    /// clients; see `http.Transport.Options.http_client`. Null owns one.
    http_client: ?*std.http.Client = null,
};

pub fn init(io: std.Io, allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    return .{
        .transport = .init(io, allocator, .{
            .retry_policy = options.retry_policy,
            .request_timeout_ms = options.request_timeout_ms,
            .http_client = options.http_client,
            .log_tag = "Exa",
        }),
        .api_key = api_key,
        .base_url = options.base_url,
    };
}

pub fn deinit(self: *Client) void {
    self.transport.deinit();
}

pub const Response = http.Response;

pub const ApiError = error{MissingApiKey} || http.FetchError;

/// Run a search. Caller owns the returned `Response` and must call `deinit()`.
pub fn search(
    self: *Client,
    query: []const u8,
    options: SearchOptions,
) ApiError!Response(SearchResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;

    const url = try self.transport.allocator.print("{s}/search", .{self.base_url});
    defer self.transport.allocator.free(url);

    var request = options;
    request.query = query;

    const extra_headers = [_]std.http.Header{
        .{ .name = "x-api-key", .value = self.api_key },
    };

    return self.transport.postJson(url, &extra_headers, request, SearchResponse);
}

test "search rejects empty api key" {
    var client = init(std.testing.io, std.testing.allocator, "", .{});
    defer client.deinit();
    try std.testing.expectError(error.MissingApiKey, client.search("anything", .{}));
}
