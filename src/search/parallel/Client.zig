//! Parallel Search API client. https://docs.parallel.ai/search/search-quickstart
//!
//! A search API built for agents: each result carries markdown excerpts
//! picked for the request's `objective` and keyword `search_queries`, so no
//! separate contents call is needed.

const std = @import("std");
const types = @import("types.zig");
const http = @import("../../http.zig");
const retry = @import("../../retry.zig");

pub const RetryPolicy = retry.RetryPolicy;

const SearchOptions = types.SearchOptions;
const SearchResponse = types.SearchResponse;

const Client = @This();

allocator: std.mem.Allocator,
api_key: []const u8,
base_url: []const u8,
http_client: std.http.Client,
retry_policy: RetryPolicy,
request_timeout_ms: ?u32,
last_error: http.ErrorDetail = .{},

pub const InitOptions = struct {
    base_url: []const u8 = "https://api.parallel.ai",
    retry_policy: RetryPolicy = .{},
    /// Per-attempt wall-clock bound on non-streaming requests, from an
    /// established connection to the end of the body (connect excluded);
    /// exceeding it fails with `error.Timeout`. `null` waits indefinitely.
    request_timeout_ms: ?u32 = null,
};

pub fn init(io: std.Io, allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .http_client = .{ .allocator = allocator, .io = http.addrConfigIo(io) },
        .retry_policy = options.retry_policy,
        .request_timeout_ms = options.request_timeout_ms,
    };
}

pub fn deinit(self: *Client) void {
    self.http_client.deinit();
    self.last_error.deinit(self.allocator);
}

pub const Response = http.Response;

pub const ApiError = error{MissingApiKey} || http.FetchError;

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error.setLogged(self.allocator, status_code, body, "Parallel");
}

/// Run a search. `query` becomes the single keyword query unless
/// `options.search_queries` is set, in which case those are sent instead.
/// Caller owns the returned `Response` and must call `deinit()`.
pub fn search(
    self: *Client,
    query: []const u8,
    options: SearchOptions,
) ApiError!Response(SearchResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;

    const url = try self.allocator.print("{s}/v1/search", .{self.base_url});
    defer self.allocator.free(url);

    const queries = [_][]const u8{query};
    var request = options;
    if (request.search_queries.len == 0) request.search_queries = &queries;

    const extra_headers = [_]std.http.Header{
        .{ .name = "x-api-key", .value = self.api_key },
    };

    return http.postJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, self.request_timeout_ms, url, &extra_headers, request, SearchResponse, self);
}

test "search rejects empty api key" {
    var client = init(std.testing.io, std.testing.allocator, "", .{});
    defer client.deinit();
    try std.testing.expectError(error.MissingApiKey, client.search("anything", .{}));
}
