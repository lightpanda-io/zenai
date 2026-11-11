//! Tavily Search API client. https://docs.tavily.com/
//!
//! Tavily is an AI-friendly search API: a query goes in, a clean JSON list
//! of `{title, url, content}` results comes out (plus an optional synthesized
//! `answer`). Designed as a low-noise alternative to scraping a SERP.

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
bearer: http.BearerAuth = .{},

/// The environment variable `envApiKey` reads.
pub const env_var_name = "TAVILY_API_KEY";

pub fn envApiKey(environ: std.process.Environ) ?[:0]const u8 {
    return env.get(environ, env_var_name);
}

pub const InitOptions = struct {
    base_url: []const u8 = "https://api.tavily.com",
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
            .log_tag = "Tavily",
        }),
        .api_key = api_key,
        .base_url = options.base_url,
    };
}

pub fn deinit(self: *Client) void {
    self.bearer.deinit(self.transport.allocator);
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

    const extra_headers = [_]std.http.Header{try self.bearer.header(self.transport.allocator, self.api_key)};

    return self.transport.postJson(url, &extra_headers, request, SearchResponse);
}

test "search rejects empty api key" {
    var client = init(std.testing.io, std.testing.allocator, "", .{});
    defer client.deinit();
    try std.testing.expectError(error.MissingApiKey, client.search("anything", .{}));
}
