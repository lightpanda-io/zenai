//! Keenable Search API client. https://docs.keenable.ai/
//!
//! Keenable is an AI-friendly search API: a query goes in, a clean JSON list
//! of `{title, url, snippet}` results comes out. Unlike the other search
//! providers here it also works without an API key: with an empty `api_key`
//! the client calls the public endpoint (rate-limited per client IP), which
//! requires an application title header for attribution. A key lifts the
//! rate limits; it is not a prerequisite.

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
app_title: []const u8,
http_client: std.http.Client,
retry_policy: RetryPolicy,
last_error: http.ErrorDetail = .{},

pub const InitOptions = struct {
    base_url: []const u8 = "https://api.keenable.ai",
    /// Sent as `X-Keenable-Title`. Required by the keyless endpoint (it
    /// rejects requests without one); pure attribution on keyed calls.
    app_title: []const u8 = "zenai",
    retry_policy: RetryPolicy = .{},
};

pub fn init(io: std.Io, allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .app_title = options.app_title,
        .http_client = .{ .allocator = allocator, .io = io },
        .retry_policy = options.retry_policy,
    };
}

pub fn deinit(self: *Client) void {
    self.http_client.deinit();
    self.last_error.deinit(self.allocator);
}

pub const Response = http.Response;

/// No `MissingApiKey`: an empty key routes to the public endpoint instead.
pub const ApiError = http.FetchError;

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error.set(self.allocator, status_code, body);
}

/// `/v1/search` keyed, `/v1/search/public` keyless.
fn searchPath(api_key: []const u8) []const u8 {
    return if (api_key.len == 0) "/v1/search/public" else "/v1/search";
}

/// Run a search. Caller owns the returned `Response` and must call `deinit()`.
pub fn search(
    self: *Client,
    query: []const u8,
    options: SearchOptions,
) ApiError!Response(SearchResponse) {
    const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.base_url, searchPath(self.api_key) });
    defer self.allocator.free(url);

    var request = options;
    request.query = query;
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(request, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    var headers_buf: [2]std.http.Header = undefined;
    var n: usize = 0;
    headers_buf[n] = .{ .name = "X-Keenable-Title", .value = self.app_title };
    n += 1;
    if (self.api_key.len > 0) {
        headers_buf[n] = .{ .name = "X-API-Key", .value = self.api_key };
        n += 1;
    }

    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, .{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload_buf.written(),
        .extra_headers = headers_buf[0..n],
        .headers = .{ .content_type = .{ .override = "application/json" } },
    }, SearchResponse, self);
}

test "empty api key routes to the public endpoint" {
    try std.testing.expectEqualStrings("/v1/search/public", searchPath(""));
    try std.testing.expectEqualStrings("/v1/search", searchPath("sk-abc"));
}
