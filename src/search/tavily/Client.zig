//! Tavily Search API client. https://docs.tavily.com/
//!
//! Tavily is an AI-friendly search API: a query goes in, a clean JSON list
//! of `{title, url, content}` results comes out (plus an optional synthesized
//! `answer`). Designed as a low-noise alternative to scraping a SERP.

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
    base_url: []const u8 = "https://api.tavily.com",
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
        .http_client = .{ .allocator = allocator, .io = io },
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
    self.last_error.set(self.allocator, status_code, body);
}

/// Run a search. Caller owns the returned `Response` and must call `deinit()`.
pub fn search(
    self: *Client,
    query: []const u8,
    options: SearchOptions,
) ApiError!Response(SearchResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;

    const url = try std.fmt.allocPrint(self.allocator, "{s}/search", .{self.base_url});
    defer self.allocator.free(url);

    var request = options;
    request.query = query;
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(request, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    const auth = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_key});
    defer self.allocator.free(auth);
    const extra_headers = [_]std.http.Header{
        .{ .name = "Authorization", .value = auth },
    };

    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, self.request_timeout_ms, .{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload_buf.written(),
        .extra_headers = &extra_headers,
        .headers = .{ .content_type = .{ .override = "application/json" } },
    }, SearchResponse, self);
}

test "search rejects empty api key" {
    var client = init(std.testing.io, std.testing.allocator, "", .{});
    defer client.deinit();
    try std.testing.expectError(error.MissingApiKey, client.search("anything", .{}));
}
