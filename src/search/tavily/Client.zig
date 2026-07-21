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
const SearchRequest = types.SearchRequest;
const SearchResponse = types.SearchResponse;

const Client = @This();

allocator: std.mem.Allocator,
api_key: []const u8,
base_url: []const u8,
http_client: std.http.Client,
retry_policy: RetryPolicy,
last_error_status: ?u10 = null,
last_error_body: ?[]const u8 = null,

pub const InitOptions = struct {
    base_url: []const u8 = "https://api.tavily.com",
    retry_policy: RetryPolicy = .{},
};

pub fn init(allocator: std.mem.Allocator, io: std.Io, api_key: []const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .http_client = .{ .allocator = allocator, .io = io },
        .retry_policy = options.retry_policy,
    };
}

pub fn deinit(self: *Client) void {
    self.http_client.deinit();
    if (self.last_error_body) |b| self.allocator.free(b);
}

pub const Response = http.Response;

pub const ApiError = error{
    ApiError,
    MissingApiKey,
    EmptyResponse,
} || std.http.Client.FetchError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error_status = status_code;
    if (self.last_error_body) |old| self.allocator.free(old);
    self.last_error_body = if (body.len == 0) null else self.allocator.dupe(u8, body) catch null;
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

    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(SearchRequest{
        .query = query,
        .max_results = options.max_results,
        .search_depth = options.search_depth,
        .topic = options.topic,
        .include_answer = options.include_answer,
        .include_raw_content = options.include_raw_content,
        .time_range = options.time_range,
        .include_domains = options.include_domains,
        .exclude_domains = options.exclude_domains,
    }, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    const auth = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_key});
    defer self.allocator.free(auth);
    const extra_headers = [_]std.http.Header{
        .{ .name = "Authorization", .value = auth },
    };

    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, .{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload_buf.written(),
        .extra_headers = &extra_headers,
        .headers = .{ .content_type = .{ .override = "application/json" } },
    }, SearchResponse, self);
}

test "search rejects empty api key" {
    var client = init(std.testing.allocator, std.testing.io, "", .{});
    defer client.deinit();
    try std.testing.expectError(error.MissingApiKey, client.search("anything", .{}));
}
