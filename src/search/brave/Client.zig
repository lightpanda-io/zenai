//! Brave Web Search API client.
//! https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
//!
//! An independent-index web search API: a GET with the query in the URL comes
//! back as JSON sections (`web.results`, `news.results`, …) of
//! `{title, url, description}` hits. Descriptions carry `<strong>` highlight
//! markup unless `text_decorations` is set to false.

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
last_error_status: ?u10 = null,
last_error_body: ?[]const u8 = null,

pub const InitOptions = struct {
    base_url: []const u8 = "https://api.search.brave.com",
    retry_policy: RetryPolicy = .{},
};

pub fn init(io: std.Io, allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
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

    const url = try buildSearchUrl(self.allocator, self.base_url, query, options);
    defer self.allocator.free(url);

    const extra_headers = [_]std.http.Header{
        .{ .name = "X-Subscription-Token", .value = self.api_key },
        .{ .name = "Accept", .value = "application/json" },
    };

    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, .{
        .location = .{ .url = url },
        .method = .GET,
        .extra_headers = &extra_headers,
    }, SearchResponse, self);
}

fn buildSearchUrl(
    allocator: std.mem.Allocator,
    base_url: []const u8,
    query: []const u8,
    options: SearchOptions,
) std.mem.Allocator.Error![]u8 {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    errdefer buf.deinit();
    writeSearchUrl(&buf.writer, base_url, query, options) catch return error.OutOfMemory;
    return buf.toOwnedSlice();
}

fn writeSearchUrl(
    w: *std.Io.Writer,
    base_url: []const u8,
    query: []const u8,
    options: SearchOptions,
) std.Io.Writer.Error!void {
    try w.print("{s}/res/v1/web/search?q=", .{base_url});
    try percentEncode(w, query);
    if (options.count) |v| try w.print("&count={d}", .{v});
    if (options.offset) |v| try w.print("&offset={d}", .{v});
    if (options.country) |v| {
        try w.writeAll("&country=");
        try percentEncode(w, v);
    }
    if (options.search_lang) |v| {
        try w.writeAll("&search_lang=");
        try percentEncode(w, v);
    }
    if (options.safesearch) |v| try w.print("&safesearch={s}", .{@tagName(v)});
    if (options.freshness) |v| {
        try w.writeAll("&freshness=");
        try percentEncode(w, v);
    }
    if (options.result_filter) |v| {
        try w.writeAll("&result_filter=");
        try percentEncode(w, v);
    }
    if (options.extra_snippets) |v| try w.print("&extra_snippets={s}", .{boolStr(v)});
    if (options.text_decorations) |v| try w.print("&text_decorations={s}", .{boolStr(v)});
    if (options.spellcheck) |v| try w.print("&spellcheck={s}", .{boolStr(v)});
}

fn boolStr(v: bool) []const u8 {
    return if (v) "true" else "false";
}

// Not `formatQuery`: the RFC 3986 query production passes `&`/`=`/`+` through
// raw, which would corrupt parameter framing.
fn percentEncode(w: *std.Io.Writer, raw: []const u8) std.Io.Writer.Error!void {
    try std.Uri.Component.percentEncode(w, raw, isUnreserved);
}

fn isUnreserved(c: u8) bool {
    return switch (c) {
        'A'...'Z', 'a'...'z', '0'...'9', '-', '.', '_', '~' => true,
        else => false,
    };
}

test "search rejects empty api key" {
    var client = init(std.testing.io, std.testing.allocator, "", .{});
    defer client.deinit();
    try std.testing.expectError(error.MissingApiKey, client.search("anything", .{}));
}

test "buildSearchUrl percent-encodes the query" {
    const url = try buildSearchUrl(std.testing.allocator, "https://api.search.brave.com", "fish & chips = +1?", .{});
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings(
        "https://api.search.brave.com/res/v1/web/search?q=fish%20%26%20chips%20%3D%20%2B1%3F",
        url,
    );
}

test "buildSearchUrl appends set options" {
    const url = try buildSearchUrl(std.testing.allocator, "http://localhost", "zig", .{
        .count = 10,
        .country = "US",
        .safesearch = .off,
        .result_filter = "web,news",
        .text_decorations = false,
    });
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings(
        "http://localhost/res/v1/web/search?q=zig&count=10&country=US&safesearch=off&result_filter=web%2Cnews&text_decorations=false",
        url,
    );
}
