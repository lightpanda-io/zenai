//! Keenable Search API client. https://docs.keenable.ai/
//!
//! Keenable is an AI-friendly search API: a query goes in, a clean JSON list
//! of `{title, url, snippet}` results comes out. Unlike the other search
//! providers here it also works without an API key: pass `null` as `api_key`
//! and the client calls the public endpoint (rate-limited per client IP),
//! which requires an application title header for attribution. A key lifts
//! the rate limits; it is not a prerequisite.

const std = @import("std");
const types = @import("types.zig");
const http = @import("../../http.zig");
const retry = @import("../../retry.zig");

pub const RetryPolicy = retry.RetryPolicy;

const SearchOptions = types.SearchOptions;
const SearchResponse = types.SearchResponse;

const Client = @This();

allocator: std.mem.Allocator,
/// `null` selects the keyless public endpoint. A non-null key — even an
/// empty one, e.g. from a set-but-empty environment variable — selects the
/// keyed endpoint, so a misconfigured key fails loudly instead of silently
/// changing endpoint and rate-limit regime.
api_key: ?[]const u8,
base_url: []const u8,
app_title: []const u8,
http_client: std.http.Client,
retry_policy: RetryPolicy,
request_timeout_ms: ?u32,
last_error: http.ErrorDetail = .{},

pub const InitOptions = struct {
    base_url: []const u8 = "https://api.keenable.ai",
    /// Sent as `X-Keenable-Title`. Required by the keyless endpoint (it
    /// rejects requests without one); pure attribution on keyed calls.
    /// Deliberately no default: only the embedding application knows its
    /// own name, and a library default would self-attribute every embedder
    /// that forgets to set it.
    app_title: []const u8,
    retry_policy: RetryPolicy = .{},
    /// Per-attempt wall-clock bound on non-streaming requests, from an
    /// established connection to the end of the body (connect excluded);
    /// exceeding it fails with `error.Timeout`. `null` waits indefinitely.
    request_timeout_ms: ?u32 = null,
};

pub fn init(io: std.Io, allocator: std.mem.Allocator, api_key: ?[]const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .app_title = options.app_title,
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

/// No `MissingApiKey` — a `null` key is the supported keyless mode. What is
/// required instead is the attribution title, checked before any request.
pub const ApiError = error{MissingAppTitle} || http.FetchError;

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error.set(self.allocator, status_code, body);
}

/// `/v1/search` keyed, `/v1/search/public` keyless.
fn searchPath(api_key: ?[]const u8) []const u8 {
    return if (api_key == null) "/v1/search/public" else "/v1/search";
}

/// Run a search. Caller owns the returned `Response` and must call `deinit()`.
pub fn search(
    self: *Client,
    query: []const u8,
    options: SearchOptions,
) ApiError!Response(SearchResponse) {
    if (self.app_title.len == 0) return error.MissingAppTitle;

    const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.base_url, searchPath(self.api_key) });
    defer self.allocator.free(url);

    var request = options;
    request.query = query;
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(request, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    const headers = [_]std.http.Header{
        .{ .name = "X-Keenable-Title", .value = self.app_title },
        .{ .name = "X-API-Key", .value = self.api_key orelse "" },
    };

    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, self.request_timeout_ms, .{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload_buf.written(),
        .extra_headers = headers[0..if (self.api_key == null) 1 else 2],
        .headers = .{ .content_type = .{ .override = "application/json" } },
    }, SearchResponse, self);
}

test "null api key routes to the public endpoint" {
    try std.testing.expectEqualStrings("/v1/search/public", searchPath(null));
    try std.testing.expectEqualStrings("/v1/search", searchPath("sk-abc"));
    // Set-but-empty is keyed on purpose: it fails loudly server-side rather
    // than silently downgrading to the keyless rate-limit regime.
    try std.testing.expectEqualStrings("/v1/search", searchPath(""));
}

test "search rejects an empty app title" {
    var client = init(std.testing.io, std.testing.allocator, null, .{ .app_title = "" });
    defer client.deinit();
    try std.testing.expectError(error.MissingAppTitle, client.search("anything", .{}));
}
