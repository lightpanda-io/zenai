//! Exa Search API request/response shapes. https://exa.ai/docs/reference/search

const std = @import("std");

pub const SearchType = enum {
    auto,
    instant,
    fast,
    @"deep-lite",
    deep,
    @"deep-reasoning",
};

/// A bare search returns metadata only; snippet text must be requested here.
/// The API also accepts object forms (maxCharacters etc.) — not modeled.
pub const Contents = struct {
    text: ?bool = null,
    highlights: ?bool = null,
    summary: ?bool = null,
};

pub const Result = struct {
    title: ?[]const u8 = null,
    url: []const u8 = "",
    publishedDate: ?[]const u8 = null,
    author: ?[]const u8 = null,
    highlights: ?[]const []const u8 = null,
    text: ?[]const u8 = null,
    summary: ?[]const u8 = null,
    score: ?f32 = null,
};

pub const SearchResponse = struct {
    requestId: ?[]const u8 = null,
    results: []const Result = &.{},
};

/// Doubles as the request body: `Client.search` fills in `query`.
pub const SearchOptions = struct {
    query: []const u8 = "",
    type: ?SearchType = null,
    /// 1–100 (provider default 10).
    numResults: ?u8 = null,
    /// Focus area, e.g. "company", "news", "people".
    category: ?[]const u8 = null,
    includeDomains: ?[]const []const u8 = null,
    excludeDomains: ?[]const []const u8 = null,
    /// ISO 8601, e.g. "2026-01-01T00:00:00.000Z".
    startPublishedDate: ?[]const u8 = null,
    endPublishedDate: ?[]const u8 = null,
    contents: ?Contents = null,
};

test "SearchResponse parses Exa fixture" {
    const fixture =
        \\{
        \\  "requestId": "b5947044c4b78efa9552a7c89b306d95",
        \\  "resolvedSearchType": "neural",
        \\  "results": [
        \\    {"id": "https://en.wikipedia.org/wiki/Paris", "title": "Paris - Wikipedia", "url": "https://en.wikipedia.org/wiki/Paris", "publishedDate": "2026-01-02T00:00:00.000Z", "author": null, "score": 0.42, "highlights": ["Paris is the capital of France."], "highlightScores": [0.99]},
        \\    {"id": "https://example.org/fr", "title": null, "url": "https://example.org/fr"}
        \\  ],
        \\  "costDollars": {"total": 0.005}
        \\}
    ;
    const parsed = try std.json.parseFromSlice(SearchResponse, std.testing.allocator, fixture, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqualStrings("b5947044c4b78efa9552a7c89b306d95", parsed.value.requestId.?);
    const results = parsed.value.results;
    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("Paris - Wikipedia", results[0].title.?);
    try std.testing.expectEqualStrings("Paris is the capital of France.", results[0].highlights.?[0]);
    try std.testing.expect(results[1].title == null);
    try std.testing.expect(results[1].highlights == null);
}

test "SearchOptions stringifies with enum tag and no null fields" {
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(SearchOptions{
        .query = "zig",
        .type = .@"deep-lite",
        .contents = .{ .highlights = true },
    }, .{ .emit_null_optional_fields = false }, &buf.writer);
    try std.testing.expectEqualStrings(
        \\{"query":"zig","type":"deep-lite","contents":{"highlights":true}}
    , buf.written());
}
