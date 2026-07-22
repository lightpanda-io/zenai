//! Brave Web Search API request/response shapes.
//! https://api-dashboard.search.brave.com/app/documentation/web-search/get-started

const std = @import("std");

pub const SafeSearch = enum {
    off,
    moderate,
    strict,
};

/// One search hit — the same shape serves `web.results` and `news.results`.
pub const Result = struct {
    title: []const u8 = "",
    url: []const u8 = "",
    description: []const u8 = "",
    age: ?[]const u8 = null,
    extra_snippets: ?[]const []const u8 = null,
};

pub const WebResults = struct {
    results: []const Result = &.{},
};

pub const NewsResults = struct {
    results: []const Result = &.{},
};

pub const QueryInfo = struct {
    original: []const u8 = "",
    more_results_available: ?bool = null,
};

pub const SearchResponse = struct {
    query: ?QueryInfo = null,
    web: ?WebResults = null,
    news: ?NewsResults = null,
};

pub const SearchOptions = struct {
    /// Results per page, max 20 (provider default 20).
    count: ?u8 = null,
    /// Zero-based page offset, max 9.
    offset: ?u8 = null,
    /// Two-letter country code, e.g. "US".
    country: ?[]const u8 = null,
    /// ISO 639-1 language code for result content.
    search_lang: ?[]const u8 = null,
    safesearch: ?SafeSearch = null,
    /// `pd`/`pw`/`pm`/`py`, or a `YYYY-MM-DDtoYYYY-MM-DD` range.
    freshness: ?[]const u8 = null,
    /// Comma-separated result sections to include, e.g. "web,news".
    result_filter: ?[]const u8 = null,
    /// Up to 5 additional excerpts per result.
    extra_snippets: ?bool = null,
    /// Highlight markup (`<strong>`) in descriptions; set false for plain text.
    text_decorations: ?bool = null,
    spellcheck: ?bool = null,
};

test "SearchResponse parses Brave fixture" {
    const fixture =
        \\{
        \\  "type": "search",
        \\  "query": {"original": "capital of france", "more_results_available": true, "spellcheck_off": false},
        \\  "mixed": {"type": "mixed", "main": []},
        \\  "web": {
        \\    "type": "search",
        \\    "results": [
        \\      {"title": "Paris - Wikipedia", "url": "https://en.wikipedia.org/wiki/Paris", "description": "Paris is the capital.", "age": "January 2, 2026", "profile": {"name": "Wikipedia"}},
        \\      {"title": "France", "url": "https://example.org/fr", "description": "Country.", "extra_snippets": ["Paris is its capital."]}
        \\    ]
        \\  }
        \\}
    ;
    const parsed = try std.json.parseFromSlice(SearchResponse, std.testing.allocator, fixture, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqualStrings("capital of france", parsed.value.query.?.original);
    try std.testing.expectEqual(true, parsed.value.query.?.more_results_available.?);
    const results = parsed.value.web.?.results;
    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("Paris - Wikipedia", results[0].title);
    try std.testing.expectEqualStrings("January 2, 2026", results[0].age.?);
    try std.testing.expectEqualStrings("Paris is its capital.", results[1].extra_snippets.?[0]);
    try std.testing.expect(parsed.value.news == null);
}
