//! Tavily Search API request/response shapes. https://docs.tavily.com/

const std = @import("std");

pub const SearchDepth = enum {
    basic,
    advanced,
};

pub const Topic = enum {
    general,
    news,
};

pub const Result = struct {
    title: []const u8 = "",
    url: []const u8 = "",
    content: []const u8 = "",
    score: ?f32 = null,
};

pub const SearchResponse = struct {
    query: []const u8 = "",
    answer: ?[]const u8 = null,
    results: []const Result = &.{},
    response_time: ?f32 = null,
};

/// Doubles as the request body: `Client.search` fills in `query`.
pub const SearchOptions = struct {
    query: []const u8 = "",
    max_results: ?u8 = null,
    search_depth: ?SearchDepth = null,
    topic: ?Topic = null,
    include_answer: ?bool = null,
    include_raw_content: ?bool = null,
    time_range: ?[]const u8 = null,
    include_domains: ?[]const []const u8 = null,
    exclude_domains: ?[]const []const u8 = null,
};

test "SearchResponse parses Tavily fixture" {
    const fixture =
        \\{
        \\  "query": "capital of france",
        \\  "answer": "Paris",
        \\  "results": [
        \\    {"title": "Paris - Wikipedia", "url": "https://en.wikipedia.org/wiki/Paris", "content": "Paris is the capital.", "score": 0.99},
        \\    {"title": "France", "url": "https://example.org/fr", "content": "Country."}
        \\  ],
        \\  "response_time": 0.42
        \\}
    ;
    const parsed = try std.json.parseFromSlice(SearchResponse, std.testing.allocator, fixture, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqualStrings("capital of france", parsed.value.query);
    try std.testing.expectEqualStrings("Paris", parsed.value.answer.?);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.results.len);
    try std.testing.expectEqualStrings("Paris - Wikipedia", parsed.value.results[0].title);
}

test "SearchOptions stringifies with enum tag and no null fields" {
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(SearchOptions{
        .query = "zig",
        .search_depth = .advanced,
    }, .{ .emit_null_optional_fields = false }, &buf.writer);
    try std.testing.expectEqualStrings(
        \\{"query":"zig","search_depth":"advanced"}
    , buf.written());
}
