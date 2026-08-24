//! Keenable Search API request/response shapes. https://docs.keenable.ai/

const std = @import("std");

pub const Result = struct {
    title: []const u8 = "",
    url: []const u8 = "",
    /// The page-text excerpt — this is the field that carries the result's
    /// text. The wire format also has `description` (the page's meta
    /// description), which comes back empty on essentially every result; it
    /// is deliberately not mapped, so nothing can bind to it by mistake —
    /// parsing ignores unknown fields.
    snippet: []const u8 = "",
    published_at: ?[]const u8 = null,
    acquired_at: ?[]const u8 = null,
};

pub const SearchResponse = struct {
    query: []const u8 = "",
    mode: ?[]const u8 = null,
    results: []const Result = &.{},
};

/// Doubles as the request body: `Client.search` fills in `query`.
pub const SearchOptions = struct {
    query: []const u8 = "",
    /// 1–50 (provider default 10).
    max_results: ?u8 = null,
    /// Excerpt budget per result, in characters. The API treats it as a
    /// hint and may round up to a word boundary — enforce locally when the
    /// budget is a hard cap.
    snippet_max_length: ?u16 = null,
    /// Restrict results to one site, e.g. "ziglang.org".
    site: ?[]const u8 = null,
    /// ISO dates (YYYY-MM-DD).
    published_after: ?[]const u8 = null,
    published_before: ?[]const u8 = null,
};

test "SearchResponse parses Keenable fixture" {
    // Real response shape: `description` present but empty, text in
    // `snippet`. The fixture keeps `description` to prove the unmapped
    // field is ignored, not tripped over.
    const fixture =
        \\{
        \\  "query": "zig systems programming language",
        \\  "mode": "pro",
        \\  "results": [
        \\    {"title": "Zig (programming language)", "url": "https://en.wikipedia.org/wiki/Zig_(programming_language)", "description": "", "snippet": "Zig is a system programming language.", "published_at": "2026-08-15T09:22:57Z", "acquired_at": "2026-08-21T05:39:17Z"},
        \\    {"title": "Zig guide", "url": "https://example.org/zig", "description": "", "snippet": "Compile-time execution."}
        \\  ]
        \\}
    ;
    const parsed = try std.json.parseFromSlice(SearchResponse, std.testing.allocator, fixture, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqualStrings("zig systems programming language", parsed.value.query);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.results.len);
    try std.testing.expectEqualStrings("Zig (programming language)", parsed.value.results[0].title);
    try std.testing.expectEqualStrings("Zig is a system programming language.", parsed.value.results[0].snippet);
}

test "SearchOptions stringifies with no null fields" {
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(SearchOptions{
        .query = "zig",
        .max_results = 10,
        .snippet_max_length = 500,
    }, .{ .emit_null_optional_fields = false }, &buf.writer);
    try std.testing.expectEqualStrings(
        \\{"query":"zig","max_results":10,"snippet_max_length":500}
    , buf.written());
}
