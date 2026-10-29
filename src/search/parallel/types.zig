//! Parallel Search API request/response shapes. https://docs.parallel.ai/search/search-quickstart

const std = @import("std");

pub const Mode = enum {
    /// Fastest responses (p50 ~200ms).
    turbo,
    /// High quality within a ~1s budget.
    fast,
    /// Low latency; works best with 2-3 good `search_queries`.
    basic,
    /// Provider default (~3s): better retrieval and excerpt compression.
    advanced,
};

/// Plain domains match their subdomains; entries may also be domain/path
/// prefixes ("docs.python.org/3") or bare extensions (".edu"). A non-empty
/// `include_domains` is the full allowlist and `exclude_domains` is ignored.
pub const SourcePolicy = struct {
    include_domains: ?[]const []const u8 = null,
    exclude_domains: ?[]const []const u8 = null,
    /// RFC 3339 date (YYYY-MM-DD): only content published on or after it.
    after_date: ?[]const u8 = null,
};

/// Live fetching instead of the index cache; adds noticeable latency.
pub const FetchPolicy = struct {
    /// Cached content older than this triggers a live fetch; minimum 600.
    max_age_seconds: ?u32 = null,
    timeout_seconds: ?f32 = null,
    /// Error out instead of falling back to stale cache when a fetch fails.
    disable_cache_fallback: ?bool = null,
};

pub const ExcerptSettings = struct {
    /// Upper bound; the API may return fewer characters per URL.
    max_chars_per_result: ?u32 = null,
};

pub const AdvancedSettings = struct {
    source_policy: ?SourcePolicy = null,
    fetch_policy: ?FetchPolicy = null,
    excerpt_settings: ?ExcerptSettings = null,
    /// ISO 3166-1 alpha-2 country code, e.g. "us".
    location: ?[]const u8 = null,
    /// Provider default 10.
    max_results: ?u8 = null,
};

pub const Result = struct {
    url: []const u8 = "",
    title: ?[]const u8 = null,
    /// YYYY-MM-DD.
    publish_date: ?[]const u8 = null,
    /// Markdown excerpts relevant to the objective and queries.
    excerpts: []const []const u8 = &.{},
};

pub const Warning = struct {
    type: []const u8 = "",
    message: []const u8 = "",
};

pub const UsageItem = struct {
    name: []const u8 = "",
    count: u32 = 0,
};

pub const SearchResponse = struct {
    search_id: []const u8 = "",
    /// Ordered by decreasing relevance.
    results: []const Result = &.{},
    warnings: ?[]const Warning = null,
    usage: ?[]const UsageItem = null,
    /// Pass back as `SearchOptions.session_id` on related calls.
    session_id: ?[]const u8 = null,
};

/// Doubles as the request body: `Client.search` fills `search_queries`
/// with its `query` when left empty.
pub const SearchOptions = struct {
    /// Natural-language goal behind the search, used with the queries to
    /// pick relevant excerpts. Should stand on its own.
    objective: ?[]const u8 = null,
    /// Keyword queries, 3-6 words each; the API recommends 2-3.
    search_queries: []const []const u8 = &.{},
    mode: ?Mode = null,
    /// Upper bound on excerpt characters across all results.
    max_chars_total: ?u32 = null,
    /// Ties related search/extract calls together for better context.
    session_id: ?[]const u8 = null,
    /// Model consuming the results, e.g. "claude-opus-5-5".
    client_model: ?[]const u8 = null,
    advanced_settings: ?AdvancedSettings = null,
};

test "SearchResponse parses Parallel fixture" {
    const fixture =
        \\{
        \\  "search_id": "search_cad0a6d2dec046bd95ae900527d880e7",
        \\  "results": [
        \\    {"url": "https://en.wikipedia.org/wiki/Paris", "title": "Paris - Wikipedia", "publish_date": "2026-01-02", "excerpts": ["Paris is the capital of France.", "## Geography\nThe Seine..."]},
        \\    {"url": "https://example.org/fr", "title": null, "publish_date": null, "excerpts": []}
        \\  ],
        \\  "warnings": null,
        \\  "usage": [{"name": "sku_search", "count": 1}],
        \\  "session_id": "session_8a911eb27c7a4afaa20d0d9dc98d07c0"
        \\}
    ;
    const parsed = try std.json.parseFromSlice(SearchResponse, std.testing.allocator, fixture, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqualStrings("search_cad0a6d2dec046bd95ae900527d880e7", parsed.value.search_id);
    const results = parsed.value.results;
    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("Paris - Wikipedia", results[0].title.?);
    try std.testing.expectEqualStrings("2026-01-02", results[0].publish_date.?);
    try std.testing.expectEqual(@as(usize, 2), results[0].excerpts.len);
    try std.testing.expect(results[1].title == null);
    try std.testing.expectEqual(@as(usize, 0), results[1].excerpts.len);
    try std.testing.expect(parsed.value.warnings == null);
    try std.testing.expectEqualStrings("sku_search", parsed.value.usage.?[0].name);
}

test "SearchOptions stringifies with enum tag and no null fields" {
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(SearchOptions{
        .objective = "find the zig release notes",
        .search_queries = &.{"zig 0.17 release notes"},
        .mode = .fast,
        .advanced_settings = .{ .max_results = 5, .source_policy = .{ .include_domains = &.{"ziglang.org"} } },
    }, .{ .emit_null_optional_fields = false }, &buf.writer);
    try std.testing.expectEqualStrings(
        \\{"objective":"find the zig release notes","search_queries":["zig 0.17 release notes"],"mode":"fast","advanced_settings":{"source_policy":{"include_domains":["ziglang.org"]},"max_results":5}}
    , buf.written());
}
