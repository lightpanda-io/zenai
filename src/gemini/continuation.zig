//! Automatic continuation: a candidate that stops with finish reason
//! `CONTINUATION` (or carries a checkpoint `continuationToken` mid-stream) is
//! resumed by sending the same request again with that token, until the model
//! finishes. Ported from go-genai's continuation.go.

const std = @import("std");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Candidate = types.Candidate;
const FinishReason = types.FinishReason;
const GenerateContentResponse = types.GenerateContentResponse;
const ModalityTokenCount = types.ModalityTokenCount;
const SafetyRating = types.SafetyRating;
const UsageMetadata = types.UsageMetadata;

/// Per the continuation protocol, the backend pairs a token only with no
/// finish reason (an intermediate checkpoint) or `CONTINUATION` (the end of a
/// request that the model has not finished).
fn isResumable(finish_reason: ?FinishReason) bool {
    const reason = finish_reason orelse return true;
    return switch (reason) {
        .FINISH_REASON_UNSPECIFIED, .CONTINUATION => true,
        else => false,
    };
}

fn isSet(finish_reason: ?FinishReason) bool {
    const reason = finish_reason orelse return false;
    return reason != .FINISH_REASON_UNSPECIFIED;
}

/// The token to resume the first candidate from, or null when the model has
/// finished.
pub fn resumeToken(response: GenerateContentResponse) ?[]const u8 {
    const candidates = response.candidates orelse return null;
    if (candidates.len == 0) return null;
    const token = candidates[0].continuationToken orelse return null;
    if (token.len == 0 or !isResumable(candidates[0].finishReason)) return null;
    return token;
}

/// Merge the response of a continuation request into the responses before it:
/// parts are concatenated, usage is summed, and the finish reason and token are
/// the latest request's. Slices of both are borrowed, so they must outlive the
/// result; new slices are allocated in `arena`.
pub fn merge(arena: Allocator, prev: GenerateContentResponse, curr: GenerateContentResponse) Allocator.Error!GenerateContentResponse {
    var out = latest(GenerateContentResponse, prev, curr);
    out.candidates = try mergeCandidates(arena, prev.candidates, curr.candidates);
    out.usageMetadata = try mergeUsage(arena, prev.usageMetadata, curr.usageMetadata);
    return out;
}

/// Each optional field of `curr`, falling back to `prev`'s where it is null.
fn latest(comptime T: type, prev: T, curr: T) T {
    var out: T = undefined;
    inline for (@typeInfo(T).@"struct".field_names) |name| {
        @field(out, name) = @field(curr, name) orelse @field(prev, name);
    }
    return out;
}

fn concat(comptime T: type, arena: Allocator, prev: ?[]const T, curr: ?[]const T) Allocator.Error!?[]const T {
    const p = prev orelse return curr;
    const c = curr orelse return p;
    if (p.len == 0) return c;
    if (c.len == 0) return p;
    return try std.mem.concat(arena, T, &.{ p, c });
}

fn addOpt(a: ?i32, b: ?i32) ?i32 {
    if (a == null and b == null) return null;
    return (a orelse 0) + (b orelse 0);
}

fn mergeCandidates(arena: Allocator, prev: ?[]const Candidate, curr: ?[]const Candidate) Allocator.Error!?[]const Candidate {
    const p = prev orelse return curr;
    const c = curr orelse return p;
    const out = try arena.alloc(Candidate, @max(p.len, c.len));
    for (out, 0..) |*slot, i| {
        slot.* = if (i >= p.len)
            c[i]
        else if (i >= c.len)
            p[i]
        else
            try mergeCandidate(arena, p[i], c[i]);
    }
    return out;
}

fn mergeCandidate(arena: Allocator, prev: Candidate, curr: Candidate) Allocator.Error!Candidate {
    var out = latest(Candidate, prev, curr);
    // How the latest request ended, even when it says nothing about it.
    out.finishReason = curr.finishReason;
    out.finishMessage = curr.finishMessage;
    out.continuationToken = curr.continuationToken;
    out.tokenCount = addOpt(prev.tokenCount, curr.tokenCount);
    out.safetyRatings = try mergeSafetyRatings(arena, prev.safetyRatings, curr.safetyRatings);
    if (prev.content != null and curr.content != null) {
        out.content = .{
            .role = curr.content.?.role orelse prev.content.?.role,
            .parts = (try concat(types.Part, arena, prev.content.?.parts, curr.content.?.parts)).?,
        };
    }
    if (prev.citationMetadata != null and curr.citationMetadata != null) {
        out.citationMetadata = .{ .citationSources = try concat(types.CitationSource, arena, prev.citationMetadata.?.citationSources, curr.citationMetadata.?.citationSources) };
    }
    if (prev.logprobsResult != null and curr.logprobsResult != null) {
        out.logprobsResult = .{
            .topCandidates = try concat(types.TopCandidates, arena, prev.logprobsResult.?.topCandidates, curr.logprobsResult.?.topCandidates),
            .chosenCandidates = try concat(types.TokenLogprob, arena, prev.logprobsResult.?.chosenCandidates, curr.logprobsResult.?.chosenCandidates),
        };
    }
    return out;
}

fn sameCategory(a: ?types.HarmCategory, b: ?types.HarmCategory) bool {
    const x = a orelse return b == null;
    const y = b orelse return false;
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .unknown => |s| std.mem.eql(u8, s, y.unknown),
        else => true,
    };
}

/// One rating per category, the latest one winning, in first-seen order.
fn mergeSafetyRatings(arena: Allocator, prev: ?[]const SafetyRating, curr: ?[]const SafetyRating) Allocator.Error!?[]const SafetyRating {
    const p = prev orelse return curr;
    const c = curr orelse return p;
    var out: std.ArrayList(SafetyRating) = try .initCapacity(arena, p.len + c.len);
    for ([_][]const SafetyRating{ p, c }) |ratings| {
        next: for (ratings) |rating| {
            for (out.items) |*seen| {
                if (sameCategory(seen.category, rating.category)) {
                    seen.* = rating;
                    continue :next;
                }
            }
            out.appendAssumeCapacity(rating);
        }
    }
    return out.items;
}

fn sameModality(a: ?types.MediaModality, b: ?types.MediaModality) bool {
    const x = a orelse return b == null;
    const y = b orelse return false;
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .unknown => |s| std.mem.eql(u8, s, y.unknown),
        else => true,
    };
}

/// One count per modality, summed, in first-seen order.
fn mergeModalityCounts(arena: Allocator, prev: ?[]const ModalityTokenCount, curr: ?[]const ModalityTokenCount) Allocator.Error!?[]const ModalityTokenCount {
    const p = prev orelse return curr;
    const c = curr orelse return p;
    var out: std.ArrayList(ModalityTokenCount) = try .initCapacity(arena, p.len + c.len);
    for ([_][]const ModalityTokenCount{ p, c }) |counts| {
        next: for (counts) |count| {
            for (out.items) |*seen| {
                if (sameModality(seen.modality, count.modality)) {
                    seen.tokenCount = addOpt(seen.tokenCount, count.tokenCount);
                    continue :next;
                }
            }
            out.appendAssumeCapacity(count);
        }
    }
    return out.items;
}

/// Every count summed, so the usage covers all the requests a continued
/// generation took (each of which is billed).
pub fn mergeUsage(arena: Allocator, prev: ?UsageMetadata, curr: ?UsageMetadata) Allocator.Error!?UsageMetadata {
    const p = prev orelse return curr;
    const c = curr orelse return p;
    var out: UsageMetadata = .{};
    const info = @typeInfo(UsageMetadata).@"struct";
    inline for (info.field_names, info.field_types) |name, F| {
        @field(out, name) = switch (F) {
            ?i32 => addOpt(@field(p, name), @field(c, name)),
            ?[]const ModalityTokenCount => try mergeModalityCounts(arena, @field(p, name), @field(c, name)),
            else => @field(c, name) orelse @field(p, name),
        };
    }
    return out;
}

fn dupeUsage(arena: Allocator, usage: UsageMetadata) Allocator.Error!UsageMetadata {
    var out = usage;
    const info = @typeInfo(UsageMetadata).@"struct";
    inline for (info.field_names, info.field_types) |name, F| {
        if (F == ?[]const ModalityTokenCount) {
            if (@field(usage, name)) |counts| {
                const copy = try arena.alloc(ModalityTokenCount, counts.len);
                for (copy, counts) |*dst, src| {
                    dst.* = src;
                    if (src.modality) |m| switch (m) {
                        .unknown => |s| dst.modality = .{ .unknown = try arena.dupe(u8, s) },
                        else => {},
                    };
                }
                @field(out, name) = copy;
            }
        }
    }
    return out;
}

/// Wraps a stream callback across the requests of a continued generation:
/// chunks of every request reach `user_cb` in order, a `CONTINUATION` finish
/// reason the next request supersedes is cleared, and usage is summed over the
/// requests so far.
pub fn Stream(comptime Context: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        user_ctx: Context,
        user_cb: *const fn (Context, GenerateContentResponse) void,
        /// Usage summed over the finished requests.
        prior_usage: ?UsageMetadata = null,
        prior_arena: std.heap.ArenaAllocator,
        /// This request's latest usage, `prior_usage` included.
        hop_usage: ?UsageMetadata = null,
        hop_arena: std.heap.ArenaAllocator,
        /// Copies of the chunk handed to `user_cb`, freed after it returns.
        chunk_arena: std.heap.ArenaAllocator,
        /// This request's latest checkpoint token.
        token: std.ArrayList(u8) = .empty,
        has_token: bool = false,
        /// Whether this request's finish reason, if any, is resumable.
        resumable: bool = true,
        /// The token the next request is sent with.
        sent: std.ArrayList(u8) = .empty,
        err: ?Allocator.Error = null,

        pub fn init(allocator: Allocator, user_ctx: Context, user_cb: *const fn (Context, GenerateContentResponse) void) Self {
            return .{
                .allocator = allocator,
                .user_ctx = user_ctx,
                .user_cb = user_cb,
                .prior_arena = .init(allocator),
                .hop_arena = .init(allocator),
                .chunk_arena = .init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.prior_arena.deinit();
            self.hop_arena.deinit();
            self.chunk_arena.deinit();
            self.token.deinit(self.allocator);
            self.sent.deinit(self.allocator);
        }

        /// Whether the request that just ended, normally or cut off, can be
        /// resumed from a checkpoint.
        pub fn canResume(self: *const Self) bool {
            return self.has_token and self.resumable;
        }

        /// Call when a request ends. Returns the token to send the next request
        /// with, valid until the following `next` call, or null when the model
        /// has finished.
        pub fn next(self: *Self) Allocator.Error!?[]const u8 {
            if (!self.canResume()) return null;
            if (self.hop_usage) |u| {
                // `hop_usage` already includes `prior_usage`, so this is a
                // replace; `prior_arena` grows by one copy per request.
                self.prior_usage = try dupeUsage(self.prior_arena.allocator(), u);
            }
            std.mem.swap(std.ArrayList(u8), &self.token, &self.sent);
            self.token.clearRetainingCapacity();
            self.has_token = false;
            self.resumable = true;
            self.hop_usage = null;
            _ = self.hop_arena.reset(.retain_capacity);
            return self.sent.items;
        }

        pub fn onChunk(self: *Self, chunk: GenerateContentResponse) void {
            const out = self.handle(chunk) catch |e| blk: {
                self.err = e;
                break :blk chunk;
            };
            self.user_cb(self.user_ctx, out);
            _ = self.chunk_arena.reset(.retain_capacity);
        }

        fn handle(self: *Self, chunk: GenerateContentResponse) Allocator.Error!GenerateContentResponse {
            var out = chunk;
            const scratch = self.chunk_arena.allocator();
            const candidates = chunk.candidates orelse &.{};
            if (candidates.len > 0) {
                const cand = candidates[0];
                if (isSet(cand.finishReason)) self.resumable = isResumable(cand.finishReason);
                const token = cand.continuationToken orelse "";
                if (token.len > 0) {
                    self.token.clearRetainingCapacity();
                    try self.token.appendSlice(self.allocator, token);
                    self.has_token = true;
                } else if (isSet(cand.finishReason) and !self.resumable) {
                    self.has_token = false;
                }
                // The next request carries on from here, so this one has not
                // finished the candidate.
                if (isSet(cand.finishReason) and self.canResume()) {
                    const copy = try scratch.dupe(Candidate, candidates);
                    copy[0].finishReason = null;
                    out.candidates = copy;
                }
            }
            if (chunk.usageMetadata) |u| {
                const total = (try mergeUsage(scratch, self.prior_usage, u)).?;
                out.usageMetadata = total;
                _ = self.hop_arena.reset(.retain_capacity);
                self.hop_usage = try dupeUsage(self.hop_arena.allocator(), total);
            }
            return out;
        }
    };
}

test "resumeToken: only a resumable finish reason continues" {
    const cases = [_]struct { Candidate, ?[]const u8 }{
        .{ .{ .finishReason = .CONTINUATION, .continuationToken = "t" }, "t" },
        .{ .{ .continuationToken = "t" }, "t" },
        .{ .{ .finishReason = .CONTINUATION }, null },
        .{ .{ .finishReason = .CONTINUATION, .continuationToken = "" }, null },
        .{ .{ .finishReason = .STOP, .continuationToken = "t" }, null },
    };
    for (cases) |case| {
        const token = resumeToken(.{ .candidates = &.{case[0]} });
        if (case[1]) |want| {
            try std.testing.expectEqualStrings(want, token.?);
        } else {
            try std.testing.expect(token == null);
        }
    }
    try std.testing.expect(resumeToken(.{}) == null);
}

test "merge concatenates parts, sums usage, and keeps the latest finish" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const a_parts = [_]types.Part{.{ .text = "Hello, " }};
    const b_parts = [_]types.Part{.{ .text = "world." }};
    const a_details = [_]ModalityTokenCount{.{ .modality = .TEXT, .tokenCount = 3 }};
    const b_details = [_]ModalityTokenCount{ .{ .modality = .TEXT, .tokenCount = 2 }, .{ .modality = .IMAGE, .tokenCount = 1 } };
    const a_ratings = [_]SafetyRating{.{ .category = .HARM_CATEGORY_HARASSMENT, .probability = .LOW }};
    const b_ratings = [_]SafetyRating{.{ .category = .HARM_CATEGORY_HARASSMENT, .probability = .NEGLIGIBLE }};

    const first: GenerateContentResponse = .{
        .candidates = &.{.{
            .content = .{ .role = "model", .parts = &a_parts },
            .finishReason = .CONTINUATION,
            .continuationToken = "abc",
            .safetyRatings = &a_ratings,
            .tokenCount = 3,
        }},
        .usageMetadata = .{ .promptTokenCount = 10, .candidatesTokenCount = 3, .totalTokenCount = 13, .candidatesTokensDetails = &a_details },
        .modelVersion = "v1",
        .responseId = "r1",
    };
    const second: GenerateContentResponse = .{
        .candidates = &.{.{
            .content = .{ .role = "model", .parts = &b_parts },
            .finishReason = .STOP,
            .safetyRatings = &b_ratings,
            .tokenCount = 2,
        }},
        .usageMetadata = .{ .promptTokenCount = 12, .candidatesTokenCount = 3, .totalTokenCount = 15, .candidatesTokensDetails = &b_details },
        .responseId = "r2",
    };

    const merged = try merge(arena, first, second);
    const cand = merged.candidates.?[0];
    try std.testing.expectEqual(@as(usize, 2), cand.content.?.parts.len);
    try std.testing.expectEqualStrings("world.", cand.content.?.parts[1].text.?);
    try std.testing.expect(cand.finishReason.? == .STOP);
    try std.testing.expect(cand.continuationToken == null);
    try std.testing.expectEqual(@as(?i32, 5), cand.tokenCount);
    try std.testing.expectEqual(@as(usize, 1), cand.safetyRatings.?.len);
    try std.testing.expect(cand.safetyRatings.?[0].probability.? == .NEGLIGIBLE);
    try std.testing.expect(resumeToken(merged) == null);

    const usage = merged.usageMetadata.?;
    try std.testing.expectEqual(@as(?i32, 22), usage.promptTokenCount);
    try std.testing.expectEqual(@as(?i32, 28), usage.totalTokenCount);
    try std.testing.expect(usage.cachedContentTokenCount == null);
    try std.testing.expectEqual(@as(usize, 2), usage.candidatesTokensDetails.?.len);
    try std.testing.expectEqual(@as(?i32, 5), usage.candidatesTokensDetails.?[0].tokenCount);

    try std.testing.expectEqualStrings("v1", merged.modelVersion.?);
    try std.testing.expectEqualStrings("r2", merged.responseId.?);
}

const Collect = struct {
    arena: std.heap.ArenaAllocator,
    finishes: std.ArrayList(?FinishReason) = .empty,
    totals: std.ArrayList(?i32) = .empty,

    fn cb(self: *Collect, chunk: GenerateContentResponse) void {
        const a = self.arena.allocator();
        const fr = if (chunk.candidates) |c| c[0].finishReason else null;
        self.finishes.append(a, fr) catch unreachable;
        self.totals.append(a, if (chunk.usageMetadata) |u| u.totalTokenCount else null) catch unreachable;
    }
};

test "Stream clears a superseded CONTINUATION and sums usage across requests" {
    var collect: Collect = .{ .arena = .init(std.testing.allocator) };
    defer collect.arena.deinit();
    var s = Stream(*Collect).init(std.testing.allocator, &collect, Collect.cb);
    defer s.deinit();

    // Request 1: a checkpoint mid-stream, then CONTINUATION at the end.
    s.onChunk(.{ .candidates = &.{.{ .continuationToken = "ck1" }}, .usageMetadata = .{ .totalTokenCount = 5 } });
    s.onChunk(.{ .candidates = &.{.{ .finishReason = .CONTINUATION, .continuationToken = "ck2" }}, .usageMetadata = .{ .totalTokenCount = 10 } });
    try std.testing.expectEqualStrings("ck2", (try s.next()).?);

    // Request 2 finishes.
    s.onChunk(.{ .candidates = &.{.{ .finishReason = .STOP }}, .usageMetadata = .{ .totalTokenCount = 7 } });
    try std.testing.expect((try s.next()) == null);
    try std.testing.expect(s.err == null);

    try std.testing.expectEqual(@as(usize, 3), collect.finishes.items.len);
    try std.testing.expect(collect.finishes.items[0] == null);
    try std.testing.expect(collect.finishes.items[1] == null);
    try std.testing.expect(collect.finishes.items[2].? == .STOP);
    try std.testing.expectEqualSlices(?i32, &.{ 5, 10, 17 }, collect.totals.items);
}

test "Stream resumes a cut-off request from its last checkpoint" {
    var collect: Collect = .{ .arena = .init(std.testing.allocator) };
    defer collect.arena.deinit();
    var s = Stream(*Collect).init(std.testing.allocator, &collect, Collect.cb);
    defer s.deinit();

    try std.testing.expect(!s.canResume());
    s.onChunk(.{ .candidates = &.{.{ .continuationToken = "ck1" }} });
    // The stream errors out here, before a finish reason.
    try std.testing.expect(s.canResume());
    try std.testing.expectEqualStrings("ck1", (try s.next()).?);
    // A request that fails before its first checkpoint is not resumed.
    try std.testing.expect(!s.canResume());
}

test "Stream: a final finish reason drops the checkpoint" {
    var collect: Collect = .{ .arena = .init(std.testing.allocator) };
    defer collect.arena.deinit();
    var s = Stream(*Collect).init(std.testing.allocator, &collect, Collect.cb);
    defer s.deinit();

    s.onChunk(.{ .candidates = &.{.{ .continuationToken = "ck1" }} });
    s.onChunk(.{ .candidates = &.{.{ .finishReason = .MAX_TOKENS }} });
    try std.testing.expect((try s.next()) == null);
    try std.testing.expect(collect.finishes.items[1].? == .MAX_TOKENS);
}
