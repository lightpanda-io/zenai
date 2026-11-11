//! The provider-neutral questions and answers (TypeSafe's shapes, which are
//! the superset) to and from Gemini's Decisions API. Gemini takes only strings
//! where TypeSafe also takes JSON, so JSON content is sent as compact JSON
//! text.

const std = @import("std");
const types = @import("../typesafe/types.zig");
const gemini = @import("../gemini/decisions.zig");
const http = @import("../http.zig");
const jsonutil = @import("../json.zig");

const Content = types.Content;
const Questions = types.Questions;

pub const Request = struct {
    input: gemini.Input,
    questions: gemini.Questions,
};

/// Everything returned is allocated in `arena` or borrows `state`/`questions`.
pub fn toGemini(arena: std.mem.Allocator, state: Content, questions: Questions) std.mem.Allocator.Error!Request {
    const entries = try arena.alloc(gemini.QuestionEntry, questions.count());
    for (questions.entries, entries) |entry, *out| {
        out.* = .{ .key = entry.key, .value = try question(arena, entry.value) };
    }
    return .{
        .input = .{ .text = try text(arena, state) },
        .questions = .init(entries),
    };
}

fn question(arena: std.mem.Allocator, q: types.Question) std.mem.Allocator.Error!gemini.Question {
    return switch (q) {
        .noul => |n| .{ .noul = .{
            .instructions = try text(arena, n.instructions),
            .criteria = if (n.criteria) |c| .{
                .true_description = try optionalText(arena, c.true),
                .false_description = try optionalText(arena, c.false),
            } else null,
        } },
        .choice => |c| blk: {
            const options = try arena.alloc(gemini.ChoiceCriteria.Entry, c.criteria.count());
            for (c.criteria.entries, options) |entry, *out| {
                out.* = .{ .key = entry.key, .value = try optionalText(arena, entry.value) orelse "" };
            }
            break :blk .{ .choice = .{ .instructions = try text(arena, c.instructions), .criteria = .init(options) } };
        },
        .score => |s| blk: {
            const levels = try arena.alloc([]const u8, s.criteria.len);
            for (s.criteria, levels) |level, *out| out.* = try text(arena, level);
            break :blk .{ .score = .{ .instructions = try text(arena, s.instructions), .criteria = levels } };
        },
    };
}

fn text(arena: std.mem.Allocator, content: Content) std.mem.Allocator.Error![]const u8 {
    return switch (content) {
        .text => |t| t,
        .json => |value| jsonutil.stringifyAlloc(arena, value, .{}),
    };
}

fn optionalText(arena: std.mem.Allocator, content: ?Content) std.mem.Allocator.Error!?[]const u8 {
    return if (content) |c| try text(arena, c) else null;
}

pub const ToNeutralError = std.mem.Allocator.Error || error{InputBlocked};

/// Gemini's answers as an `AskResponse` that owns its own memory, so the Gemini
/// response can be freed right away. A blocked input is `error.InputBlocked`,
/// with the reason in `failure` (allocated in `arena`).
pub fn fromGemini(
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    response: gemini.CreateDecisionResponse,
    model: []const u8,
    failure: *http.ErrorDetail,
) ToNeutralError!http.Response(types.AskResponse) {
    if (response.prompt_feedback) |feedback| {
        if (feedback.block_reason) |reason| {
            failure.* = .{ .message = if (feedback.block_reason_message) |m|
                try arena.print("{s}: {s}", .{ reason, m })
            else
                try arena.dupe(u8, reason) };
            return error.InputBlocked;
        }
    }

    const owned = try allocator.create(std.heap.ArenaAllocator);
    owned.* = .init(allocator);
    errdefer {
        owned.deinit();
        allocator.destroy(owned);
    }
    const a = owned.allocator();

    var answers: std.ArrayList(types.Answers.Entry) = try .initCapacity(a, response.answers.count());
    for (response.answers.entries) |entry| {
        const answer: types.Answer = switch (entry.value) {
            .noul => |n| .{ .noul = .{ .noul = n.noul } },
            .choice => |c| .{ .choice = .{
                .choice = try a.dupe(u8, c.choice),
                .probabilities = try dupeProbabilities(a, c.probabilities),
                .confidence = c.confidence,
            } },
            .score => |s| .{ .score = .{
                .score = s.score,
                .legend = try dupeLegend(a, s.legend),
                .probabilities = try dupeProbabilities(a, s.probabilities),
                .confidence = s.confidence,
            } },
            // An answer type the neutral shapes cannot hold reads as missing.
            .unknown => continue,
        };
        answers.appendAssumeCapacity(.{ .key = try a.dupe(u8, entry.key), .value = answer });
    }

    const usage: gemini.Usage = response.usage orelse .{};
    const value: types.AskResponse = .{
        .model = try a.dupe(u8, model),
        .answers = .init(answers.items),
        .usage = .{
            .input_tokens = usage.total_input_tokens orelse 0,
            .output_tokens = usage.total_output_tokens orelse 0,
        },
    };
    return .{ .value = value, .json_buf = .init(allocator), .parsed = .{ .arena = owned, .value = value } };
}

fn dupeProbabilities(a: std.mem.Allocator, p: gemini.Probabilities) std.mem.Allocator.Error!types.Probabilities {
    const out = try a.alloc(types.Probabilities.Entry, p.count());
    for (p.entries, out) |entry, *o| o.* = .{ .key = try a.dupe(u8, entry.key), .value = entry.value };
    return .init(out);
}

fn dupeLegend(a: std.mem.Allocator, legend: gemini.Legend) std.mem.Allocator.Error!types.Legend {
    const out = try a.alloc(types.Legend.Entry, legend.count());
    for (legend.entries, out) |entry, *o| o.* = .{ .key = try a.dupe(u8, entry.key), .value = .{ .text = try a.dupe(u8, entry.value) } };
    return .init(out);
}

fn expectJson(value: anytype, expected: []const u8) !void {
    const out = try jsonutil.stringifyAlloc(std.testing.allocator, value, .{ .emit_null_optional_fields = false });
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings(expected, out);
}

test "toGemini sends JSON content as compact JSON text" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var state: std.json.ObjectMap = .empty;
    try state.put(arena, "url", .{ .string = "https://example.com" });
    var link: std.json.ObjectMap = .empty;
    try link.put(arena, "role", .{ .string = "link" });

    const request = try toGemini(arena, .{ .json = .{ .object = state } }, .init(&.{
        .{ .key = "blocked", .value = .{ .noul = .{
            .instructions = .{ .text = "Blocked?" },
            .criteria = .{ .true = .{ .text = "A CAPTCHA" } },
        } } },
        .{ .key = "pick", .value = .choiceText("Which?", .init(&.{
            .{ .key = "1", .value = .{ .json = .{ .object = link } } },
            .{ .key = "none", .value = null },
        })) },
        .{ .key = "level", .value = .scoreText("How much?", types.levels(&.{ "low", "high" })) },
    }));

    try expectJson(request.input, "\"{\\\"url\\\":\\\"https://example.com\\\"}\"");
    try expectJson(request.questions,
        \\{"blocked":{"type":"noul","instructions":"Blocked?","criteria":{"true_description":"A CAPTCHA"}},"pick":{"type":"choice","instructions":"Which?","criteria":{"1":"{\"role\":\"link\"}","none":""}},"level":{"type":"score","instructions":"How much?","criteria":["low","high"]}}
    );
}

test "fromGemini copies the answers out of the Gemini response" {
    var failure: http.ErrorDetail = .{};
    var response = blk: {
        const parsed = try std.json.parseFromSlice(gemini.CreateDecisionResponse, std.testing.allocator,
            \\{"answers":{
            \\  "blocked":{"type":"noul","noul":0.9,"confidence":0.9},
            \\  "pick":{"type":"choice","choice":"1","probabilities":{"1":0.8,"none":0.2},"confidence":0.8},
            \\  "level":{"type":"score","score":0.7,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.3,"1":0.7},"confidence":0.7},
            \\  "future":{"type":"ranking"}
            \\},"usage":{"total_input_tokens":12,"total_output_tokens":0}}
        , .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        break :blk try fromGemini(std.testing.allocator, std.testing.allocator, parsed.value, gemini.default_model, &failure);
    };
    defer response.deinit();
    const r = response.value;

    try std.testing.expectEqualStrings(gemini.default_model, r.model);
    try std.testing.expectEqual(@as(u32, 12), r.usage.input_tokens);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), r.noul("blocked").?, 1e-12);
    try std.testing.expectEqualStrings("1", try r.choice("pick", .init(&.{
        .{ .key = "pick", .value = .choiceText("?", types.choices(&.{ "1", "none" })) },
    })));
    try std.testing.expectEqualStrings("high", r.answer("level").?.score.nearestLevel().?.value.text);
    try std.testing.expect(r.answer("future") == null);
}

test "fromGemini turns a blocked input into an error" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();

    var failure: http.ErrorDetail = .{};
    try std.testing.expectError(error.InputBlocked, fromGemini(std.testing.allocator, arena_state.allocator(), .{
        .prompt_feedback = .{ .block_reason = "safety", .block_reason_message = "nope" },
    }, gemini.default_model, &failure));
    try std.testing.expectEqualStrings("safety: nope", failure.message.?);
}
