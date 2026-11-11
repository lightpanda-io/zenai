//! The provider-neutral questions and answers to and from TypeSafe System
//! One's wire schema. The two have the same shapes today; converting keeps a
//! change to TypeSafe's from reaching the neutral API.

const std = @import("std");
const types = @import("types.zig");
const typesafe = @import("../typesafe/types.zig");
const http = @import("../http.zig");
const jsonutil = @import("../json.zig");

const Content = types.Content;
const Questions = types.Questions;

pub const Request = struct {
    state: typesafe.Content,
    questions: typesafe.Questions,
};

/// Everything returned is allocated in `arena` or borrows `state`/`questions`.
pub fn toTypesafe(arena: std.mem.Allocator, state: Content, questions: Questions) std.mem.Allocator.Error!Request {
    const entries = try arena.alloc(typesafe.QuestionEntry, questions.count());
    for (questions.entries, entries) |entry, *out| {
        out.* = .{ .key = entry.key, .value = try question(arena, entry.value) };
    }
    return .{ .state = content(state), .questions = .init(entries) };
}

fn question(arena: std.mem.Allocator, q: types.Question) std.mem.Allocator.Error!typesafe.Question {
    return switch (q) {
        .noul => |n| .{ .noul = .{
            .instructions = content(n.instructions),
            .criteria = if (n.criteria) |c| .{
                .true = optionalContent(c.true),
                .false = optionalContent(c.false),
            } else null,
        } },
        .choice => |c| blk: {
            const options = try arena.alloc(typesafe.ChoiceCriteria.Entry, c.criteria.count());
            for (c.criteria.entries, options) |entry, *out| {
                out.* = .{ .key = entry.key, .value = optionalContent(entry.value) };
            }
            break :blk .{ .choice = .{ .instructions = content(c.instructions), .criteria = .init(options) } };
        },
        .score => |s| blk: {
            const levels = try arena.alloc(typesafe.Content, s.criteria.len);
            for (s.criteria, levels) |level, *out| out.* = content(level);
            break :blk .{ .score = .{ .instructions = content(s.instructions), .criteria = levels } };
        },
    };
}

fn content(c: Content) typesafe.Content {
    return switch (c) {
        .text => |t| .{ .text = t },
        .json => |v| .{ .json = v },
    };
}

fn optionalContent(c: ?Content) ?typesafe.Content {
    return if (c) |v| content(v) else null;
}

/// TypeSafe's answers as a neutral `Response`, which takes over `wire`'s
/// memory: on success `wire` must not be freed; on error it is still the
/// caller's.
pub fn fromTypesafe(wire: *http.Response(typesafe.AskResponse)) std.mem.Allocator.Error!types.Response {
    const a = wire.parsed.arena.allocator();
    const response = wire.value;

    const answers = try a.alloc(types.Answers.Entry, response.answers.count());
    for (response.answers.entries, answers) |entry, *out| {
        out.* = .{ .key = entry.key, .value = switch (entry.value) {
            .noul => |n| .{ .noul = .{ .noul = n.noul } },
            .choice => |c| .{ .choice = .{
                .choice = c.choice,
                .probabilities = c.probabilities,
                .confidence = c.confidence,
            } },
            .score => |s| .{ .score = .{
                .score = s.score,
                .legend = try legend(a, s.legend),
                .probabilities = s.probabilities,
                .confidence = s.confidence,
            } },
        } };
    }

    return types.adopt(wire, .{
        .model = response.model,
        .answers = .init(answers),
        .usage = .{ .input_tokens = response.usage.input_tokens, .output_tokens = response.usage.output_tokens },
    });
}

fn legend(a: std.mem.Allocator, wire: typesafe.Legend) std.mem.Allocator.Error!types.Legend {
    const out = try a.alloc(types.Legend.Entry, wire.count());
    for (wire.entries, out) |entry, *o| o.* = .{ .key = entry.key, .value = switch (entry.value) {
        .text => |t| .{ .text = t },
        .json => |v| .{ .json = v },
    } };
    return .init(out);
}

fn expectJson(value: anytype, expected: []const u8) !void {
    const out = try jsonutil.stringifyAlloc(std.testing.allocator, value, .{ .emit_null_optional_fields = false });
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings(expected, out);
}

test "toTypesafe passes text and JSON content through" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var state: std.json.ObjectMap = .empty;
    try state.put(arena, "url", .{ .string = "https://example.com" });

    const request = try toTypesafe(arena, .{ .json = .{ .object = state } }, .init(&.{
        .{ .key = "blocked", .value = .{ .noul = .{
            .instructions = .{ .text = "Blocked?" },
            .criteria = .{ .true = .{ .text = "A CAPTCHA" } },
        } } },
        .{ .key = "pick", .value = .choiceText("Which?", .init(&.{
            .{ .key = "1", .value = .{ .text = "the first" } },
            .{ .key = "none", .value = null },
        })) },
        .{ .key = "level", .value = .scoreText("How much?", types.levels(&.{ "low", "high" })) },
    }));

    try expectJson(typesafe.AskRequest{ .state = request.state, .questions = request.questions },
        \\{"state":{"url":"https://example.com"},"model":"jev-latest","questions":{"blocked":{"type":"noul","instructions":"Blocked?","criteria":{"true":"A CAPTCHA"}},"pick":{"type":"choice","instructions":"Which?","criteria":{"1":"the first","none":null}},"level":{"type":"score","instructions":"How much?","criteria":["low","high"]}}}
    );
}

test "fromTypesafe takes over the wire response" {
    const parsed = try std.json.parseFromSlice(typesafe.AskResponse, std.testing.allocator,
        \\{"model":"jev-1.13.0","answers":{
        \\  "blocked":{"type":"noul","noul":0.9},
        \\  "pick":{"type":"choice","choice":"1","probabilities":{"1":0.8,"none":0.2},"confidence":0.8},
        \\  "level":{"type":"score","score":0.7,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.3,"1":0.7},"confidence":0.7}
        \\},"usage":{"input_tokens":12,"output_tokens":3}}
    , .{ .ignore_unknown_fields = true });
    var wire: http.Response(typesafe.AskResponse) = .{ .value = parsed.value, .json_buf = .init(std.testing.allocator), .parsed = parsed };
    var response = fromTypesafe(&wire) catch |err| {
        wire.deinit();
        return err;
    };
    defer response.deinit();
    const r = response.value;

    try std.testing.expectEqualStrings("jev-1.13.0", r.model);
    try std.testing.expectEqual(@as(u32, 3), r.usage.output_tokens);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), r.noul("blocked").?, 1e-12);
    try std.testing.expectEqualStrings("1", try r.choice("pick", .init(&.{
        .{ .key = "pick", .value = .choiceText("?", types.choices(&.{ "1", "none" })) },
    })));
    try std.testing.expectEqualStrings("high", r.answer("level").?.score.nearestLevel().?.value.text);
}
