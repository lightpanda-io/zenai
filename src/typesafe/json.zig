//! Questions from JSON, answers back as JSON — the runtime counterpart to
//! `types.choices`/`types.levels`/`types.enumChoices`, which build the same
//! questions at comptime.
//!
//! The shapes `prepareQuestions` accepts:
//!
//!   - an array of strings — the categories of one `choice` question
//!   - an object, keyed by the name each answer comes back under:
//!       - `true` or `""`      — a question the caller's `Vocabulary` names
//!       - a string            — that string as a yes/no question
//!       - an array of strings — a `choice` over them
//!       - `{question, options|criteria}` — a `choice`
//!       - `{question, levels}`           — a `score` over 2-10 levels

const std = @import("std");
const types = @import("types.zig");

const ChoiceEntry = types.ChoiceCriteria.Entry;
const Content = types.Content;
const Question = types.Question;
const QuestionEntry = types.QuestionEntry;
const Questions = types.Questions;

/// The key a bare category array answers under, since it names no question of
/// its own.
pub const category_key = "__category";

/// A caller's own question names: given a key, the text to ask for it, or null
/// when the key names nothing. It is what makes `{"isCaptcha": true}` mean
/// something without this module knowing what a captcha is.
pub const Vocabulary = *const fn (key: []const u8) ?[]const u8;

pub const Options = struct {
    vocabulary: ?Vocabulary = null,
    /// What to ask of a bare category array, which names no question of its
    /// own. The wording is the caller's: it is a prompt, not a protocol.
    category_instructions: []const u8,
};

pub const ResponseKind = enum {
    /// One bare choice: the answer is the category, not an object.
    single_choice,
    /// An answer per question asked, keyed as the request was.
    questions_object,
};

pub const Prepared = struct {
    questions: Questions,
    kind: ResponseKind,
};

pub const PrepareError = error{
    InvalidQuestionsJson,
    /// An array with no categories in it.
    EmptyQuestions,
    /// An object with no questions in it. Callers that give `{}` a meaning of
    /// their own — every question they know, say — handle this rather than
    /// reporting it.
    EmptyObject,
    InvalidQuestionFormat,
    OutOfMemory,
};

/// `questions` is the array or object itself, or a string holding its JSON.
/// Everything is allocated in `arena` and borrows from it and from `questions`.
pub fn prepareQuestions(
    arena: std.mem.Allocator,
    questions: std.json.Value,
    options: Options,
) PrepareError!Prepared {
    const parsed = switch (questions) {
        .string => |source| std.json.parseFromSliceLeaky(std.json.Value, arena, source, .{}) catch
            return error.InvalidQuestionsJson,
        else => questions,
    };

    switch (parsed) {
        .array => |arr| {
            if (arr.items.len == 0) return error.EmptyQuestions;

            // An array is always the categories of one choice question.
            const entries = try arena.alloc(QuestionEntry, 1);
            entries[0] = .{
                .key = category_key,
                .value = .choiceText(options.category_instructions, try stringChoices(arena, arr.items)),
            };
            return .{ .questions = .init(entries), .kind = .single_choice };
        },
        .object => |obj| {
            if (obj.count() == 0) return error.EmptyObject;

            const entries = try arena.alloc(QuestionEntry, obj.count());
            for (entries, obj.keys(), obj.values()) |*entry, key, val| {
                entry.* = .{ .key = key, .value = try prepareQuestion(arena, key, val, options.vocabulary) };
            }
            return .{ .questions = .init(entries), .kind = .questions_object };
        },
        else => return error.InvalidQuestionFormat,
    }
}

fn prepareQuestion(
    arena: std.mem.Allocator,
    key: []const u8,
    val: std.json.Value,
    vocabulary: ?Vocabulary,
) PrepareError!Question {
    switch (val) {
        .bool => |b| {
            if (!b) return error.InvalidQuestionFormat;
            // Without a vocabulary the key has to speak for itself.
            const named = if (vocabulary) |vocab| vocab(key) else null;
            return .noulText(named orelse key);
        },
        .string => |s| {
            if (s.len > 0) return .noulText(s);
            const vocab = vocabulary orelse return error.InvalidQuestionFormat;
            return .noulText(vocab(key) orelse return error.InvalidQuestionFormat);
        },
        .array => |arr| return .choiceText(key, try stringChoices(arena, arr.items)),
        .object => |obj| return objectQuestion(arena, key, obj),
        else => return error.InvalidQuestionFormat,
    }
}

fn objectQuestion(arena: std.mem.Allocator, key: []const u8, obj: std.json.ObjectMap) PrepareError!Question {
    const instructions = blk: {
        if (obj.get("question")) |qv| if (qv == .string) break :blk qv.string;
        if (obj.get("instructions")) |iv| if (iv == .string) break :blk iv.string;
        break :blk key;
    };

    if (obj.get("levels")) |levels_val| {
        if (levels_val != .array) return error.InvalidQuestionFormat;
        const items = levels_val.array.items;
        if (items.len < 2 or items.len > 10) return error.InvalidQuestionFormat;
        const criteria = try arena.alloc(Content, items.len);
        for (items, criteria) |item, *level| {
            if (item != .string) return error.InvalidQuestionFormat;
            level.* = .{ .text = item.string };
        }
        return .scoreText(instructions, criteria);
    }

    const options = obj.get("options") orelse obj.get("criteria") orelse return .noulText(instructions);
    switch (options) {
        .array => |arr| return .choiceText(instructions, try stringChoices(arena, arr.items)),
        .object => |crit_obj| {
            const choice_entries = try arena.alloc(ChoiceEntry, crit_obj.count());
            for (choice_entries, crit_obj.keys(), crit_obj.values()) |*entry, crit_key, crit_val| {
                entry.* = .{
                    .key = crit_key,
                    .value = if (crit_val == .string) .{ .text = crit_val.string } else null,
                };
            }
            return .choiceText(instructions, .init(choice_entries));
        },
        else => return error.InvalidQuestionFormat,
    }
}

fn stringChoices(arena: std.mem.Allocator, items: []const std.json.Value) PrepareError!types.ChoiceCriteria {
    const entries = try arena.alloc(ChoiceEntry, items.len);
    for (items, entries) |item, *entry| {
        if (item != .string) return error.InvalidQuestionFormat;
        entry.* = .{ .key = item.string, .value = null };
    }
    return .init(entries);
}

/// `prepared` plus `extra`, minus any question `prepared` already asks, so one
/// call answers both. Needs the object form, since a bare category array
/// answers with one unkeyed choice.
pub fn addQuestions(
    arena: std.mem.Allocator,
    prepared: Prepared,
    extra: Questions,
) PrepareError!Prepared {
    if (prepared.kind != .questions_object) return error.InvalidQuestionFormat;
    var entries: std.ArrayList(QuestionEntry) = try .initCapacity(arena, prepared.questions.count() + extra.count());
    entries.appendSliceAssumeCapacity(prepared.questions.entries);
    for (extra.entries) |entry| {
        if (prepared.questions.has(entry.key)) continue;
        entries.appendAssumeCapacity(entry);
    }
    return .{ .questions = .init(entries.items), .kind = .questions_object };
}

pub const FormatError = error{
    OutOfMemory,
    WriteFailed,
} || types.ChoiceError;

/// The answers as JSON, shaped by what was asked: a bare string for the
/// category form, otherwise an object keyed as the request was. A question
/// that came back unanswered is null, not a confident "no".
pub fn formatAnswers(
    arena: std.mem.Allocator,
    response: types.AskResponse,
    prepared: Prepared,
) FormatError![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(arena);
    var jw: std.json.Stringify = .{ .writer = &aw.writer };

    switch (prepared.kind) {
        .single_choice => try jw.write(try response.choice(category_key, prepared.questions)),
        .questions_object => {
            try jw.beginObject();
            for (prepared.questions.entries) |entry| {
                try jw.objectField(entry.key);
                const answer = response.answer(entry.key) orelse {
                    try jw.write(null);
                    continue;
                };
                switch (answer) {
                    .noul => |n| try jw.write(n.noul),
                    .choice => |c| try jw.write(c),
                    .score => |s| try writeScore(&jw, s),
                }
            }
            try jw.endObject();
        },
    }
    return aw.written();
}

/// A score answer with its level indices swapped for the legend's labels, and
/// `level` naming the one nearest the score.
fn writeScore(jw: *std.json.Stringify, answer: types.ScoreAnswer) !void {
    try jw.beginObject();
    try jw.objectField("score");
    try jw.write(answer.score);
    if (answer.nearestLevel()) |nearest| {
        try jw.objectField("level");
        try jw.write(levelLabel(answer, nearest.key));
    }
    try jw.objectField("confidence");
    try jw.write(answer.confidence);
    try jw.objectField("probabilities");
    try jw.beginObject();
    for (answer.probabilities.entries) |entry| {
        try jw.objectField(levelLabel(answer, entry.key));
        try jw.write(entry.value);
    }
    try jw.endObject();
    try jw.endObject();
}

/// The text the legend gives level `index`, or the index itself.
fn levelLabel(answer: types.ScoreAnswer, index: []const u8) []const u8 {
    const level = answer.legend.get(index) orelse return index;
    return switch (level) {
        .text => |text| text,
        .json => index,
    };
}

fn testVocabulary(key: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, key, "isCaptcha")) return "Is there a captcha?";
    return null;
}

fn prepareFromSource(arena: std.mem.Allocator, source: []const u8) PrepareError!Prepared {
    return prepareQuestions(arena, .{ .string = source }, .{
        .vocabulary = testVocabulary,
        .category_instructions = "Which of these fits?",
    });
}

test "typesafe.json: an array is one choice question" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const prepared = try prepareFromSource(arena.allocator(),
        \\["product", "catalog"]
    );
    try std.testing.expectEqual(ResponseKind.single_choice, prepared.kind);
    try std.testing.expectEqual(1, prepared.questions.count());
    const question = prepared.questions.get(category_key).?;
    try std.testing.expectEqual(2, question.choice.criteria.count());
    try std.testing.expect(question.choice.criteria.has("product"));
}

test "typesafe.json: the category question asks what the caller worded" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const prepared = try prepareQuestions(arena.allocator(), .{ .string =
        \\["product", "catalog"]
    }, .{ .category_instructions = "What kind of page is this?" });
    try std.testing.expectEqualStrings(
        "What kind of page is this?",
        prepared.questions.get(category_key).?.choice.instructions.text,
    );
}

test "typesafe.json: an empty array has no categories, an empty object no questions" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.EmptyQuestions, prepareFromSource(arena.allocator(), "[]"));
    try std.testing.expectError(error.EmptyObject, prepareFromSource(arena.allocator(), "{}"));
}

test "typesafe.json: the vocabulary names a question the request only keys" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const prepared = try prepareFromSource(arena.allocator(),
        \\{"isCaptcha": true, "mystery": true, "spelled": "Is it spelled out?"}
    );
    try std.testing.expectEqual(ResponseKind.questions_object, prepared.kind);
    try std.testing.expectEqualStrings("Is there a captcha?", prepared.questions.get("isCaptcha").?.noul.instructions.text);
    // No vocabulary entry, so the key asks for itself.
    try std.testing.expectEqualStrings("mystery", prepared.questions.get("mystery").?.noul.instructions.text);
    try std.testing.expectEqualStrings("Is it spelled out?", prepared.questions.get("spelled").?.noul.instructions.text);
}

test "typesafe.json: an empty string needs the vocabulary to name it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const prepared = try prepareFromSource(arena.allocator(),
        \\{"isCaptcha": ""}
    );
    try std.testing.expectEqualStrings("Is there a captcha?", prepared.questions.get("isCaptcha").?.noul.instructions.text);
    try std.testing.expectError(error.InvalidQuestionFormat, prepareFromSource(arena.allocator(),
        \\{"mystery": ""}
    ));
    try std.testing.expectError(error.InvalidQuestionFormat, prepareFromSource(arena.allocator(),
        \\{"no": false}
    ));
}

test "typesafe.json: options and criteria both make a choice" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const prepared = try prepareFromSource(arena.allocator(),
        \\{"route": {"question": "Where to?", "options": ["a", "b"]},
        \\ "described": {"question": "Which?", "criteria": {"x": "the x one", "y": null}},
        \\ "listed": ["p", "q"]}
    );
    const route = prepared.questions.get("route").?;
    try std.testing.expectEqualStrings("Where to?", route.choice.instructions.text);
    try std.testing.expectEqual(2, route.choice.criteria.count());
    const described = prepared.questions.get("described").?;
    try std.testing.expectEqualStrings("the x one", described.choice.criteria.get("x").?.?.text);
    try std.testing.expectEqual(null, described.choice.criteria.get("y").?);
    // A bare array under a key keys the question by that name.
    try std.testing.expectEqualStrings("listed", prepared.questions.get("listed").?.choice.instructions.text);
}

test "typesafe.json: levels make a score, between two and ten of them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const prepared = try prepareFromSource(arena.allocator(),
        \\{"depth": {"question": "How deep?", "levels": ["shallow", "deep"]}}
    );
    const depth = prepared.questions.get("depth").?;
    try std.testing.expectEqual(2, depth.score.criteria.len);
    try std.testing.expectEqualStrings("How deep?", depth.score.instructions.text);
    try std.testing.expectError(error.InvalidQuestionFormat, prepareFromSource(arena.allocator(),
        \\{"depth": {"levels": ["only one"]}}
    ));
}

test "typesafe.json: addQuestions keeps what was asked and adds the rest" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const prepared = try prepareFromSource(aa,
        \\{"isCaptcha": true}
    );
    const extra: Questions = .init(&.{
        .{ .key = "isCaptcha", .value = .noulText("a duplicate the caller already asked") },
        .{ .key = "isBlocked", .value = .noulText("Is it blocked?") },
    });
    const both = try addQuestions(aa, prepared, extra);
    try std.testing.expectEqual(2, both.questions.count());
    // The caller's own wording wins over the one it would have been given.
    try std.testing.expectEqualStrings("Is there a captcha?", both.questions.get("isCaptcha").?.noul.instructions.text);
    try std.testing.expectEqualStrings("Is it blocked?", both.questions.get("isBlocked").?.noul.instructions.text);

    const categories = try prepareFromSource(aa,
        \\["product", "catalog"]
    );
    try std.testing.expectError(error.InvalidQuestionFormat, addQuestions(aa, categories, extra));
}

test "typesafe.json: formatAnswers shapes itself by what was asked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const categories = try prepareFromSource(aa,
        \\["product", "catalog"]
    );
    const choice_answer: types.AskResponse = .{
        .model = "jev-latest",
        .answers = .init(&.{.{ .key = category_key, .value = .{ .choice = .{
            .choice = "product",
            .probabilities = .init(&.{ .{ .key = "product", .value = 0.9 }, .{ .key = "catalog", .value = 0.1 } }),
            .confidence = 0.9,
        } } }}),
    };
    try std.testing.expectEqualStrings("\"product\"", try formatAnswers(aa, choice_answer, categories));

    const asked = try prepareFromSource(aa,
        \\{"isCaptcha": true, "unanswered": true}
    );
    const noul_answer: types.AskResponse = .{
        .model = "jev-latest",
        .answers = .init(&.{.{ .key = "isCaptcha", .value = .{ .noul = .{ .noul = 0.25 } } }}),
    };
    // A question the model skipped is null, not a confident no.
    try std.testing.expectEqualStrings(
        \\{"isCaptcha":0.25,"unanswered":null}
    , try formatAnswers(aa, noul_answer, asked));
}

test "typesafe.json: formatAnswers names a score's levels" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const asked = try prepareFromSource(aa,
        \\{"depth": {"levels": ["shallow", "deep"]}}
    );
    const response: types.AskResponse = .{
        .model = "jev-latest",
        .answers = .init(&.{.{ .key = "depth", .value = .{ .score = .{
            .score = 0.8,
            .legend = .init(&.{ .{ .key = "0", .value = .{ .text = "shallow" } }, .{ .key = "1", .value = .{ .text = "deep" } } }),
            .probabilities = .init(&.{ .{ .key = "0", .value = 0.2 }, .{ .key = "1", .value = 0.8 } }),
            .confidence = 0.8,
        } } }}),
    };
    // Level indices read back as the labels that were asked for.
    try std.testing.expectEqualStrings(
        \\{"depth":{"score":0.8,"level":"deep","confidence":0.8,"probabilities":{"shallow":0.2,"deep":0.8}}}
    , try formatAnswers(aa, response, asked));
}
