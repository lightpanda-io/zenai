//! The provider-neutral questions and answers `Pool` takes and returns.
//! `typesafe.zig` and `gemini.zig` convert them to and from each provider's
//! wire schema, so a change on either wire stops at its adapter.
//!
//! Question types:
//!
//!   - `noul`   — the probability, in [0,1], that a proposition holds.
//!   - `choice` — one option from a labelled set, with per-option probabilities.
//!   - `score`  — a continuous position on an ordered scale of levels.

const std = @import("std");
const http = @import("../http.zig");
const jsonutil = @import("../json.zig");

const StringMap = jsonutil.StringMap;

/// The state under judgement, an instruction, or a criteria description:
/// plain text, or a JSON value. TypeSafe takes JSON as is; Gemini gets it as
/// compact JSON text.
///
/// `.json` is borrowed, not copied, so it must outlive the ask.
pub const Content = union(enum) {
    text: []const u8,
    json: std.json.Value,
};

// --- Questions ---

/// Optional descriptions of what the two poles of a `noul` question mean.
pub const NoulCriteria = struct {
    true: ?Content = null,
    false: ?Content = null,
};

/// Option id -> description, or null when the id speaks for itself.
/// 2–255 options.
pub const ChoiceCriteria = StringMap(?Content);

/// The scale levels, ordered low to high. TypeSafe takes 2–10 of them,
/// Gemini up to 255.
pub const ScoreCriteria = []const Content;

pub const NoulQuestion = struct {
    instructions: Content,
    criteria: ?NoulCriteria = null,
};

pub const ChoiceQuestion = struct {
    instructions: Content,
    criteria: ChoiceCriteria,
};

pub const ScoreQuestion = struct {
    instructions: Content,
    criteria: ScoreCriteria,
};

pub const Question = union(enum) {
    noul: NoulQuestion,
    choice: ChoiceQuestion,
    score: ScoreQuestion,

    /// `.noul` from plain instruction text.
    pub fn noulText(instructions: []const u8) Question {
        return .{ .noul = .{ .instructions = .{ .text = instructions } } };
    }

    pub fn choiceText(instructions: []const u8, criteria: ChoiceCriteria) Question {
        return .{ .choice = .{ .instructions = .{ .text = instructions }, .criteria = criteria } };
    }

    pub fn scoreText(instructions: []const u8, criteria: ScoreCriteria) Question {
        return .{ .score = .{ .instructions = .{ .text = instructions }, .criteria = criteria } };
    }
};

/// Caller-chosen question id -> question. The same ids key the answers.
pub const Questions = StringMap(Question);
pub const QuestionEntry = Questions.Entry;

/// Ordered `score` levels from plain labels:
///     .criteria = levels(&.{ "Calm", "Annoyed", "Furious" })
pub fn levels(comptime labels: []const []const u8) ScoreCriteria {
    return &struct {
        const value = blk: {
            var out: [labels.len]Content = undefined;
            for (labels, 0..) |label, i| out[i] = .{ .text = label };
            break :blk out;
        };
    }.value;
}

/// `choice` options that need no description beyond their own id:
///     .criteria = choices(&.{ "billing", "delivery", "other" })
pub fn choices(comptime names: []const []const u8) ChoiceCriteria {
    return .init(&struct {
        const value = blk: {
            var out: [names.len]ChoiceCriteria.Entry = undefined;
            for (names, 0..) |name, i| out[i] = .{ .key = name, .value = null };
            break :blk out;
        };
    }.value);
}

/// `choice` options keyed by the tags in `values` (the offered subset of `E`),
/// each described by `describe`. `AskResponse.choiceEnum` maps the answer back:
///     .criteria = try enumChoices(arena, Class, &.{ .content, .captcha }, describe)
pub fn enumChoices(
    allocator: std.mem.Allocator,
    comptime E: type,
    values: []const E,
    describe: *const fn (E) []const u8,
) std.mem.Allocator.Error!ChoiceCriteria {
    const entries = try allocator.alloc(ChoiceCriteria.Entry, values.len);
    for (values, entries) |value, *entry| {
        entry.* = .{ .key = @tagName(value), .value = .{ .text = describe(value) } };
    }
    return .init(entries);
}

// --- Answers ---

/// Option id (`choice`) or level index (`score`) -> probability. Sums to ~1.
pub const Probabilities = StringMap(f64);

/// Level index -> the level's description, echoed from the question's criteria.
pub const Legend = StringMap(Content);

pub const NoulAnswer = struct {
    /// 0 (no) … 1 (yes).
    noul: f64 = 0,
};

pub const ChoiceAnswer = struct {
    /// The selected option id — a key of the question's criteria.
    choice: []const u8 = "",
    probabilities: Probabilities = .{},
    confidence: f64 = 0,
};

pub const ScoreAnswer = struct {
    /// Probability-weighted position across the level indices; can land
    /// between levels.
    score: f64 = 0,
    legend: Legend = .{},
    probabilities: Probabilities = .{},
    confidence: f64 = 0,

    /// The legend entry whose level index is nearest `score`; null when the
    /// legend is empty.
    pub fn nearestLevel(self: ScoreAnswer) ?Legend.Entry {
        var best: ?Legend.Entry = null;
        var best_distance = std.math.inf(f64);
        for (self.legend.entries) |entry| {
            const index = std.fmt.parseFloat(f64, entry.key) catch continue;
            const distance = @abs(index - self.score);
            if (distance < best_distance) {
                best = entry;
                best_distance = distance;
            }
        }
        return best;
    }
};

/// One answer. Every slice borrows the owning `Response`.
pub const Answer = union(enum) {
    noul: NoulAnswer,
    choice: ChoiceAnswer,
    score: ScoreAnswer,

    /// The `noul` probability, or null for another answer type.
    pub fn noulValue(self: Answer) ?f64 {
        return switch (self) {
            .noul => |a| a.noul,
            else => null,
        };
    }

    /// `noul` answers carry no confidence.
    pub fn confidence(self: Answer) ?f64 {
        return switch (self) {
            .noul => null,
            inline .choice, .score => |a| a.confidence,
        };
    }

    /// Probability assigned to one option id (`choice`) or level index (`score`).
    pub fn probability(self: Answer, key: []const u8) ?f64 {
        return switch (self) {
            .noul => null,
            inline .choice, .score => |a| a.probabilities.get(key),
        };
    }
};

/// Question id -> answer, keyed exactly as the request's questions.
pub const Answers = StringMap(Answer);

pub const Usage = struct {
    input_tokens: u32 = 0,
    output_tokens: u32 = 0,
};

/// Every slice here borrows the owning `Response`, which must outlive them.
pub const AskResponse = struct {
    /// The concrete model that answered, e.g. "jev-1.13.0".
    model: []const u8 = "",
    answers: Answers = .{},
    usage: Usage = .{},

    /// The answer stored under `id`, or null if the model omitted it.
    pub fn answer(self: AskResponse, id: []const u8) ?Answer {
        return self.answers.get(id);
    }

    /// The `choice` under `id`, validated against the criteria in the
    /// `questions` that were sent.
    pub fn choice(self: AskResponse, id: []const u8, questions: Questions) ChoiceError![]const u8 {
        const question = questions.get(id) orelse return error.QuestionNotAsked;
        const criteria = switch (question) {
            .choice => |c| c.criteria,
            else => return error.NotAChoice,
        };
        const found = self.answer(id) orelse return error.AnswerMissing;
        try validateChoice(found, criteria);
        return found.choice.choice;
    }

    /// `choice` under `id` as an `E`. A choice that is not a tag of `E` is
    /// `error.ChoiceNotOffered`.
    pub fn choiceEnum(
        self: AskResponse,
        id: []const u8,
        questions: Questions,
        comptime E: type,
    ) ChoiceError!E {
        const name = try self.choice(id, questions);
        return std.meta.stringToEnum(E, name) orelse error.ChoiceNotOffered;
    }

    /// Null when unanswered or not a `noul`.
    pub fn noul(self: AskResponse, id: []const u8) ?f64 {
        return (self.answer(id) orelse return null).noulValue();
    }

    /// Null when unanswered or a `noul`.
    pub fn confidence(self: AskResponse, id: []const u8) ?f64 {
        return (self.answer(id) orelse return null).confidence();
    }

    /// Null when unanswered or a `noul`.
    pub fn probability(self: AskResponse, id: []const u8, key: []const u8) ?f64 {
        return (self.answer(id) orelse return null).probability(key);
    }

    /// `model`, copied into `allocator` to outlive the response.
    pub fn dupeModel(self: AskResponse, allocator: std.mem.Allocator) std.mem.Allocator.Error![]const u8 {
        return allocator.dupe(u8, self.model);
    }
};

/// What `Pool.ask` returns. Caller must call `deinit()`.
pub const Response = http.Response(AskResponse);

/// `value` as a `Response` that takes over the memory of `wire`, a provider's
/// own response, so `value` may borrow it. `wire` must not be freed after.
pub fn adopt(wire: anytype, value: AskResponse) Response {
    return .{ .value = value, .json_buf = wire.json_buf, .parsed = .{ .arena = wire.parsed.arena, .value = value } };
}

// --- Validation ---

/// Why `validateChoice` rejected an answer.
pub const ValidateError = error{
    NotAChoice,
    ChoiceNotOffered,
    ProbabilityKeyMismatch,
    ProbabilityNotFinite,
    ProbabilityOutOfRange,
    ProbabilitySumOffByTooMuch,
    ChoiceNotArgmax,
};

/// Why `AskResponse.choice` found no usable choice.
pub const ChoiceError = ValidateError || error{
    QuestionNotAsked,
    AnswerMissing,
};

/// Allowed drift of the probabilities' sum from 1.
pub const probability_sum_tolerance = 0.02;

/// Reject a `choice` answer that strayed outside `offered`: the choice must be
/// one of them, `probabilities` must cover exactly that set with finite values
/// in [0,1] summing to 1 (within `probability_sum_tolerance`), and the choice
/// must be the argmax (ties accepted).
pub fn validateChoice(answer: Answer, offered: ChoiceCriteria) ValidateError!void {
    const a = switch (answer) {
        .choice => |c| c,
        else => return error.NotAChoice,
    };
    if (!offered.has(a.choice)) return error.ChoiceNotOffered;
    if (a.probabilities.count() != offered.count()) return error.ProbabilityKeyMismatch;

    var sum: f64 = 0;
    var chosen: f64 = 0;
    var max: f64 = 0;
    for (a.probabilities.entries) |entry| {
        if (!offered.has(entry.key)) return error.ProbabilityKeyMismatch;
        if (!std.math.isFinite(entry.value)) return error.ProbabilityNotFinite;
        if (entry.value < 0 or entry.value > 1) return error.ProbabilityOutOfRange;
        sum += entry.value;
        if (entry.value > max) max = entry.value;
        if (std.mem.eql(u8, entry.key, a.choice)) chosen = entry.value;
    }
    if (@abs(sum - 1) > probability_sum_tolerance) return error.ProbabilitySumOffByTooMuch;
    if (chosen < max) return error.ChoiceNotArgmax;
}

test "validateChoice accepts a well-formed answer, ties included" {
    const offered = choices(&.{ "a", "b", "c" });
    try validateChoice(.{ .choice = .{ .choice = "b", .probabilities = .init(&.{
        .{ .key = "a", .value = 0.2 },
        .{ .key = "b", .value = 0.7 },
        .{ .key = "c", .value = 0.1 },
    }) } }, offered);
    try validateChoice(.{ .choice = .{ .choice = "a", .probabilities = .init(&.{
        .{ .key = "a", .value = 0.45 },
        .{ .key = "b", .value = 0.1 },
        .{ .key = "c", .value = 0.45 },
    }) } }, offered);
}

test "validateChoice rejects every way an answer can stray" {
    const offered = choices(&.{ "a", "b" });
    const cases = [_]struct { expected: ValidateError, answer: Answer }{
        .{ .expected = error.NotAChoice, .answer = .{ .noul = .{ .noul = 0.5 } } },
        .{ .expected = error.ChoiceNotOffered, .answer = .{ .choice = .{ .choice = "z", .probabilities = .init(&.{
            .{ .key = "a", .value = 0.5 },
            .{ .key = "b", .value = 0.5 },
        }) } } },
        // Too few keys.
        .{ .expected = error.ProbabilityKeyMismatch, .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
            .{ .key = "a", .value = 1 },
        }) } } },
        // Right count, wrong key.
        .{ .expected = error.ProbabilityKeyMismatch, .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
            .{ .key = "a", .value = 0.5 },
            .{ .key = "z", .value = 0.5 },
        }) } } },
        .{ .expected = error.ProbabilityNotFinite, .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
            .{ .key = "a", .value = std.math.nan(f64) },
            .{ .key = "b", .value = 0.5 },
        }) } } },
        .{ .expected = error.ProbabilityOutOfRange, .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
            .{ .key = "a", .value = 1.5 },
            .{ .key = "b", .value = -0.5 },
        }) } } },
        .{ .expected = error.ProbabilitySumOffByTooMuch, .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
            .{ .key = "a", .value = 0.5 },
            .{ .key = "b", .value = 0.4 },
        }) } } },
        .{ .expected = error.ChoiceNotArgmax, .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
            .{ .key = "a", .value = 0.3 },
            .{ .key = "b", .value = 0.7 },
        }) } } },
    };
    for (cases) |case| {
        try std.testing.expectError(case.expected, validateChoice(case.answer, offered));
    }
}

test "choice: validated against the criteria the question carried" {
    const questions: Questions = .init(&.{
        .{ .key = "route", .value = .choiceText("Where to?", choices(&.{ "a", "b" })) },
        .{ .key = "urgent", .value = .noulText("Is it urgent?") },
    });
    const response: AskResponse = .{ .answers = .init(&.{
        .{ .key = "route", .value = .{ .choice = .{ .choice = "b", .probabilities = .init(&.{
            .{ .key = "a", .value = 0.1 },
            .{ .key = "b", .value = 0.9 },
        }) } } },
        .{ .key = "strayed", .value = .{ .choice = .{ .choice = "z", .probabilities = .init(&.{
            .{ .key = "z", .value = 1 },
        }) } } },
    }) };

    try std.testing.expectEqualStrings("b", try response.choice("route", questions));
    try std.testing.expectError(error.QuestionNotAsked, response.choice("strayed", questions));
    try std.testing.expectError(error.NotAChoice, response.choice("urgent", questions));
    try std.testing.expectError(error.AnswerMissing, response.choice("missing", .init(&.{
        .{ .key = "missing", .value = .choiceText("?", choices(&.{"a"})) },
    })));
}

const Fruit = enum { apple, pear, quince };

fn describeFruit(f: Fruit) []const u8 {
    return switch (f) {
        .apple => "A pome, red or green.",
        .pear => "A pome, narrower at the stem.",
        .quince => "A pome nobody eats raw.",
    };
}

test "enumChoices and choiceEnum round-trip an enum subset" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    const criteria = try enumChoices(arena.allocator(), Fruit, &.{ .apple, .quince }, describeFruit);
    try std.testing.expectEqual(2, criteria.count());
    try std.testing.expect(!criteria.has("pear"));
    try std.testing.expectEqualStrings("A pome nobody eats raw.", criteria.get("quince").?.?.text);

    const questions: Questions = .init(&.{.{ .key = "fruit", .value = .choiceText("Which one?", criteria) }});
    const response: AskResponse = .{ .answers = .init(&.{
        .{ .key = "fruit", .value = .{ .choice = .{ .choice = "quince", .probabilities = .init(&.{
            .{ .key = "apple", .value = 0.2 },
            .{ .key = "quince", .value = 0.8 },
        }) } } },
    }) };
    try std.testing.expectEqual(Fruit.quince, try response.choiceEnum("fruit", questions, Fruit));
    try std.testing.expectError(error.ChoiceNotOffered, response.choiceEnum("fruit", questions, enum { plum }));
}

test "the answer accessors return null instead of unwrapping a missing answer" {
    const response: AskResponse = .{ .answers = .init(&.{
        .{ .key = "urgent", .value = .{ .noul = .{ .noul = 0.82 } } },
        .{ .key = "route", .value = .{ .choice = .{
            .choice = "a",
            .confidence = 0.71,
            .probabilities = .init(&.{.{ .key = "a", .value = 1 }}),
        } } },
    }) };

    try std.testing.expectEqual(0.82, response.noul("urgent").?);
    try std.testing.expectEqual(0.71, response.confidence("route").?);
    try std.testing.expectEqual(1, response.probability("route", "a").?);

    try std.testing.expectEqual(null, response.noul("absent"));
    try std.testing.expectEqual(null, response.noul("route"));
    try std.testing.expectEqual(null, response.confidence("urgent"));
    try std.testing.expectEqual(null, response.probability("urgent", "a"));
}

test "nearestLevel picks the legend entry closest to the score" {
    var answer: ScoreAnswer = .{
        .score = 1.64,
        .legend = .init(&.{
            .{ .key = "0", .value = .{ .text = "none" } },
            .{ .key = "1", .value = .{ .text = "some" } },
            .{ .key = "2", .value = .{ .text = "lots" } },
        }),
    };
    try std.testing.expectEqualStrings("lots", answer.nearestLevel().?.value.text);
    answer.score = 0.4;
    try std.testing.expectEqualStrings("0", answer.nearestLevel().?.key);
    answer.legend = .{};
    try std.testing.expect(answer.nearestLevel() == null);
}
