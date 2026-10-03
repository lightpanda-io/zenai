//! TypeSafe System One request/response shapes. https://docs.typesafe.ai
//!
//! Question types:
//!
//!   - `noul`   — the probability, in [0,1], that a proposition holds.
//!   - `choice` — one option from a labelled set, with per-option probabilities.
//!   - `score`  — a continuous position on an ordered 2–10 level scale.

const std = @import("std");
const jsonutil = @import("../json.zig");

const StringMap = jsonutil.StringMap;

/// `Client.listModels` lists the concrete versions and other aliases.
pub const default_model = "jev-latest";

/// A field the API accepts as a string, an object, or an array — `state`,
/// `instructions`, and every criteria description.
///
/// `.json` is borrowed, not copied, so it must outlive the request.
pub const Content = union(enum) {
    text: []const u8,
    json: std.json.Value,

    pub const jsonStringify = jsonutil.PayloadUnionMethods(@This()).jsonStringify;

    pub fn jsonParse(
        allocator: std.mem.Allocator,
        source: anytype,
        options: std.json.ParseOptions,
    ) std.json.ParseError(@TypeOf(source.*))!Content {
        return switch (try source.peekNextTokenType()) {
            .string => .{ .text = try std.json.innerParse([]const u8, allocator, source, options) },
            else => .{ .json = try std.json.innerParse(std.json.Value, allocator, source, options) },
        };
    }
};

// --- Questions ---

/// Optional descriptions of what the two poles of a `noul` question mean.
pub const NoulCriteria = struct {
    true: ?Content = null,
    false: ?Content = null,
};

/// Option id -> description, or null when the id speaks for itself.
/// At most 255 options.
pub const ChoiceCriteria = StringMap(?Content);

/// The scale levels, ordered low to high. 2–10 entries.
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

/// One question. Serializes flat: `{"type": <tag>, "instructions": …,
/// "criteria": …}`.
pub const Question = union(enum) {
    noul: NoulQuestion,
    choice: ChoiceQuestion,
    score: ScoreQuestion,

    /// Honours `emit_null_optional_fields` the way std does for a plain struct.
    pub fn jsonStringify(self: Question, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        try jw.objectField("type");
        try jw.write(@tagName(self));
        switch (self) {
            inline else => |payload| {
                const info = @typeInfo(@TypeOf(payload)).@"struct";
                inline for (info.field_names, info.field_types) |name, T| {
                    const v = @field(payload, name);
                    const skip = if (@typeInfo(T) == .optional) v == null and !jw.options.emit_null_optional_fields else false;
                    if (!skip) {
                        try jw.objectField(name);
                        try jw.write(v);
                    }
                }
            },
        }
        try jw.endObject();
    }

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

pub const AskOptions = struct {
    model: []const u8 = default_model,
};

/// The request body `Client.ask` sends. Declaration order is wire order.
pub const AskRequest = struct {
    state: Content,
    model: []const u8 = default_model,
    questions: Questions,
};

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

/// One answer, discriminated by the wire's `"type"`. Every slice borrows the
/// owning `Response`.
pub const Answer = union(enum) {
    noul: NoulAnswer,
    choice: ChoiceAnswer,
    score: ScoreAnswer,

    /// Every answer type's fields, so `"type"` may arrive in any position.
    const Wire = struct {
        type: std.meta.Tag(Answer),
        noul: f64 = 0,
        choice: []const u8 = "",
        score: f64 = 0,
        legend: Legend = .{},
        probabilities: Probabilities = .{},
        confidence: f64 = 0,
    };

    pub fn jsonParse(
        allocator: std.mem.Allocator,
        source: anytype,
        options: std.json.ParseOptions,
    ) std.json.ParseError(@TypeOf(source.*))!Answer {
        const w = try std.json.innerParse(Wire, allocator, source, options);
        switch (w.type) {
            inline else => |tag| {
                const Payload = @FieldType(Answer, @tagName(tag));
                var payload: Payload = .{};
                inline for (@typeInfo(Payload).@"struct".field_names) |name| @field(payload, name) = @field(w, name);
                return @unionInit(Answer, @tagName(tag), payload);
            },
        }
    }

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

pub const Model = struct {
    /// Model id or alias, e.g. "jev-1.13.0", "jev-latest".
    name: []const u8 = "",
    description: ?[]const u8 = null,
    /// ISO 8601 date.
    release_date: ?[]const u8 = null,
};

pub const ListModelsResponse = struct {
    models: []const Model = &.{},
};

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

/// Stringify `value` with the options the client sends it with.
fn expectJson(value: anytype, expected: []const u8) !void {
    const out = try jsonutil.stringifyAlloc(std.testing.allocator, value, .{ .emit_null_optional_fields = false });
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings(expected, out);
}

test "AskResponse parses a System One fixture" {
    const fixture =
        \\{
        \\  "model": "jev-1.13.0",
        \\  "request_id": "req_01HZXV",
        \\  "answers": {
        \\    "is_complaint": {"type": "noul", "noul": 0.95},
        \\    "topic": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.88, "delivery": 0.1, "other": 0.02}, "confidence": 0.81},
        \\    "anger": {"type": "score", "score": 1.05, "legend": {"0": "Calm", "1": "Annoyed", "2": "Furious"}, "probabilities": {"0": 0, "1": 0.9, "2": 0.1}, "confidence": 0.92, "rationale": "ignored"}
        \\  },
        \\  "usage": {"input_tokens": 296, "output_tokens": 20, "cache_read_tokens": 0}
        \\}
    ;
    const parsed = try std.json.parseFromSlice(AskResponse, std.testing.allocator, fixture, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    try std.testing.expectEqualStrings("jev-1.13.0", parsed.value.model);
    try std.testing.expectEqual(@as(u32, 296), parsed.value.usage.input_tokens);

    const complaint = parsed.value.answer("is_complaint").?;
    try std.testing.expectApproxEqAbs(@as(f64, 0.95), complaint.noulValue().?, 1e-12);
    try std.testing.expectEqual(@as(?f64, null), complaint.confidence());

    const topic = parsed.value.answer("topic").?;
    try std.testing.expectEqualStrings("billing", topic.choice.choice);
    try std.testing.expectApproxEqAbs(@as(f64, 0.88), topic.probability("billing").?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.81), topic.confidence().?, 1e-12);

    const anger = parsed.value.answer("anger").?;
    switch (anger) {
        .score => |s| {
            try std.testing.expectApproxEqAbs(@as(f64, 1.05), s.score, 1e-12);
            try std.testing.expectEqualStrings("Annoyed", s.legend.get("1").?.text);
            // An integer token still lands in an f64 probability.
            try std.testing.expectEqual(@as(?f64, 0), s.probabilities.get("0"));
        },
        else => return error.TestUnexpectedResult,
    }

    try std.testing.expect(parsed.value.answer("no_such_question") == null);
}

test "AskRequest stringifies the System One request body" {
    const request: AskRequest = .{
        .state = .{ .text = "The pizza arrived cold." },
        .questions = .init(&.{
            .{ .key = "is_complaint", .value = .noulText("Is the customer complaining?") },
            .{ .key = "topic", .value = .{ .choice = .{
                .instructions = .{ .text = "What is this about?" },
                .criteria = .init(&.{
                    .{ .key = "billing", .value = null },
                    .{ .key = "delivery", .value = .{ .text = "Anything about shipping" } },
                }),
            } } },
            .{ .key = "anger", .value = .scoreText("How angry?", levels(&.{ "Calm", "Furious" })) },
        }),
    };

    try expectJson(request,
        \\{"state":"The pizza arrived cold.","model":"jev-latest","questions":{"is_complaint":{"type":"noul","instructions":"Is the customer complaining?"},"topic":{"type":"choice","instructions":"What is this about?","criteria":{"billing":null,"delivery":"Anything about shipping"}},"anger":{"type":"score","instructions":"How angry?","criteria":["Calm","Furious"]}}}
    );
}

test "AskRequest passes a structured state through untouched" {
    const doc = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
        \\{"ticket":{"id":7,"lines":["cold","late"]}}
    , .{});
    defer doc.deinit();

    const request: AskRequest = .{
        .state = .{ .json = doc.value },
        .questions = .init(&.{
            .{ .key = "complaint", .value = .noulText("Is this a complaint?") },
        }),
    };
    try expectJson(request,
        \\{"state":{"ticket":{"id":7,"lines":["cold","late"]}},"model":"jev-latest","questions":{"complaint":{"type":"noul","instructions":"Is this a complaint?"}}}
    );
}

test "validateChoice accepts a well-formed answer" {
    const offered = choices(&.{ "a", "b", "c" });
    const answer: Answer = .{ .choice = .{
        .choice = "b",
        .probabilities = .init(&.{
            .{ .key = "a", .value = 0.2 },
            .{ .key = "b", .value = 0.7 },
            .{ .key = "c", .value = 0.1 },
        }),
        .confidence = 0.6,
    } };
    try validateChoice(answer, offered);
}

test "validateChoice rejects every way an answer can stray" {
    const offered = choices(&.{ "a", "b" });
    const cases = [_]struct { expected: ValidateError, answer: Answer }{
        .{
            .expected = error.NotAChoice,
            .answer = .{ .noul = .{ .noul = 0.5 } },
        },
        .{
            .expected = error.ChoiceNotOffered,
            .answer = .{ .choice = .{ .choice = "z", .probabilities = .init(&.{
                .{ .key = "a", .value = 0.5 },
                .{ .key = "b", .value = 0.5 },
            }) } },
        },
        .{
            // Too few keys.
            .expected = error.ProbabilityKeyMismatch,
            .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
                .{ .key = "a", .value = 1 },
            }) } },
        },
        .{
            // Right count, wrong key.
            .expected = error.ProbabilityKeyMismatch,
            .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
                .{ .key = "a", .value = 0.5 },
                .{ .key = "z", .value = 0.5 },
            }) } },
        },
        .{
            .expected = error.ProbabilityNotFinite,
            .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
                .{ .key = "a", .value = std.math.nan(f64) },
                .{ .key = "b", .value = 0.5 },
            }) } },
        },
        .{
            .expected = error.ProbabilityOutOfRange,
            .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
                .{ .key = "a", .value = 1.5 },
                .{ .key = "b", .value = -0.5 },
            }) } },
        },
        .{
            .expected = error.ProbabilitySumOffByTooMuch,
            .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
                .{ .key = "a", .value = 0.5 },
                .{ .key = "b", .value = 0.4 },
            }) } },
        },
        .{
            .expected = error.ChoiceNotArgmax,
            .answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
                .{ .key = "a", .value = 0.3 },
                .{ .key = "b", .value = 0.7 },
            }) } },
        },
    };
    for (cases) |case| {
        try std.testing.expectError(case.expected, validateChoice(case.answer, offered));
    }
}

test "validateChoice accepts a tie on the chosen option" {
    const offered = choices(&.{ "a", "b" });
    const answer: Answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
        .{ .key = "a", .value = 0.5 },
        .{ .key = "b", .value = 0.5 },
    }) } };
    try validateChoice(answer, offered);
}

test "choices and choiceText build the README's question list" {
    const questions = [_]QuestionEntry{
        .{ .key = "is_complaint", .value = .noulText("Is the customer complaining?") },
        .{ .key = "topic", .value = .choiceText(
            "What is this message about?",
            choices(&.{ "billing", "delivery", "other" }),
        ) },
        .{ .key = "anger", .value = .scoreText(
            "How angry is the customer?",
            levels(&.{ "Calm", "Annoyed", "Furious" }),
        ) },
    };
    const request: AskRequest = .{
        .state = .{ .text = "The pizza arrived cold." },
        .questions = .init(&questions),
    };

    try expectJson(request,
        \\{"state":"The pizza arrived cold.","model":"jev-latest","questions":{"is_complaint":{"type":"noul","instructions":"Is the customer complaining?"},"topic":{"type":"choice","instructions":"What is this message about?","criteria":{"billing":null,"delivery":null,"other":null}},"anger":{"type":"score","instructions":"How angry is the customer?","criteria":["Calm","Annoyed","Furious"]}}}
    );
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

test "enumChoices keys the criteria by tag and carries a description" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    const criteria = try enumChoices(arena.allocator(), Fruit, &.{ .apple, .quince }, describeFruit);

    try std.testing.expectEqual(2, criteria.count());
    try std.testing.expect(criteria.has("apple"));
    try std.testing.expect(!criteria.has("pear"));
    try std.testing.expectEqualStrings("A pome nobody eats raw.", criteria.get("quince").?.?.text);
}

test "choiceEnum maps a validated answer back to the enum" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    const questions: Questions = .init(&.{
        .{ .key = "fruit", .value = .choiceText(
            "Which one?",
            try enumChoices(arena.allocator(), Fruit, &.{ .apple, .quince }, describeFruit),
        ) },
    });
    const response: AskResponse = .{ .answers = .init(&.{
        .{ .key = "fruit", .value = .{ .choice = .{ .choice = "quince", .probabilities = .init(&.{
            .{ .key = "apple", .value = 0.2 },
            .{ .key = "quince", .value = 0.8 },
        }) } } },
    }) };

    try std.testing.expectEqual(Fruit.quince, try response.choiceEnum("fruit", questions, Fruit));

    // A choice that is not one of the enum's tags.
    const Other = enum { plum };
    try std.testing.expectError(
        error.ChoiceNotOffered,
        response.choiceEnum("fruit", questions, Other),
    );
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

    // Unanswered, or the wrong answer type.
    try std.testing.expectEqual(null, response.noul("absent"));
    try std.testing.expectEqual(null, response.confidence("absent"));
    try std.testing.expectEqual(null, response.probability("absent", "a"));
    try std.testing.expectEqual(null, response.noul("route"));
    try std.testing.expectEqual(null, response.confidence("urgent"));
    try std.testing.expectEqual(null, response.probability("urgent", "a"));
}

test "dupeModel outlives the response the model borrows" {
    const response: AskResponse = .{ .model = "jev-1.13.0" };
    const owned = try response.dupeModel(std.testing.allocator);
    defer std.testing.allocator.free(owned);
    try std.testing.expectEqualStrings("jev-1.13.0", owned);
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
