//! TypeSafe System One request/response shapes. https://docs.typesafe.ai
//!
//! One POST carries the `state` under judgement plus a map of typed questions
//! about it, keyed by ids the caller chooses; the response carries one answer
//! per id under the same keys. Three question types:
//!
//!   - `noul`   — the probability, in [0,1], that a proposition holds.
//!   - `choice` — one option from a labelled set, with per-option probabilities.
//!   - `score`  — a continuous position on an ordered 2–10 level scale.

const std = @import("std");
const jsonutil = @import("../json.zig");

/// A JSON object with runtime keys; see `json.StringMap`.
pub const StringMap = jsonutil.StringMap;

/// The flagship alias. Concrete versions (`jev-1.13.0`) and `jev-preview` also
/// work; `Client.listModels` enumerates them.
pub const default_model = "jev-latest";

/// Jev's context budget: 64k tokens per request, of which `state` plus the
/// single longest question must fit in 32k. Overruns come back as HTTP 422.
pub const max_context_tokens = 64_000;
pub const max_state_tokens = 32_000;

/// A field the API accepts as a string, an object, or an array — `state`,
/// `instructions`, and every criteria description.
///
/// `.text` covers the common case without dragging callers through
/// `std.json.Value`; `.json` passes structured input straight through,
/// borrowed rather than copied, so it must outlive the request.
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

    pub fn jsonParseFromValue(
        _: std.mem.Allocator,
        source: std.json.Value,
        _: std.json.ParseOptions,
    ) std.json.ParseFromValueError!Content {
        return switch (source) {
            .string => |s| .{ .text = s },
            else => .{ .json = source },
        };
    }

    /// The `.text` payload, or null for a structured value.
    pub fn asText(self: Content) ?[]const u8 {
        return switch (self) {
            .text => |s| s,
            .json => null,
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
    /// Required by the API.
    criteria: ChoiceCriteria,
};

pub const ScoreQuestion = struct {
    instructions: Content,
    /// Required by the API; 2–10 ordered levels.
    criteria: ScoreCriteria,
};

/// One question. Serializes flat — `{"type": <tag>, "instructions": …,
/// "criteria": …}` — with a separate payload per type, so a `choice` without
/// criteria or a `score` with a criteria map cannot be built.
pub const Question = union(enum) {
    noul: NoulQuestion,
    choice: ChoiceQuestion,
    score: ScoreQuestion,

    pub fn jsonStringify(self: Question, jw: *std.json.Stringify) !void {
        return writeTagged(self, jw);
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

/// Doubles as the request body: `Client.ask` fills in `state` and `questions`.
/// Declaration order is wire order.
pub const AskOptions = struct {
    state: Content = .{ .text = "" },
    model: []const u8 = default_model,
    questions: Questions = .{},
};

// --- Answers ---

/// Option id (`choice`) or level index (`score`) -> probability. Sums to ~1.
pub const Probabilities = StringMap(f64);

/// Level index -> the level's description, echoed back from the question's
/// criteria — `Content`, because a level may have been an object or array.
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
};

/// One answer, discriminated by the wire's `"type"`. Every slice borrows the
/// owning `Response`.
pub const Answer = union(enum) {
    noul: NoulAnswer,
    choice: ChoiceAnswer,
    score: ScoreAnswer,

    /// Buffers the object into a `std.json.Value` first: `"type"` may arrive in
    /// any position and a JSON scanner cannot rewind.
    pub fn jsonParse(
        allocator: std.mem.Allocator,
        source: anytype,
        options: std.json.ParseOptions,
    ) std.json.ParseError(@TypeOf(source.*))!Answer {
        const value = try std.json.innerParse(std.json.Value, allocator, source, options);
        return jsonParseFromValue(allocator, value, options);
    }

    pub fn jsonParseFromValue(
        allocator: std.mem.Allocator,
        source: std.json.Value,
        options: std.json.ParseOptions,
    ) std.json.ParseFromValueError!Answer {
        const object = switch (source) {
            .object => |o| o,
            else => return error.UnexpectedToken,
        };
        const tag = switch (object.get("type") orelse return error.MissingField) {
            .string => |s| s,
            else => return error.UnexpectedToken,
        };
        // The payload structs do not model `type` itself, and the API may add
        // fields, so the discriminated re-parse always ignores unknown keys.
        var payload_options = options;
        payload_options.ignore_unknown_fields = true;
        inline for (@typeInfo(Answer).@"union".fields) |field| {
            if (std.mem.eql(u8, field.name, tag)) {
                return @unionInit(
                    Answer,
                    field.name,
                    try std.json.innerParseFromValue(field.type, allocator, source, payload_options),
                );
            }
        }
        return error.UnknownField;
    }

    pub fn jsonStringify(self: Answer, jw: *std.json.Stringify) !void {
        return writeTagged(self, jw);
    }

    /// The `noul` probability, or null for another answer type.
    pub fn noulValue(self: Answer) ?f64 {
        return switch (self) {
            .noul => |a| a.noul,
            else => null,
        };
    }

    /// The selected option id, or null for another answer type.
    pub fn choiceValue(self: Answer) ?[]const u8 {
        return switch (self) {
            .choice => |a| a.choice,
            else => null,
        };
    }

    /// The score, or null for another answer type.
    pub fn scoreValue(self: Answer) ?f64 {
        return switch (self) {
            .score => |a| a.score,
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

pub const ChoiceError = error{
    NotAChoice,
    ChoiceNotOffered,
    ProbabilityKeyMismatch,
    ProbabilityNotFinite,
    ProbabilityOutOfRange,
    ProbabilitySumOffByTooMuch,
    ChoiceNotArgmax,
};

/// How far the probabilities may drift from summing to 1 before the answer is
/// rejected.
pub const probability_sum_tolerance = 0.02;

/// Reject a `choice` answer that strayed outside the option set it was given,
/// before anything acts on it. `offered` is the exact criteria list sent for
/// that question: the choice must be one of them, `probabilities` must cover
/// exactly that set with finite values in [0,1] summing to 1 (within
/// `probability_sum_tolerance`), and the chosen option must be the argmax —
/// `>=`, so ties are accepted.
pub fn validateChoice(answer: Answer, offered: []const []const u8) ChoiceError!void {
    const a = switch (answer) {
        .choice => |c| c,
        else => return error.NotAChoice,
    };
    if (indexOfString(offered, a.choice) == null) return error.ChoiceNotOffered;
    if (a.probabilities.count() != offered.len) return error.ProbabilityKeyMismatch;

    var sum: f64 = 0;
    var chosen: f64 = 0;
    var max: f64 = 0;
    for (a.probabilities.entries) |entry| {
        if (indexOfString(offered, entry.key) == null) return error.ProbabilityKeyMismatch;
        if (!std.math.isFinite(entry.value)) return error.ProbabilityNotFinite;
        if (entry.value < 0 or entry.value > 1) return error.ProbabilityOutOfRange;
        sum += entry.value;
        if (entry.value > max) max = entry.value;
        if (std.mem.eql(u8, entry.key, a.choice)) chosen = entry.value;
    }
    if (@abs(sum - 1) > probability_sum_tolerance) return error.ProbabilitySumOffByTooMuch;
    if (chosen < max) return error.ChoiceNotArgmax;
}

// --- Internal ---

fn indexOfString(haystack: []const []const u8, needle: []const u8) ?usize {
    for (haystack, 0..) |candidate, i| {
        if (std.mem.eql(u8, candidate, needle)) return i;
    }
    return null;
}

/// Emit a tagged union as one flat object: the active tag under `"type"`, then
/// the payload struct's own fields, honouring `emit_null_optional_fields` the
/// way std treats a plain struct.
fn writeTagged(value: anytype, jw: *std.json.Stringify) !void {
    try jw.beginObject();
    try jw.objectField("type");
    try jw.write(@tagName(value));
    switch (value) {
        inline else => |payload| {
            inline for (@typeInfo(@TypeOf(payload)).@"struct".fields) |field| {
                const field_value = @field(payload, field.name);
                if (comptime @typeInfo(field.type) == .optional) {
                    if (field_value) |unwrapped| {
                        try jw.objectField(field.name);
                        try jw.write(unwrapped);
                    } else if (jw.options.emit_null_optional_fields) {
                        try jw.objectField(field.name);
                        try jw.write(null);
                    }
                } else {
                    try jw.objectField(field.name);
                    try jw.write(field_value);
                }
            }
        },
    }
    try jw.endObject();
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
    try std.testing.expectEqual(@as(?[]const u8, null), complaint.choiceValue());

    const topic = parsed.value.answer("topic").?;
    try std.testing.expectEqualStrings("billing", topic.choiceValue().?);
    try std.testing.expectApproxEqAbs(@as(f64, 0.88), topic.probability("billing").?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.81), topic.confidence().?, 1e-12);

    const anger = parsed.value.answer("anger").?;
    switch (anger) {
        .score => |s| {
            try std.testing.expectApproxEqAbs(@as(f64, 1.05), s.score, 1e-12);
            try std.testing.expectEqualStrings("Annoyed", s.legend.get("1").?.asText().?);
            // An integer token still lands in an f64 probability.
            try std.testing.expectEqual(@as(?f64, 0), s.probabilities.get("0"));
        },
        else => return error.TestUnexpectedResult,
    }

    try std.testing.expect(parsed.value.answer("no_such_question") == null);
}

test "AskOptions stringifies the System One request body" {
    const request: AskOptions = .{
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

    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(request, .{ .emit_null_optional_fields = false }, &buf.writer);
    try std.testing.expectEqualStrings(
        \\{"state":"The pizza arrived cold.","model":"jev-latest","questions":{"is_complaint":{"type":"noul","instructions":"Is the customer complaining?"},"topic":{"type":"choice","instructions":"What is this about?","criteria":{"billing":null,"delivery":"Anything about shipping"}},"anger":{"type":"score","instructions":"How angry?","criteria":["Calm","Furious"]}}}
    , buf.written());
}

test "AskOptions passes a structured state through untouched" {
    const doc = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
        \\{"ticket":{"id":7,"lines":["cold","late"]}}
    , .{});
    defer doc.deinit();

    const request: AskOptions = .{
        .state = .{ .json = doc.value },
        .questions = .init(&.{
            .{ .key = "complaint", .value = .noulText("Is this a complaint?") },
        }),
    };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(request, .{ .emit_null_optional_fields = false }, &buf.writer);
    try std.testing.expectEqualStrings(
        \\{"state":{"ticket":{"id":7,"lines":["cold","late"]}},"model":"jev-latest","questions":{"complaint":{"type":"noul","instructions":"Is this a complaint?"}}}
    , buf.written());
}

test "validateChoice accepts a well-formed answer" {
    const offered = [_][]const u8{ "a", "b", "c" };
    const answer: Answer = .{ .choice = .{
        .choice = "b",
        .probabilities = .init(&.{
            .{ .key = "a", .value = 0.2 },
            .{ .key = "b", .value = 0.7 },
            .{ .key = "c", .value = 0.1 },
        }),
        .confidence = 0.6,
    } };
    try validateChoice(answer, &offered);
}

test "validateChoice rejects every way an answer can stray" {
    const offered = [_][]const u8{ "a", "b" };
    const cases = [_]struct { expected: ChoiceError, answer: Answer }{
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
        try std.testing.expectError(case.expected, validateChoice(case.answer, &offered));
    }
}

test "validateChoice accepts a tie on the chosen option" {
    const offered = [_][]const u8{ "a", "b" };
    const answer: Answer = .{ .choice = .{ .choice = "a", .probabilities = .init(&.{
        .{ .key = "a", .value = 0.5 },
        .{ .key = "b", .value = 0.5 },
    }) } };
    try validateChoice(answer, &offered);
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
    const request: AskOptions = .{
        .state = .{ .text = "The pizza arrived cold." },
        .questions = .init(&questions),
    };

    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(request, .{ .emit_null_optional_fields = false }, &buf.writer);
    try std.testing.expectEqualStrings(
        \\{"state":"The pizza arrived cold.","model":"jev-latest","questions":{"is_complaint":{"type":"noul","instructions":"Is the customer complaining?"},"topic":{"type":"choice","instructions":"What is this message about?","criteria":{"billing":null,"delivery":null,"other":null}},"anger":{"type":"score","instructions":"How angry is the customer?","criteria":["Calm","Annoyed","Furious"]}}}
    , buf.written());
}
