//! Gemini Decisions API (`POST /v1beta/decisions`) request/response shapes.
//!
//! One shared `input` is evaluated once against a map of typed questions, and
//! each question gets a calibrated answer back under the same id:
//!
//!   - `noul`   — the probability, in [0.01, 0.99], that a proposition holds.
//!   - `choice` — one option from a labelled set, with per-option probabilities.
//!   - `score`  — an expected position on an ordered 2–255 level scale.
//!
//! Developer API only.

const std = @import("std");
const jsonutil = @import("../json.zig");

const StringMap = jsonutil.StringMap;

/// The model that serves `createDecision`; generation models are rejected.
pub const default_model = "gemini-decision-1";

// --- Input ---

pub const PartType = enum { text, image, audio, document, video };

/// One input content block. Set `text` for `.text`; for media, either inline
/// base64 `data` with its `mime_type`, or a `uri` (e.g. a Files API URI).
pub const InputPart = struct {
    type: PartType,
    text: ?[]const u8 = null,
    data: ?[]const u8 = null,
    mime_type: ?[]const u8 = null,
    uri: ?[]const u8 = null,
};

/// The shared input every question is evaluated against: plain text, or a
/// list of content blocks.
pub const Input = union(enum) {
    text: []const u8,
    parts: []const InputPart,

    pub const jsonStringify = jsonutil.PayloadUnionMethods(@This()).jsonStringify;
};

// --- Questions ---

/// Optional descriptions of what the `true` and `false` outcomes mean.
pub const NoulCriteria = struct {
    true_description: ?[]const u8 = null,
    false_description: ?[]const u8 = null,
};

/// Option id -> description ("" when the id speaks for itself).
/// 2–255 options.
pub const ChoiceCriteria = StringMap([]const u8);

/// The score levels, ordered low to high; level `i` has value `i`.
/// 2–255 entries.
pub const ScoreCriteria = []const []const u8;

pub const NoulQuestion = struct {
    instructions: []const u8,
    criteria: ?NoulCriteria = null,
};

pub const ChoiceQuestion = struct {
    instructions: []const u8,
    criteria: ChoiceCriteria,
};

pub const ScoreQuestion = struct {
    instructions: []const u8,
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

    pub fn noulText(instructions: []const u8) Question {
        return .{ .noul = .{ .instructions = instructions } };
    }

    pub fn choiceText(instructions: []const u8, criteria: ChoiceCriteria) Question {
        return .{ .choice = .{ .instructions = instructions, .criteria = criteria } };
    }

    pub fn scoreText(instructions: []const u8, criteria: ScoreCriteria) Question {
        return .{ .score = .{ .instructions = instructions, .criteria = criteria } };
    }
};

/// Caller-chosen question id -> question. The same ids key the answers.
pub const Questions = StringMap(Question);
pub const QuestionEntry = Questions.Entry;

/// `choice` options that need no description beyond their own id:
///     .criteria = choices(&.{ "billing", "delivery", "other" })
pub fn choices(comptime names: []const []const u8) ChoiceCriteria {
    return .init(&struct {
        const value = blk: {
            var out: [names.len]ChoiceCriteria.Entry = undefined;
            for (names, 0..) |name, i| out[i] = .{ .key = name, .value = "" };
            break :blk out;
        };
    }.value);
}

/// The request body `Client.createDecision` sends.
pub const CreateDecisionRequest = struct {
    input: Input,
    /// `models/{model}`.
    model: []const u8,
    questions: Questions,
};

// --- Answers ---

/// Option id (`choice`) or level index (`score`) -> probability. Sums to ~1.
pub const Probabilities = StringMap(f64);

/// Level index -> the level's description, echoed from the question's criteria.
pub const Legend = StringMap([]const u8);

pub const NoulAnswer = struct {
    /// Calibrated P(true), in [0.01, 0.99].
    noul: f64 = 0,
    /// `max(noul, 1 - noul)`.
    confidence: f64 = 0,
};

pub const ChoiceAnswer = struct {
    /// The option id with the highest probability.
    choice: []const u8 = "",
    probabilities: Probabilities = .{},
    /// The probability of `choice`.
    confidence: f64 = 0,
};

pub const ScoreAnswer = struct {
    /// Probability-weighted expected level (`sum_k k * p_k`); can land
    /// between levels.
    score: f64 = 0,
    legend: Legend = .{},
    probabilities: Probabilities = .{},
    /// The highest level probability.
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

/// One answer, discriminated by the wire's `"type"`. An answer type this
/// client doesn't know yet parses as `.unknown` instead of failing the whole
/// response. Every slice borrows the owning `Response`.
pub const Answer = union(enum) {
    noul: NoulAnswer,
    choice: ChoiceAnswer,
    score: ScoreAnswer,
    unknown: []const u8,

    /// Every answer type's fields, so `"type"` may arrive in any position.
    const Wire = struct {
        type: []const u8,
        noul: f64 = 0,
        choice: []const u8 = "",
        score: f64 = 0,
        legend: Legend = .{},
        probabilities: Probabilities = .{},
        confidence: f64 = 0,
    };

    const Known = enum { noul, choice, score };

    pub fn jsonParse(
        allocator: std.mem.Allocator,
        source: anytype,
        options: std.json.ParseOptions,
    ) std.json.ParseError(@TypeOf(source.*))!Answer {
        var opts = options;
        opts.ignore_unknown_fields = true;
        const w = try std.json.innerParse(Wire, allocator, source, opts);
        const known = std.meta.stringToEnum(Known, w.type) orelse return .{ .unknown = w.type };
        switch (known) {
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

    pub fn confidence(self: Answer) ?f64 {
        return switch (self) {
            inline .noul, .choice, .score => |a| a.confidence,
            .unknown => null,
        };
    }

    /// Probability assigned to one option id (`choice`) or level index (`score`).
    pub fn probability(self: Answer, key: []const u8) ?f64 {
        return switch (self) {
            inline .choice, .score => |a| a.probabilities.get(key),
            else => null,
        };
    }
};

/// Question id -> answer, keyed exactly as the request's questions.
pub const Answers = StringMap(Answer);

/// Set when the input was blocked; `answers` is then empty.
pub const PromptFeedback = struct {
    /// "safety", "other", "blocklist", "prohibited_content", "image_safety".
    block_reason: ?[]const u8 = null,
    block_reason_message: ?[]const u8 = null,
};

pub const Usage = struct {
    total_input_tokens: ?u32 = null,
    total_output_tokens: ?u32 = null,
    total_thought_tokens: ?u32 = null,
    total_cached_tokens: ?u32 = null,
    total_tokens: ?u32 = null,
};

/// Every slice here borrows the owning `Response`, which must outlive them.
pub const CreateDecisionResponse = struct {
    answers: Answers = .{},
    prompt_feedback: ?PromptFeedback = null,
    usage: ?Usage = null,

    /// The answer stored under `id`, or null if the model omitted it.
    pub fn answer(self: CreateDecisionResponse, id: []const u8) ?Answer {
        return self.answers.get(id);
    }

    /// The `choice` under `id`, validated against the criteria in the
    /// `questions` that were sent.
    pub fn choice(self: CreateDecisionResponse, id: []const u8, questions: Questions) ChoiceError![]const u8 {
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
        self: CreateDecisionResponse,
        id: []const u8,
        questions: Questions,
        comptime E: type,
    ) ChoiceError!E {
        const name = try self.choice(id, questions);
        return std.meta.stringToEnum(E, name) orelse error.ChoiceNotOffered;
    }

    /// Null when unanswered or not a `noul`.
    pub fn noul(self: CreateDecisionResponse, id: []const u8) ?f64 {
        return (self.answer(id) orelse return null).noulValue();
    }

    /// Null when unanswered.
    pub fn confidence(self: CreateDecisionResponse, id: []const u8) ?f64 {
        return (self.answer(id) orelse return null).confidence();
    }

    /// Null when unanswered or a `noul`.
    pub fn probability(self: CreateDecisionResponse, id: []const u8, key: []const u8) ?f64 {
        return (self.answer(id) orelse return null).probability(key);
    }
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

/// Why `CreateDecisionResponse.choice` found no usable choice.
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

fn expectJson(value: anytype, expected: []const u8) !void {
    const out = try jsonutil.stringifyAlloc(std.testing.allocator, value, .{ .emit_null_optional_fields = false });
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings(expected, out);
}

test "CreateDecisionRequest stringifies the request body" {
    const request: CreateDecisionRequest = .{
        .input = .{ .text = "The pizza arrived cold." },
        .model = "models/gemini-decision-1",
        .questions = .init(&.{
            .{ .key = "is_complaint", .value = .{ .noul = .{
                .instructions = "Is the customer complaining?",
                .criteria = .{ .true_description = "A complaint" },
            } } },
            .{ .key = "topic", .value = .choiceText("What is this about?", .init(&.{
                .{ .key = "billing", .value = "" },
                .{ .key = "delivery", .value = "Anything about shipping" },
            })) },
            .{ .key = "anger", .value = .scoreText("How angry?", &.{ "Calm", "Furious" }) },
        }),
    };

    try expectJson(request,
        \\{"input":"The pizza arrived cold.","model":"models/gemini-decision-1","questions":{"is_complaint":{"type":"noul","instructions":"Is the customer complaining?","criteria":{"true_description":"A complaint"}},"topic":{"type":"choice","instructions":"What is this about?","criteria":{"billing":"","delivery":"Anything about shipping"}},"anger":{"type":"score","instructions":"How angry?","criteria":["Calm","Furious"]}}}
    );
}

test "Input stringifies content blocks" {
    const input: Input = .{ .parts = &.{
        .{ .type = .text, .text = "Is this a cat?" },
        .{ .type = .image, .data = "aGk=", .mime_type = "image/png" },
        .{ .type = .document, .uri = "https://example.com/f.pdf" },
    } };
    try expectJson(input,
        \\[{"type":"text","text":"Is this a cat?"},{"type":"image","data":"aGk=","mime_type":"image/png"},{"type":"document","uri":"https://example.com/f.pdf"}]
    );
}

test "CreateDecisionResponse parses every answer type" {
    const fixture =
        \\{
        \\  "answers": {
        \\    "is_complaint": {"type": "noul", "noul": 0.95, "confidence": 0.95},
        \\    "topic": {"confidence": 0.81, "type": "choice", "choice": "billing", "probabilities": {"billing": 0.81, "delivery": 0.17, "other": 0.02}},
        \\    "anger": {"type": "score", "score": 1.05, "legend": {"0": "Calm", "1": "Annoyed", "2": "Furious"}, "probabilities": {"0": 0, "1": 0.9, "2": 0.1}, "confidence": 0.9},
        \\    "future": {"type": "ranking", "order": ["a", "b"]}
        \\  },
        \\  "usage": {"total_input_tokens": 296, "total_output_tokens": 3, "total_tokens": 299, "input_tokens_by_modality": []}
        \\}
    ;
    const parsed = try std.json.parseFromSlice(CreateDecisionResponse, std.testing.allocator, fixture, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const r = parsed.value;

    try std.testing.expectEqual(@as(?u32, 296), r.usage.?.total_input_tokens);
    try std.testing.expect(r.prompt_feedback == null);

    try std.testing.expectApproxEqAbs(@as(f64, 0.95), r.noul("is_complaint").?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.95), r.confidence("is_complaint").?, 1e-12);

    const questions: Questions = .init(&.{
        .{ .key = "topic", .value = .choiceText("?", choices(&.{ "billing", "delivery", "other" })) },
    });
    try std.testing.expectEqualStrings("billing", try r.choice("topic", questions));
    try std.testing.expectApproxEqAbs(@as(f64, 0.17), r.probability("topic", "delivery").?, 1e-12);

    switch (r.answer("anger").?) {
        .score => |s| {
            try std.testing.expectEqualStrings("Annoyed", s.nearestLevel().?.value);
            try std.testing.expectEqual(@as(?f64, 0), s.probabilities.get("0"));
        },
        else => return error.TestUnexpectedResult,
    }

    try std.testing.expectEqualStrings("ranking", r.answer("future").?.unknown);
    try std.testing.expectEqual(@as(?f64, null), r.confidence("future"));
    try std.testing.expect(r.answer("absent") == null);
}

test "CreateDecisionResponse parses a live response" {
    const parsed = try std.json.parseFromSlice(CreateDecisionResponse, std.testing.allocator,
        \\{"answers":{"topic":{"confidence":0.9937235540489243,"choice":"delivery","probabilities":{"delivery":0.9937235540489243,"billing":0.0005501006753907835,"other":0.005726345275685044},"type":"choice"},"anger":{"confidence":0.5785060966601108,"score":1.2988803052621227,"probabilities":{"0":0.061306799038883183,"1":0.5785060966601108,"2":0.3601871043010059},"legend":{"1":"Annoyed","0":"Calm","2":"Furious"},"type":"score"},"is_complaint":{"confidence":0.9362161928061764,"noul":0.9362161928061764,"type":"noul"}},"usage":{"total_tokens":196,"total_input_tokens":196,"input_tokens_by_modality":[{"modality":"text","tokens":196}],"total_output_tokens":0}}
    , .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const r = parsed.value;
    const questions: Questions = .init(&.{
        .{ .key = "topic", .value = .choiceText("?", choices(&.{ "billing", "delivery", "other" })) },
    });
    try std.testing.expectEqualStrings("delivery", try r.choice("topic", questions));
    try std.testing.expectEqualStrings("Annoyed", r.answer("anger").?.score.nearestLevel().?.value);
    try std.testing.expectApproxEqAbs(@as(f64, 0.936), r.noul("is_complaint").?, 1e-3);
    try std.testing.expectEqual(@as(?u32, 196), r.usage.?.total_tokens);
}

test "CreateDecisionResponse parses a blocked prompt" {
    const parsed = try std.json.parseFromSlice(CreateDecisionResponse, std.testing.allocator,
        \\{"prompt_feedback":{"block_reason":"safety","block_reason_message":"nope"}}
    , .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 0), parsed.value.answers.count());
    try std.testing.expectEqualStrings("safety", parsed.value.prompt_feedback.?.block_reason.?);
}

test "validateChoice rejects an answer that strayed" {
    const offered = choices(&.{ "a", "b" });
    try validateChoice(.{ .choice = .{ .choice = "a", .probabilities = .init(&.{
        .{ .key = "a", .value = 0.5 },
        .{ .key = "b", .value = 0.5 },
    }) } }, offered);
    try std.testing.expectError(error.NotAChoice, validateChoice(.{ .noul = .{ .noul = 0.5 } }, offered));
    try std.testing.expectError(error.ChoiceNotOffered, validateChoice(.{ .choice = .{ .choice = "z" } }, offered));
    try std.testing.expectError(error.ChoiceNotArgmax, validateChoice(.{ .choice = .{ .choice = "a", .probabilities = .init(&.{
        .{ .key = "a", .value = 0.3 },
        .{ .key = "b", .value = 0.7 },
    }) } }, offered));
}
