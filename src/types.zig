const std = @import("std");

// --- Enums ---

pub const HarmCategory = enum {
    HARM_CATEGORY_UNSPECIFIED,
    HARM_CATEGORY_HARASSMENT,
    HARM_CATEGORY_HATE_SPEECH,
    HARM_CATEGORY_SEXUALLY_EXPLICIT,
    HARM_CATEGORY_DANGEROUS_CONTENT,
    HARM_CATEGORY_CIVIC_INTEGRITY,
};

pub const HarmBlockThreshold = enum {
    HARM_BLOCK_THRESHOLD_UNSPECIFIED,
    BLOCK_LOW_AND_ABOVE,
    BLOCK_MEDIUM_AND_ABOVE,
    BLOCK_ONLY_HIGH,
    BLOCK_NONE,
    OFF,
};

pub const HarmProbability = enum {
    HARM_PROBABILITY_UNSPECIFIED,
    NEGLIGIBLE,
    LOW,
    MEDIUM,
    HIGH,
};

pub const HarmSeverity = enum {
    HARM_SEVERITY_UNSPECIFIED,
    HARM_SEVERITY_NEGLIGIBLE,
    HARM_SEVERITY_LOW,
    HARM_SEVERITY_MEDIUM,
    HARM_SEVERITY_HIGH,
};

pub const FinishReason = enum {
    FINISH_REASON_UNSPECIFIED,
    STOP,
    MAX_TOKENS,
    SAFETY,
    RECITATION,
    LANGUAGE,
    OTHER,
    BLOCKLIST,
    PROHIBITED_CONTENT,
    SPII,
    MALFORMED_FUNCTION_CALL,
};

pub const FunctionCallingMode = enum {
    MODE_UNSPECIFIED,
    AUTO,
    ANY,
    NONE,
    VALIDATED,
};

pub const SchemaType = enum {
    TYPE_UNSPECIFIED,
    STRING,
    NUMBER,
    INTEGER,
    BOOLEAN,
    ARRAY,
    OBJECT,
};

pub const Outcome = enum {
    OUTCOME_UNSPECIFIED,
    OUTCOME_OK,
    OUTCOME_FAILED,
    OUTCOME_DEADLINE_EXCEEDED,
};

pub const Language = enum {
    LANGUAGE_UNSPECIFIED,
    PYTHON,
};

// --- Core Types ---

pub const Blob = struct {
    data: ?[]const u8 = null,
    mimeType: ?[]const u8 = null,
    displayName: ?[]const u8 = null,
};

pub const FileData = struct {
    fileUri: ?[]const u8 = null,
    mimeType: ?[]const u8 = null,
};

pub const FunctionCall = struct {
    id: ?[]const u8 = null,
    name: ?[]const u8 = null,
    args: ?std.json.Value = null,
};

pub const FunctionResponse = struct {
    id: ?[]const u8 = null,
    name: ?[]const u8 = null,
    response: ?std.json.Value = null,
};

pub const ExecutableCode = struct {
    code: ?[]const u8 = null,
    language: ?Language = null,
};

pub const CodeExecutionResult = struct {
    outcome: ?Outcome = null,
    output: ?[]const u8 = null,
};

pub const Part = struct {
    text: ?[]const u8 = null,
    inlineData: ?Blob = null,
    fileData: ?FileData = null,
    functionCall: ?FunctionCall = null,
    functionResponse: ?FunctionResponse = null,
    executableCode: ?ExecutableCode = null,
    codeExecutionResult: ?CodeExecutionResult = null,
    thought: ?bool = null,
};

pub const Content = struct {
    role: ?[]const u8 = null,
    parts: []const Part,
};

// --- Schema ---

pub const Property = struct {
    key: []const u8,
    value: Schema,
};

pub const Schema = struct {
    type: ?SchemaType = null,
    description: ?[]const u8 = null,
    @"enum": ?[]const []const u8 = null,
    items: ?*const Schema = null,
    properties: ?[]const Property = null,
    required: ?[]const []const u8 = null,
    nullable: ?bool = null,
    format: ?[]const u8 = null,
    title: ?[]const u8 = null,
    minimum: ?f64 = null,
    maximum: ?f64 = null,
    minItems: ?i64 = null,
    maxItems: ?i64 = null,
    minLength: ?i64 = null,
    maxLength: ?i64 = null,
    pattern: ?[]const u8 = null,

    /// Custom JSON serialization: emit `properties` as a JSON object instead of an array.
    pub fn jsonStringify(self: *const Schema, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        inline for (std.meta.fields(Schema)) |field| {
            if (comptime std.mem.eql(u8, field.name, "properties")) {
                if (self.properties) |props| {
                    try jw.objectField("properties");
                    try jw.beginObject();
                    for (props) |prop| {
                        try jw.objectField(prop.key);
                        try jw.write(prop.value);
                    }
                    try jw.endObject();
                }
            } else {
                const val = @field(self, field.name);
                if (comptime @typeInfo(field.type) == .optional) {
                    if (val) |unwrapped| {
                        try jw.objectField(field.name);
                        try jw.write(unwrapped);
                    } else if (jw.options.emit_null_optional_fields) {
                        try jw.objectField(field.name);
                        try jw.write(null);
                    }
                } else {
                    try jw.objectField(field.name);
                    try jw.write(val);
                }
            }
        }
        try jw.endObject();
    }
};

// --- Safety ---

pub const SafetySetting = struct {
    category: ?HarmCategory = null,
    threshold: ?HarmBlockThreshold = null,
};

pub const SafetyRating = struct {
    category: ?HarmCategory = null,
    probability: ?HarmProbability = null,
    probabilityScore: ?f32 = null,
    severity: ?HarmSeverity = null,
    severityScore: ?f32 = null,
    blocked: ?bool = null,
};

// --- Tools ---

pub const FunctionDeclaration = struct {
    name: ?[]const u8 = null,
    description: ?[]const u8 = null,
    parameters: ?Schema = null,
};

pub const ToolCodeExecution = struct {};

pub const GoogleSearch = struct {};

pub const Tool = struct {
    functionDeclarations: ?[]const FunctionDeclaration = null,
    codeExecution: ?ToolCodeExecution = null,
    googleSearch: ?GoogleSearch = null,
};

pub const FunctionCallingConfig = struct {
    mode: ?FunctionCallingMode = null,
    allowedFunctionNames: ?[]const []const u8 = null,
};

pub const ToolConfig = struct {
    functionCallingConfig: ?FunctionCallingConfig = null,
};

// --- Generation Config ---

pub const GenerationConfig = struct {
    temperature: ?f32 = null,
    topP: ?f32 = null,
    topK: ?f32 = null,
    candidateCount: ?i32 = null,
    maxOutputTokens: ?i32 = null,
    stopSequences: ?[]const []const u8 = null,
    presencePenalty: ?f32 = null,
    frequencyPenalty: ?f32 = null,
    seed: ?i32 = null,
    responseMimeType: ?[]const u8 = null,
    responseSchema: ?Schema = null,
    responseLogprobs: ?bool = null,
    logprobs: ?i32 = null,
};

// --- Request ---

pub const GenerateContentRequest = struct {
    contents: []const Content,
    generationConfig: ?GenerationConfig = null,
    systemInstruction: ?Content = null,
    safetySettings: ?[]const SafetySetting = null,
    tools: ?[]const Tool = null,
    toolConfig: ?ToolConfig = null,
};

// --- Response Types ---

pub const CitationSource = struct {
    startIndex: ?i32 = null,
    endIndex: ?i32 = null,
    uri: ?[]const u8 = null,
    license: ?[]const u8 = null,
};

pub const CitationMetadata = struct {
    citationSources: ?[]const CitationSource = null,
};

pub const Candidate = struct {
    content: ?Content = null,
    finishReason: ?FinishReason = null,
    safetyRatings: ?[]const SafetyRating = null,
    citationMetadata: ?CitationMetadata = null,
    tokenCount: ?i32 = null,
    avgLogprobs: ?f64 = null,
    index: ?i32 = null,
};

pub const UsageMetadata = struct {
    promptTokenCount: ?i32 = null,
    candidatesTokenCount: ?i32 = null,
    totalTokenCount: ?i32 = null,
    cachedContentTokenCount: ?i32 = null,
    thoughtsTokenCount: ?i32 = null,
};

pub const GenerateContentResponse = struct {
    candidates: ?[]const Candidate = null,
    usageMetadata: ?UsageMetadata = null,
    modelVersion: ?[]const u8 = null,
    responseId: ?[]const u8 = null,

    /// Extract text from the first candidate's first part.
    pub fn text(self: GenerateContentResponse) ?[]const u8 {
        const candidates = self.candidates orelse return null;
        if (candidates.len == 0) return null;
        const content = candidates[0].content orelse return null;
        if (content.parts.len == 0) return null;
        return content.parts[0].text;
    }

    /// Extract the first function call from the first candidate.
    pub fn firstFunctionCall(self: GenerateContentResponse) ?FunctionCall {
        const candidates = self.candidates orelse return null;
        if (candidates.len == 0) return null;
        const content = candidates[0].content orelse return null;
        for (content.parts) |part| {
            if (part.functionCall) |fc| return fc;
        }
        return null;
    }

    /// Return all parts from the first candidate.
    pub fn parts(self: GenerateContentResponse) ?[]const Part {
        const candidates = self.candidates orelse return null;
        if (candidates.len == 0) return null;
        const content = candidates[0].content orelse return null;
        if (content.parts.len == 0) return null;
        return content.parts;
    }
};

// --- Error Types ---

pub const ApiErrorResponse = struct {
    @"error": ?ApiErrorDetail = null,
};

pub const ApiErrorDetail = struct {
    code: ?u32 = null,
    message: ?[]const u8 = null,
    status: ?[]const u8 = null,
};

// --- Tests ---

test "GenerateContentRequest serializes to JSON" {
    const parts = [_]Part{.{ .text = "hello" }};
    const contents = [_]Content{.{ .role = "user", .parts = &parts }};
    const req = GenerateContentRequest{
        .contents = &contents,
        .generationConfig = .{ .temperature = 0.5 },
    };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(req, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "temperature") != null);
}

test "GenerateContentRequest with systemInstruction serializes correctly" {
    const parts = [_]Part{.{ .text = "hello" }};
    const contents = [_]Content{.{ .role = "user", .parts = &parts }};
    const sys_parts = [_]Part{.{ .text = "You are helpful." }};
    const req = GenerateContentRequest{
        .contents = &contents,
        .systemInstruction = .{ .parts = &sys_parts },
        .generationConfig = .{ .temperature = 0, .maxOutputTokens = 100 },
    };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(req, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "systemInstruction") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "You are helpful.") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "maxOutputTokens") != null);
}

test "SafetySetting serializes with enum tag names" {
    const setting = SafetySetting{
        .category = .HARM_CATEGORY_HARASSMENT,
        .threshold = .BLOCK_MEDIUM_AND_ABOVE,
    };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(setting, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "HARM_CATEGORY_HARASSMENT") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "BLOCK_MEDIUM_AND_ABOVE") != null);
}

test "GenerateContentResponse.text extracts text" {
    const json =
        \\{"candidates":[{"content":{"role":"model","parts":[{"text":"I am Gemini"}]},"finishReason":"STOP"}]}
    ;
    const parsed = try std.json.parseFromSlice(
        GenerateContentResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    try std.testing.expectEqualStrings("I am Gemini", parsed.value.text().?);
}

test "GenerateContentResponse parses safety ratings" {
    const json =
        \\{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP","safetyRatings":[{"category":"HARM_CATEGORY_HARASSMENT","probability":"NEGLIGIBLE"}]}]}
    ;
    const parsed = try std.json.parseFromSlice(
        GenerateContentResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    const ratings = parsed.value.candidates.?[0].safetyRatings.?;
    try std.testing.expect(ratings[0].category.? == .HARM_CATEGORY_HARASSMENT);
    try std.testing.expect(ratings[0].probability.? == .NEGLIGIBLE);
}

test "GenerateContentResponse parses function call" {
    const json =
        \\{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"London"}}}]},"finishReason":"STOP"}]}
    ;
    const parsed = try std.json.parseFromSlice(
        GenerateContentResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    const fc = parsed.value.candidates.?[0].content.?.parts[0].functionCall.?;
    try std.testing.expectEqualStrings("get_weather", fc.name.?);
}

test "FinishReason parses from JSON" {
    const json =
        \\{"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"MAX_TOKENS"}]}
    ;
    const parsed = try std.json.parseFromSlice(
        GenerateContentResponse,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    try std.testing.expect(parsed.value.candidates.?[0].finishReason.? == .MAX_TOKENS);
}

test "Schema serializes with enum type" {
    const schema = Schema{
        .type = .OBJECT,
        .description = "A person",
        .required = &.{"name"},
    };
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    try std.json.Stringify.value(schema, .{ .emit_null_optional_fields = false }, &buf.writer);
    const json = buf.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "OBJECT") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "A person") != null);
}
