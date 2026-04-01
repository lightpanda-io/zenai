const std = @import("std");

// --- Enums ---

/// The service tier for the request.
pub const ServiceTier = enum {
    unspecified,
    flex,
    standard,
    priority,
};

/// The harm category that a piece of content may be classified under.
pub const HarmCategory = enum {
    /// Default value. This value is unused.
    HARM_CATEGORY_UNSPECIFIED,
    /// Abusive, threatening, or content intended to bully, torment, or ridicule.
    HARM_CATEGORY_HARASSMENT,
    /// Content that promotes violence or incites hatred against individuals or groups.
    HARM_CATEGORY_HATE_SPEECH,
    /// Content that contains sexually explicit material.
    HARM_CATEGORY_SEXUALLY_EXPLICIT,
    /// Content that promotes, facilitates, or enables dangerous activities.
    HARM_CATEGORY_DANGEROUS_CONTENT,
    /// The harm category is civic integrity.
    HARM_CATEGORY_CIVIC_INTEGRITY,
};

/// The threshold for blocking content based on harm probability.
pub const HarmBlockThreshold = enum {
    /// The harm block threshold is unspecified.
    HARM_BLOCK_THRESHOLD_UNSPECIFIED,
    /// Block content with a low harm probability or higher.
    BLOCK_LOW_AND_ABOVE,
    /// Block content with a medium harm probability or higher.
    BLOCK_MEDIUM_AND_ABOVE,
    /// Block content with a high harm probability.
    BLOCK_ONLY_HIGH,
    /// Do not block any content, regardless of its harm probability.
    BLOCK_NONE,
    /// Turn off the safety filter entirely.
    OFF,
};

/// The probability level of harmful content.
pub const HarmProbability = enum {
    HARM_PROBABILITY_UNSPECIFIED,
    NEGLIGIBLE,
    LOW,
    MEDIUM,
    HIGH,
};

/// The severity level of harmful content.
pub const HarmSeverity = enum {
    HARM_SEVERITY_UNSPECIFIED,
    HARM_SEVERITY_NEGLIGIBLE,
    HARM_SEVERITY_LOW,
    HARM_SEVERITY_MEDIUM,
    HARM_SEVERITY_HIGH,
};

/// The stage of the underlying model.
pub const ModelStage = enum {
    MODEL_STAGE_UNSPECIFIED,
    UNSTABLE_EXPERIMENTAL,
    EXPERIMENTAL,
    PREVIEW,
    STABLE,
    LEGACY,
    DEPRECATED,
    RETIRED,
};

/// Current status of the model.
pub const ModelStatus = struct {
    /// A message explaining the model status.
    message: ?[]const u8 = null,
    /// The stage of the underlying model.
    modelStage: ?ModelStage = null,
    /// The time at which the model will be retired.
    retirementTime: ?[]const u8 = null,
};

/// The media resolution to use.
pub const MediaResolution = enum {
    MEDIA_RESOLUTION_UNSPECIFIED,
    LOW,
    MEDIUM,
    HIGH,
};

/// The reason why the model stopped generating tokens.
pub const FinishReason = enum {
    /// The finish reason is unspecified.
    FINISH_REASON_UNSPECIFIED,
    /// Token generation reached a natural stopping point or a configured stop sequence.
    STOP,
    /// Token generation reached the configured maximum output tokens.
    MAX_TOKENS,
    /// Token generation stopped because the content potentially contains safety violations.
    SAFETY,
    /// Token generation stopped because of potential recitation.
    RECITATION,
    /// Token generation stopped because of using an unsupported language.
    LANGUAGE,
    /// All other reasons that stopped the token generation.
    OTHER,
    /// Token generation stopped because the content contains forbidden terms.
    BLOCKLIST,
    /// Token generation stopped for potentially containing prohibited content.
    PROHIBITED_CONTENT,
    /// Token generation stopped because the content potentially contains Sensitive PII.
    SPII,
    /// Token generation stopped due to a malformed function call.
    MALFORMED_FUNCTION_CALL,
};

/// Function calling mode.
pub const FunctionCallingMode = enum {
    /// Mode is unspecified.
    MODE_UNSPECIFIED,
    /// Model decides whether to predict a function call or natural language response.
    AUTO,
    /// Model is constrained to always predict a function call only.
    ANY,
    /// Model will not predict any function call.
    NONE,
    /// Model may predict a function call or natural language response (validated).
    VALIDATED,
};

/// Data type of a schema field (subset of OpenAPI 3.0 types).
pub const SchemaType = enum {
    /// Not specified, should not be used.
    TYPE_UNSPECIFIED,
    /// OpenAPI string type.
    STRING,
    /// OpenAPI number type.
    NUMBER,
    /// OpenAPI integer type.
    INTEGER,
    /// OpenAPI boolean type.
    BOOLEAN,
    /// OpenAPI array type.
    ARRAY,
    /// OpenAPI object type.
    OBJECT,
};

/// Outcome of code execution.
pub const Outcome = enum {
    OUTCOME_UNSPECIFIED,
    /// Code execution completed successfully.
    OUTCOME_OK,
    /// Code execution failed.
    OUTCOME_FAILED,
    /// Code execution ran for too long and was cancelled.
    OUTCOME_DEADLINE_EXCEEDED,
};

/// Programming language for executable code.
pub const Language = enum {
    LANGUAGE_UNSPECIFIED,
    PYTHON,
};

// --- Core Types ---

/// A content blob containing raw bytes of a specific media type (images, audio, video).
pub const Blob = struct {
    /// The raw bytes of the data.
    data: ?[]const u8 = null,
    /// The IANA standard MIME type of the source data.
    mimeType: ?[]const u8 = null,
};

/// URI-based data pointing to a file in Google Cloud Storage.
pub const FileData = struct {
    /// The URI of the file.
    fileUri: ?[]const u8 = null,
    /// The IANA standard MIME type of the source data.
    mimeType: ?[]const u8 = null,
};

/// A predicted function call returned from the model.
pub const FunctionCall = struct {
    /// The unique ID of the function call.
    id: ?[]const u8 = null,
    /// The name of the function to call. Matches `FunctionDeclaration.name`.
    name: ?[]const u8 = null,
    /// The function parameters and values in JSON object format.
    args: ?std.json.Value = null,
};

/// The result of a function call, to be sent back to the model.
pub const FunctionResponse = struct {
    /// The ID matching the corresponding `FunctionCall.id`.
    id: ?[]const u8 = null,
    /// The name of the function called. Matches `FunctionCall.name`.
    name: ?[]const u8 = null,
    /// The function response in JSON object format.
    response: ?std.json.Value = null,
};

/// Model-generated code that is intended to be executed.
/// Only generated when using the `CodeExecution` tool.
pub const ExecutableCode = struct {
    /// The code to be executed.
    code: ?[]const u8 = null,
    /// Programming language of the code.
    language: ?Language = null,
};

/// Result of executing the `ExecutableCode`.
pub const CodeExecutionResult = struct {
    /// Outcome of the code execution.
    outcome: ?Outcome = null,
    /// Contains stdout when successful, stderr or other description otherwise.
    output: ?[]const u8 = null,
};

/// A datatype containing media content. Exactly one field within a Part should be set.
pub const Part = struct {
    /// Text content.
    text: ?[]const u8 = null,
    /// Inline media bytes (images, audio, video).
    inlineData: ?Blob = null,
    /// URI-based file reference (e.g. Google Cloud Storage).
    fileData: ?FileData = null,
    /// A predicted function call returned from the model.
    functionCall: ?FunctionCall = null,
    /// The result of a function execution, to be sent back to the model.
    functionResponse: ?FunctionResponse = null,
    /// Code generated by the model that is intended to be executed.
    executableCode: ?ExecutableCode = null,
    /// The result of executing the `ExecutableCode`.
    codeExecutionResult: ?CodeExecutionResult = null,
    /// If true, marks this part as model reasoning/thinking (not final output).
    thought: ?bool = null,
    /// An opaque signature for the thought so it can be reused in subsequent requests.
    thoughtSignature: ?[]const u8 = null,
    /// Video metadata for video content (frame rate, clipping).
    videoMetadata: ?VideoMetadata = null,
    /// Media resolution for the input media.
    mediaResolution: ?MediaResolution = null,
    /// Custom metadata associated with the Part.
    partMetadata: ?std.json.Value = null,
};

/// Contains the multi-part content of a message.
pub const Content = struct {
    /// The producer of the content. Must be either "user" or "model".
    role: ?[]const u8 = null,
    /// List of parts that constitute a single message.
    parts: []const Part,
};

// --- Schema ---

/// A key-value pair for defining object properties in a `Schema`.
pub const Property = struct {
    key: []const u8,
    value: Schema,
};

/// Defines the format of input/output data. Represents a subset of an
/// OpenAPI 3.0 schema object.
/// See https://spec.openapis.org/oas/v3.0.3#schema-object
pub const Schema = struct {
    /// Data type of the schema field.
    type: ?SchemaType = null,
    /// Description of the data. The model uses this to understand the purpose of the schema.
    description: ?[]const u8 = null,
    /// Possible values of the field (for enum types).
    @"enum": ?[]const []const u8 = null,
    /// If type is ARRAY, specifies the schema of elements in the array.
    items: ?*const Schema = null,
    /// If type is OBJECT, maps property names to their schema definitions.
    properties: ?[]const Property = null,
    /// If type is OBJECT, lists property names that must be present.
    required: ?[]const []const u8 = null,
    /// Indicates if the value can be null.
    nullable: ?bool = null,
    /// Format of the data (e.g. "float", "int32", "email", "date-time").
    format: ?[]const u8 = null,
    /// Title for the schema.
    title: ?[]const u8 = null,
    /// Minimum allowed numeric value.
    minimum: ?f64 = null,
    /// Maximum allowed numeric value.
    maximum: ?f64 = null,
    /// Minimum number of items in an array.
    minItems: ?i64 = null,
    /// Maximum number of items in an array.
    maxItems: ?i64 = null,
    /// Minimum length of a string.
    minLength: ?i64 = null,
    /// Maximum length of a string.
    maxLength: ?i64 = null,
    /// Regex pattern that a string must match.
    pattern: ?[]const u8 = null,

    /// Custom JSON serialization: emits `properties` as a JSON object instead of an array.
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

/// A safety setting that controls content blocking behavior for a specific harm category.
pub const SafetySetting = struct {
    /// The harm category to configure.
    category: ?HarmCategory = null,
    /// The threshold above which content is blocked.
    threshold: ?HarmBlockThreshold = null,
};

/// A safety rating for a piece of content, indicating harm probability and severity.
pub const SafetyRating = struct {
    /// The harm category of this rating.
    category: ?HarmCategory = null,
    /// The probability of harm for this category.
    probability: ?HarmProbability = null,
    /// The probability score of harm (0-1).
    probabilityScore: ?f32 = null,
    /// The severity level of harm.
    severity: ?HarmSeverity = null,
    /// The severity score (0-1).
    severityScore: ?f32 = null,
    /// Whether the content was blocked because of this rating.
    blocked: ?bool = null,
};

// --- Tools ---

/// Structured representation of a function declaration as defined by the OpenAPI 3.0 spec.
/// Can be used as a `Tool` by the model and executed by the client.
pub const FunctionDeclaration = struct {
    /// The name of the function. Must be a-z, A-Z, 0-9, underscores, dots, or dashes (max 64 chars).
    name: ?[]const u8 = null,
    /// Description and purpose of the function. The model uses this to decide how and whether to call it.
    description: ?[]const u8 = null,
    /// Parameters to this function in JSON Schema format.
    parameters: ?Schema = null,
};

/// Enables server-side code execution.
pub const ToolCodeExecution = struct {};

/// Enables Google Search grounding.
pub const GoogleSearch = struct {};

/// Tool details that the model may use to generate a response.
pub const Tool = struct {
    /// A list of function declarations available for the model to call.
    functionDeclarations: ?[]const FunctionDeclaration = null,
    /// Enables server-side code execution.
    codeExecution: ?ToolCodeExecution = null,
    /// Enables Google Search grounding.
    googleSearch: ?GoogleSearch = null,
};

/// Configuration for function calling behavior.
pub const FunctionCallingConfig = struct {
    /// Function calling mode (AUTO, ANY, NONE, VALIDATED).
    mode: ?FunctionCallingMode = null,
    /// Function names to call. Only used when mode is ANY.
    allowedFunctionNames: ?[]const []const u8 = null,
};

/// Tool config shared for all tools provided in the request.
pub const ToolConfig = struct {
    /// Function calling config.
    functionCallingConfig: ?FunctionCallingConfig = null,
};

/// Configuration for prompt and response sanitization using the Model Armor service.
pub const ModelArmorConfig = struct {
    /// The resource name of the Model Armor template to use for prompt screening.
    promptTemplateName: ?[]const u8 = null,
    /// The resource name of the Model Armor template to use for response screening.
    responseTemplateName: ?[]const u8 = null,
};

// --- Thinking Config ---

/// Configuration for the model's thinking/reasoning features.
pub const ThinkingConfig = struct {
    /// Whether to include the model's thoughts in the response.
    includeThoughts: ?bool = null,
    /// Budget in tokens for model thinking. Set to 0 to disable thinking.
    thinkingBudget: ?i32 = null,
};

// --- Generation Config ---

/// Optional model configuration parameters for content generation.
pub const GenerationConfig = struct {
    /// Controls the degree of randomness in token selection (0.0-2.0).
    /// Lower values produce more deterministic output.
    temperature: ?f32 = null,
    /// Nucleus sampling threshold. Tokens are selected from most to least probable
    /// until their cumulative probability equals this value.
    topP: ?f32 = null,
    /// Top-k sampling: the model considers only the top k most probable tokens.
    topK: ?f32 = null,
    /// Number of response candidates to generate.
    candidateCount: ?i32 = null,
    /// Maximum number of tokens in the generated response.
    maxOutputTokens: ?i32 = null,
    /// Sequences that will stop generation when encountered.
    stopSequences: ?[]const []const u8 = null,
    /// Penalizes tokens that have already appeared, encouraging diversity.
    presencePenalty: ?f32 = null,
    /// Penalizes tokens based on frequency of appearance, reducing repetition.
    frequencyPenalty: ?f32 = null,
    /// Random seed for deterministic generation.
    seed: ?i32 = null,
    /// Output format MIME type (e.g. "text/plain", "application/json").
    responseMimeType: ?[]const u8 = null,
    /// Schema defining the expected structure of JSON output.
    responseSchema: ?Schema = null,
    /// Whether to return log probabilities of generated tokens.
    responseLogprobs: ?bool = null,
    /// Number of top candidate tokens to return log probabilities for.
    logprobs: ?i32 = null,
    /// Configuration for the model's thinking/reasoning features.
    thinkingConfig: ?ThinkingConfig = null,
    /// Resource name of cached content to use as context.
    cachedContent: ?[]const u8 = null,
    /// Requested response modalities.
    responseModalities: ?[]const Modality = null,
    /// Media resolution for input media.
    mediaResolution: ?MediaResolution = null,
    /// Whether to include audio timestamps in the response.
    audioTimestamp: ?bool = null,
    /// Labels with user-defined metadata to break down billed charges.
    labels: ?std.json.Value = null,
    /// Output schema of the generated response (alternative to responseSchema).
    responseJsonSchema: ?std.json.Value = null,
};

// --- Request ---

/// Request body for the generateContent endpoint.
pub const GenerateContentRequest = struct {
    /// The content of the conversation so far.
    contents: []const Content,
    /// Optional generation parameters.
    generationConfig: ?GenerationConfig = null,
    /// System instructions to steer the model toward better performance.
    systemInstruction: ?Content = null,
    /// Safety settings to filter content.
    safetySettings: ?[]const SafetySetting = null,
    /// Tools the model may use to generate a response.
    tools: ?[]const Tool = null,
    /// Configuration for tool usage.
    toolConfig: ?ToolConfig = null,
    /// Settings for prompt and response sanitization using the Model Armor service.
    modelArmorConfig: ?ModelArmorConfig = null,
    /// The service tier to use for the request.
    serviceTier: ?ServiceTier = null,
};

// --- Response Types ---

/// A citation to an external source.
pub const CitationSource = struct {
    /// Start index of the cited text in the response.
    startIndex: ?i32 = null,
    /// End index of the cited text in the response.
    endIndex: ?i32 = null,
    /// URI of the cited source.
    uri: ?[]const u8 = null,
    /// License of the cited source.
    license: ?[]const u8 = null,
};

/// Source attribution metadata for generated content.
pub const CitationMetadata = struct {
    citationSources: ?[]const CitationSource = null,
};

/// Top candidate tokens with log probabilities at a generation step.
pub const TopCandidates = struct {
    /// Sorted by log probability in descending order.
    candidates: ?[]const TokenLogprob = null,
};

/// Log probability information for a token.
pub const TokenLogprob = struct {
    /// The candidate token string value.
    token: ?[]const u8 = null,
    /// The token ID.
    tokenId: ?i32 = null,
    /// The log probability of the token.
    logProbability: ?f64 = null,
};

/// Log probability results for a candidate.
pub const LogprobsResult = struct {
    /// Length = total number of decoding steps. The chosen candidates may or may not
    /// be in topCandidates.
    topCandidates: ?[]const TopCandidates = null,
    /// Length = total number of decoding steps.
    chosenCandidates: ?[]const TokenLogprob = null,
};

/// A grounding chunk from a web source.
pub const GroundingChunkWeb = struct {
    /// URI reference of the grounding chunk.
    uri: ?[]const u8 = null,
    /// Title of the grounding chunk.
    title: ?[]const u8 = null,
};

/// A grounding chunk from a retrieved context.
pub const GroundingChunkRetrievedContext = struct {
    /// URI reference of the attribution.
    uri: ?[]const u8 = null,
    /// Title of the attribution.
    title: ?[]const u8 = null,
    /// Text of the attribution.
    text: ?[]const u8 = null,
};

/// Grounding chunk — a reference to a source used to ground the response.
pub const GroundingChunk = struct {
    /// Grounding chunk from the web.
    web: ?GroundingChunkWeb = null,
    /// Grounding chunk from a retrieved context.
    retrievedContext: ?GroundingChunkRetrievedContext = null,
};

/// Segment of the content grounded by a supporting reference.
pub const GroundingSupport = struct {
    /// Indices into the grounding chunks.
    groundingChunkIndices: ?[]const i32 = null,
    /// Confidence scores of the support references (0-1).
    confidenceScores: ?[]const f64 = null,
    /// Content of the grounding support segment.
    segment: ?Segment = null,
};

/// A segment of content.
pub const Segment = struct {
    /// Start index in the response.
    startIndex: ?i32 = null,
    /// End index in the response.
    endIndex: ?i32 = null,
    /// The text corresponding to the segment.
    text: ?[]const u8 = null,
    /// Part index in the response.
    partIndex: ?i32 = null,
};

/// Search entry point returned with grounding metadata.
pub const SearchEntryPoint = struct {
    /// The rendered search entry point HTML snippet.
    renderedContent: ?[]const u8 = null,
    /// Base64 encoded JSON of the search entry point.
    sdkBlob: ?[]const u8 = null,
};

/// Metadata about grounding sources used in the response.
pub const GroundingMetadata = struct {
    /// List of grounding chunks (supporting references).
    groundingChunks: ?[]const GroundingChunk = null,
    /// List of grounding supports.
    groundingSupports: ?[]const GroundingSupport = null,
    /// Google search entry point.
    searchEntryPoint: ?SearchEntryPoint = null,
    /// Web search queries for follow-up.
    webSearchQueries: ?[]const []const u8 = null,
};

/// Video metadata for a video Part.
pub const VideoMetadata = struct {
    /// The start offset of the video.
    startOffset: ?[]const u8 = null,
    /// The end offset of the video.
    endOffset: ?[]const u8 = null,
};

/// A response candidate generated from the model.
pub const Candidate = struct {
    /// The generated content.
    content: ?Content = null,
    /// The reason why the model stopped generating tokens.
    finishReason: ?FinishReason = null,
    /// Human-readable message describing why generation stopped.
    finishMessage: ?[]const u8 = null,
    /// Safety ratings for this candidate.
    safetyRatings: ?[]const SafetyRating = null,
    /// Source attribution of the generated content.
    citationMetadata: ?CitationMetadata = null,
    /// Number of tokens for this candidate.
    tokenCount: ?i32 = null,
    /// Average log probability of tokens. Higher values suggest more confident responses.
    avgLogprobs: ?f64 = null,
    /// The index of this candidate.
    index: ?i32 = null,
    /// Detailed log probability information for the tokens in this candidate.
    logprobsResult: ?LogprobsResult = null,
    /// Grounding metadata (sources used when Google Search grounding is enabled).
    groundingMetadata: ?GroundingMetadata = null,
};

/// Response modality type.
pub const Modality = enum {
    MODALITY_UNSPECIFIED,
    TEXT,
    IMAGE,
    AUDIO,
};

/// Media modality type (superset of Modality, used in token count breakdowns).
pub const MediaModality = enum {
    MODALITY_UNSPECIFIED,
    TEXT,
    IMAGE,
    VIDEO,
    AUDIO,
    DOCUMENT,
};

/// Token count broken down by modality.
pub const ModalityTokenCount = struct {
    /// The modality.
    modality: ?MediaModality = null,
    /// The number of tokens for this modality.
    tokenCount: ?i32 = null,
};

/// Token usage metadata for a generate content request/response.
pub const UsageMetadata = struct {
    /// Number of tokens in the prompt.
    promptTokenCount: ?i32 = null,
    /// Total number of tokens in the generated candidates.
    candidatesTokenCount: ?i32 = null,
    /// Total token count (prompt + candidates).
    totalTokenCount: ?i32 = null,
    /// Number of tokens from cached content.
    cachedContentTokenCount: ?i32 = null,
    /// Number of tokens in the model's thinking/reasoning output.
    thoughtsTokenCount: ?i32 = null,
    /// Number of tokens in tool execution results included in the prompt.
    toolUsePromptTokenCount: ?i32 = null,
    /// Per-modality breakdown of cached content tokens.
    cacheTokensDetails: ?[]const ModalityTokenCount = null,
    /// Per-modality breakdown of candidate tokens.
    candidatesTokensDetails: ?[]const ModalityTokenCount = null,
    /// Per-modality breakdown of prompt tokens.
    promptTokensDetails: ?[]const ModalityTokenCount = null,
    /// Per-modality breakdown of tool use prompt tokens.
    toolUsePromptTokensDetails: ?[]const ModalityTokenCount = null,
};

/// Content filter results for a prompt. Only present when no candidates were
/// generated due to content violations.
pub const PromptFeedback = struct {
    /// The reason the prompt was blocked.
    blockReason: ?[]const u8 = null,
    /// Safety ratings for the prompt.
    safetyRatings: ?[]const SafetyRating = null,
};

/// Response from the generateContent or streamGenerateContent endpoint.
pub const GenerateContentResponse = struct {
    /// Response candidates generated by the model.
    candidates: ?[]const Candidate = null,
    /// Token usage metadata.
    usageMetadata: ?UsageMetadata = null,
    /// The model version used to generate the response.
    modelVersion: ?[]const u8 = null,
    /// Unique response identifier.
    responseId: ?[]const u8 = null,
    /// Timestamp when the request was made.
    createTime: ?[]const u8 = null,
    /// Current status of the model.
    modelStatus: ?ModelStatus = null,
    /// Content filter results for the prompt (only when candidates were blocked).
    promptFeedback: ?PromptFeedback = null,

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

/// API error response wrapper.
pub const ApiErrorResponse = struct {
    @"error": ?ApiErrorDetail = null,
};

/// Details of an API error.
pub const ApiErrorDetail = struct {
    /// HTTP status code.
    code: ?u32 = null,
    /// Human-readable error message.
    message: ?[]const u8 = null,
    /// Error status string (e.g. "INVALID_ARGUMENT").
    status: ?[]const u8 = null,
};

// --- Model Info ---

/// A trained machine learning model.
pub const Model = struct {
    /// Resource name of the model (e.g. "models/gemini-2.5-flash").
    name: ?[]const u8 = null,
    /// Human-readable display name.
    displayName: ?[]const u8 = null,
    /// Description of the model.
    description: ?[]const u8 = null,
    /// Version ID of the model.
    version: ?[]const u8 = null,
    /// Maximum number of input tokens the model can handle.
    inputTokenLimit: ?i32 = null,
    /// Maximum number of output tokens the model can generate.
    outputTokenLimit: ?i32 = null,
    /// List of generation methods the model supports.
    supportedGenerationMethods: ?[]const []const u8 = null,
    /// Default temperature for sampling.
    temperature: ?f32 = null,
    /// Maximum allowed temperature value.
    maxTemperature: ?f32 = null,
    /// Default top-p sampling threshold.
    topP: ?f32 = null,
    /// Default top-k sampling value.
    topK: ?i32 = null,
};

/// Response from the listModels endpoint.
pub const ListModelsResponse = struct {
    models: ?[]const Model = null,
    /// Token for fetching the next page of results.
    nextPageToken: ?[]const u8 = null,
};

// --- Token Counting ---

/// Request body for the countTokens endpoint.
pub const CountTokensRequest = struct {
    contents: []const Content,
    systemInstruction: ?Content = null,
    tools: ?[]const Tool = null,
    generationConfig: ?GenerationConfig = null,
};

/// Response from the countTokens endpoint.
pub const CountTokensResponse = struct {
    /// Total number of tokens.
    totalTokens: ?i32 = null,
    /// Number of tokens in the cached part of the prompt.
    cachedContentTokenCount: ?i32 = null,
};

// --- Embeddings ---

/// Request body for the embedContent endpoint.
pub const EmbedContentRequest = struct {
    /// The content to generate an embedding for.
    content: ?Content = null,
    /// Type of task for which the embedding will be used.
    taskType: ?[]const u8 = null,
    /// Title for the text. Only applicable when taskType is "RETRIEVAL_DOCUMENT".
    title: ?[]const u8 = null,
    /// Reduced dimension for the output embedding.
    outputDimensionality: ?i32 = null,
};

/// The embedding generated from an input content.
pub const ContentEmbedding = struct {
    /// A list of floats representing the embedding vector.
    values: ?[]const f32 = null,
};

/// Response from the embedContent endpoint.
pub const EmbedContentResponse = struct {
    embedding: ?ContentEmbedding = null,
};

// --- Files ---

/// Processing state of an uploaded file.
pub const FileState = enum {
    STATE_UNSPECIFIED,
    /// The file is being processed and is not yet ready.
    PROCESSING,
    /// The file is processed and ready to use.
    ACTIVE,
    /// The file failed to process.
    FAILED,
};

/// A file uploaded to the API.
pub const File = struct {
    /// Resource name of the file (e.g. "files/abc123").
    name: ?[]const u8 = null,
    /// Human-readable display name (max 512 characters).
    displayName: ?[]const u8 = null,
    /// MIME type of the file.
    mimeType: ?[]const u8 = null,
    /// Size of the file in bytes.
    sizeBytes: ?[]const u8 = null,
    /// Timestamp when the file was created.
    createTime: ?[]const u8 = null,
    /// Timestamp when the file was last updated.
    updateTime: ?[]const u8 = null,
    /// Timestamp when the file will be deleted.
    expirationTime: ?[]const u8 = null,
    /// SHA-256 hash of the uploaded file.
    sha256Hash: ?[]const u8 = null,
    /// URI for referencing the file in API calls.
    uri: ?[]const u8 = null,
    /// Processing state of the file.
    state: ?FileState = null,
};

/// Internal request body for file upload initialization.
pub const UploadFileRequest = struct {
    file: ?UploadFileMetadata = null,
};

/// Metadata for a file being uploaded.
pub const UploadFileMetadata = struct {
    name: ?[]const u8 = null,
    displayName: ?[]const u8 = null,
    mimeType: ?[]const u8 = null,
};

/// Response from the file upload endpoint.
pub const UploadFileResponse = struct {
    file: ?File = null,
};

/// Response from the listFiles endpoint.
pub const ListFilesResponse = struct {
    files: ?[]const File = null,
    /// Token for fetching the next page of results.
    nextPageToken: ?[]const u8 = null,
};

// --- Cached Content ---

/// Usage metadata for cached content.
pub const CachedContentUsageMetadata = struct {
    /// Total number of tokens in the cached content.
    totalTokenCount: ?i32 = null,
};

/// A resource used in LLM queries for users to explicitly specify what to cache.
pub const CachedContent = struct {
    /// Server-generated resource name of the cached content.
    name: ?[]const u8 = null,
    /// User-provided display name.
    displayName: ?[]const u8 = null,
    /// The model to use for cached content.
    model: ?[]const u8 = null,
    /// Timestamp when the cache was created.
    createTime: ?[]const u8 = null,
    /// Timestamp when the cache was last updated.
    updateTime: ?[]const u8 = null,
    /// Timestamp when the cache expires.
    expireTime: ?[]const u8 = null,
    /// Usage metadata for the cached content.
    usageMetadata: ?CachedContentUsageMetadata = null,
};

/// Request body for creating cached content.
pub const CreateCachedContentRequest = struct {
    model: ?[]const u8 = null,
    contents: ?[]const Content = null,
    systemInstruction: ?Content = null,
    tools: ?[]const Tool = null,
    toolConfig: ?ToolConfig = null,
    displayName: ?[]const u8 = null,
    /// Duration string (e.g. "3600s" for 1 hour).
    ttl: ?[]const u8 = null,
    /// RFC 3339 timestamp (e.g. "2026-04-01T00:00:00Z").
    expireTime: ?[]const u8 = null,
};

/// Request body for updating cached content expiration.
pub const UpdateCachedContentRequest = struct {
    /// Duration string (e.g. "3600s" for 1 hour).
    ttl: ?[]const u8 = null,
    /// RFC 3339 timestamp (e.g. "2026-04-01T00:00:00Z").
    expireTime: ?[]const u8 = null,
};

/// Response from the listCachedContents endpoint.
pub const ListCachedContentsResponse = struct {
    cachedContents: ?[]const CachedContent = null,
    /// Token for fetching the next page of results.
    nextPageToken: ?[]const u8 = null,
};

// --- Tests ---

test "GenerateContentRequest serializes to JSON" {
    const parts_arr = [_]Part{.{ .text = "hello" }};
    const contents = [_]Content{.{ .role = "user", .parts = &parts_arr }};
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
    const parts_arr = [_]Part{.{ .text = "hello" }};
    const contents = [_]Content{.{ .role = "user", .parts = &parts_arr }};
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
