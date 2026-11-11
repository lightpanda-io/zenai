const std = @import("std");
const types = @import("types.zig");
const http = @import("../http.zig");
const retry = @import("../retry.zig");
const json = @import("../json.zig");
const continuation = @import("continuation.zig");
pub const decisions = @import("decisions.zig");

pub const RetryPolicy = retry.RetryPolicy;

const Content = types.Content;
const Part = types.Part;
const GenerationConfig = types.GenerationConfig;
const GenerateContentRequest = types.GenerateContentRequest;
const GenerateContentResponse = types.GenerateContentResponse;
const SafetySetting = types.SafetySetting;
const Tool = types.Tool;
const ToolConfig = types.ToolConfig;

/// Gemini API client. Provides access to content generation, embeddings,
/// model management, file uploads, token counting, and cached content.
const Client = @This();

transport: http.Transport,
api_key: []const u8,
/// Base URL override; null derives the backend default (see `VertexConfig`).
base_url: ?[]const u8,
api_version: []const u8,
/// Vertex AI backend config; null targets the Gemini Developer API.
vertex: ?VertexConfig,
/// Used in Vertex project/location mode.
bearer: http.BearerAuth = .{},

/// Google Vertex AI backend configuration. When set on `InitOptions`, the
/// client targets Vertex AI instead of the Gemini Developer API.
pub const VertexConfig = struct {
    /// GCP project ID for project/location mode, where `api_key` must be an
    /// OAuth access token (e.g. `gcloud auth print-access-token`). Null
    /// selects express mode: plain API-key auth against the global endpoint,
    /// with no `projects/{p}/locations/{l}/` path prefix.
    project: ?[]const u8 = null,
    /// GCP location, e.g. "us-central1". "global" (the default) uses
    /// https://aiplatform.googleapis.com; other values use the regional
    /// https://{location}-aiplatform.googleapis.com host. Ignored in
    /// express mode.
    location: []const u8 = "global",
};

/// Options for customizing the API endpoint.
pub const InitOptions = struct {
    /// Base URL override. Null derives the backend default:
    /// https://generativelanguage.googleapis.com for the Developer API,
    /// the (possibly regional) aiplatform host for Vertex.
    base_url: ?[]const u8 = null,
    /// API version override. Null derives "v1beta" (Developer API) or
    /// "v1beta1" (Vertex).
    api_version: ?[]const u8 = null,
    /// Retry policy for transient HTTP failures (5xx, 429, and known
    /// flaky network errors). Pass `RetryPolicy.disabled` to opt out.
    retry_policy: RetryPolicy = .{},
    /// Per-attempt wall-clock bound on non-streaming requests, from an
    /// established connection to the end of the body (connect excluded);
    /// exceeding it fails with `error.Timeout`. `null` waits indefinitely.
    request_timeout_ms: ?u32 = null,
    /// Borrowed `std.http.Client` to share a connection pool with other
    /// clients; see `http.Transport.Options.http_client`. Null owns one.
    http_client: ?*std.http.Client = null,
    /// Target Vertex AI instead of the Gemini Developer API.
    vertex: ?VertexConfig = null,
};

/// Create a new Gemini API client.
/// The `api_key` can be obtained from https://ai.google.dev/gemini-api/docs/api-key
/// In Vertex project/location mode, pass an OAuth access token instead — the
/// `MissingApiKey` guard then means "missing access token".
pub fn init(io: std.Io, allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    return .{
        .transport = .init(io, allocator, .{
            .retry_policy = options.retry_policy,
            .request_timeout_ms = options.request_timeout_ms,
            .http_client = options.http_client,
            .log_tag = "Gemini",
        }),
        .api_key = api_key,
        .base_url = options.base_url,
        .api_version = options.api_version orelse
            (if (options.vertex != null) "v1beta1" else "v1beta"),
        .vertex = options.vertex,
    };
}

/// Release all resources held by the client, including HTTP connections.
pub fn deinit(self: *Client) void {
    self.bearer.deinit(self.transport.allocator);
    self.transport.deinit();
}

/// Per-request options for system instructions, safety, and tools.
pub const RequestOptions = struct {
    /// Instructions to steer the model toward better performance.
    systemInstruction: ?Content = null,
    /// Safety settings to filter generated content.
    safetySettings: ?[]const SafetySetting = null,
    /// Tools the model may use (function calling, code execution, search).
    tools: ?[]const Tool = null,
    /// Configuration for tool usage behavior.
    toolConfig: ?ToolConfig = null,
    /// Settings for prompt and response sanitization.
    modelArmorConfig: ?types.ModelArmorConfig = null,
    /// The service tier to use for the request.
    serviceTier: ?types.ServiceTier = null,
    /// Resume from a candidate that stopped with `CONTINUATION` (its
    /// `continuationToken`, as returned).
    continuationToken: ?[]const u8 = null,
    /// When a response stops with `CONTINUATION` before the model finishes,
    /// send the same request again with its continuation token until it does.
    /// `generateContent` returns the responses merged into one, and streaming
    /// emits the chunks of every request in order, with usage summed over
    /// them. Timeouts and retries apply to each request, and each is billed.
    automatic_continuation: bool = true,
};

fn generateRequest(contents: []const Content, config: ?GenerationConfig, options: RequestOptions) GenerateContentRequest {
    return .{
        .contents = contents,
        .generationConfig = config,
        .systemInstruction = options.systemInstruction,
        .safetySettings = options.safetySettings,
        .tools = options.tools,
        .toolConfig = options.toolConfig,
        .modelArmorConfig = options.modelArmorConfig,
        .serviceTier = options.serviceTier,
        .continuationToken = options.continuationToken,
    };
}

/// Owns the parsed response and its backing memory.
/// Call `deinit()` when done to free all resources.
pub const Response = http.Response;

pub const ApiError = error{ MissingApiKey, UnsupportedByBackend } || http.FetchError;

// --- Internal helpers ---

const default_base_url = "https://generativelanguage.googleapis.com";

/// The auth header for this client's backend: `x-goog-api-key` for the
/// Developer API and Vertex express mode, or `Authorization: Bearer` in
/// Vertex project/location mode where `api_key` holds an OAuth access token.
fn authHeader(self: *Client) error{OutOfMemory}!std.http.Header {
    if (self.vertex) |v| if (v.project != null) return self.bearer.header(self.transport.allocator, self.api_key);
    return .{ .name = "x-goog-api-key", .value = self.api_key };
}

/// Developer-API-only surface: Vertex serves embeddings, files, and cached
/// content through different endpoints (`:predict`, GCS URIs, project-scoped
/// resources) that this client doesn't implement.
fn requireDeveloperApi(self: *const Client) error{UnsupportedByBackend}!void {
    if (self.vertex != null) return error.UnsupportedByBackend;
}

/// Base URL for Developer-API-only methods (which run behind
/// `requireDeveloperApi`, so the Developer default is always correct).
fn devBaseUrl(self: *const Client) []const u8 {
    return self.base_url orelse default_base_url;
}

/// Write the request host: the caller's `base_url` override or the backend
/// default. Vertex regional hosts embed the location.
fn writeHost(self: *const Client, w: *std.Io.Writer) std.Io.Writer.Error!void {
    if (self.base_url) |u| return w.writeAll(u);
    const v = self.vertex orelse return w.writeAll(default_base_url);
    if (v.project != null and !std.mem.eql(u8, v.location, "global")) {
        return w.print("https://{s}-aiplatform.googleapis.com", .{v.location});
    }
    return w.writeAll("https://aiplatform.googleapis.com");
}

const UrlOptions = struct {
    /// RPC verb appended as ":{action}"; null for plain resource GETs.
    action: ?[]const u8 = null,
    /// Prefix "projects/{p}/locations/{l}/" in Vertex project mode. False
    /// for publisher-model GETs (getModel/listModels), which Vertex serves
    /// unprefixed even with project auth.
    project_scope: bool = true,
};

/// Full URL for a model-scoped call, allocated — caller frees. Developer API:
/// "{host}/{version}/models/{model}[:action]". Vertex: optionally prefixed
/// with "projects/{p}/locations/{l}/", the model normalized to a publisher
/// resource ("publishers/google/models/{model}" for bare names, "org/name" to
/// "publishers/{org}/models/{name}"); names already prefixed with
/// "projects/", "models/" or "publishers/" pass through.
fn modelUrl(self: *const Client, model: []const u8, opts: UrlOptions) error{OutOfMemory}![]u8 {
    var buf: std.Io.Writer.Allocating = .init(self.transport.allocator);
    errdefer buf.deinit();
    self.writeModelUrl(&buf.writer, model, opts) catch return error.OutOfMemory;
    return buf.toOwnedSlice();
}

fn writeModelUrl(self: *const Client, w: *std.Io.Writer, model: []const u8, opts: UrlOptions) std.Io.Writer.Error!void {
    try self.writeHost(w);
    try w.print("/{s}/", .{self.api_version});
    if (self.vertex) |v| {
        if (opts.project_scope and !std.mem.startsWith(u8, model, "projects/")) {
            if (v.project) |p| try w.print("projects/{s}/locations/{s}/", .{ p, v.location });
        }
        if (std.mem.startsWith(u8, model, "projects/") or
            std.mem.startsWith(u8, model, "models/") or
            std.mem.startsWith(u8, model, "publishers/"))
        {
            try w.writeAll(model);
        } else if (std.mem.findScalar(u8, model, '/')) |slash| {
            try w.print("publishers/{s}/models/{s}", .{ model[0..slash], model[slash + 1 ..] });
        } else {
            try w.print("publishers/google/models/{s}", .{model});
        }
    } else {
        try w.print("models/{s}", .{model});
    }
    if (opts.action) |a| try w.print(":{s}", .{a});
}

fn fetchGet(self: *Client, url: []const u8, comptime T: type) ApiError!Response(T) {
    const auth = [1]std.http.Header{try self.authHeader()};
    return self.transport.getJson(url, &auth, T);
}

fn fetchPost(self: *Client, url: []const u8, body: anytype, comptime T: type) ApiError!Response(T) {
    const auth = [1]std.http.Header{try self.authHeader()};
    return self.transport.postJson(url, &auth, body, T);
}

/// Pagination options for list operations.
pub const ListOptions = http.ListOptions;

// --- Generate Content ---

/// Generate content from the model given a conversation history and configuration.
/// See `RequestOptions.automatic_continuation` for responses the model has
/// not finished.
pub fn generateContent(
    self: *Client,
    model: []const u8,
    contents: []const Content,
    config: ?GenerationConfig,
    options: RequestOptions,
) ApiError!Response(GenerateContentResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try self.modelUrl(model, .{ .action = "generateContent" });
    defer self.transport.allocator.free(url);

    var response = try self.fetchPost(url, generateRequest(contents, config, options), GenerateContentResponse);
    if (!options.automatic_continuation) return response;
    errdefer response.deinit();

    // Merged into the first response's arena, so the result owns every hop.
    const arena = response.parsed.arena.allocator();
    var request = generateRequest(contents, config, options);
    // The latest response decides whether to go on, not the merge: a request
    // without candidates ends the generation.
    var token = continuation.resumeToken(response.value);
    while (token) |t| {
        request.continuationToken = t;
        var hop = try self.fetchPost(url, request, GenerateContentResponse);
        defer hop.deinit();
        const curr = try std.json.parseFromSliceLeaky(GenerateContentResponse, arena, hop.json_buf.written(), .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        });
        response.value = try continuation.merge(arena, response.value, curr);
        token = continuation.resumeToken(curr);
    }
    return response;
}

/// Convenience: generate content from a single text prompt.
pub fn generateContentFromText(
    self: *Client,
    model: []const u8,
    prompt: []const u8,
    config: ?GenerationConfig,
    options: RequestOptions,
) ApiError!Response(GenerateContentResponse) {
    const parts = [_]Part{.{ .text = prompt }};
    const contents = [_]Content{.{ .role = "user", .parts = &parts }};
    return self.generateContent(model, &contents, config, options);
}

// --- Streaming ---

pub const StreamError = error{MissingApiKey} || http.SseError;

/// Generate content with streaming via Server-Sent Events.
/// The `callback` is invoked for each chunk as it arrives.
/// Pass a `context` value to carry state into the callback.
/// See `RequestOptions.automatic_continuation` for responses the model has
/// not finished; a stream cut off after a checkpoint also resumes from it.
pub fn generateContentStream(
    self: *Client,
    model: []const u8,
    contents: []const Content,
    config: ?GenerationConfig,
    options: RequestOptions,
    context: anytype,
    callback: *const fn (@TypeOf(context), GenerateContentResponse) void,
) StreamError!void {
    if (self.api_key.len == 0) return error.MissingApiKey;

    const url = try self.modelUrl(model, .{ .action = "streamGenerateContent?alt=sse" });
    defer self.transport.allocator.free(url);

    const auth = [1]std.http.Header{try self.authHeader()};
    var request = generateRequest(contents, config, options);
    if (!options.automatic_continuation) {
        return self.transport.streamSse(url, &auth, request, GenerateContentResponse, context, callback);
    }

    const Hops = continuation.Stream(@TypeOf(context), StreamError);
    var hops: Hops = .init(self.transport.allocator, context, callback);
    defer hops.deinit();
    while (true) {
        var hop_err: ?StreamError = null;
        self.transport.streamSse(url, &auth, request, GenerateContentResponse, &hops, Hops.onChunk) catch |err| {
            if (err == error.OutOfMemory) return err;
            // A rejected request is not retried here.
            if (err == error.ApiError and !hops.receivedChunk()) return err;
            hop_err = err;
        };
        if (hops.err) |e| return e;
        // An interrupted read ends the body like a clean close; never resume it.
        if (self.transport.interrupted()) return hop_err orelse {};
        request.continuationToken = try hops.endHop(hop_err) orelse return;
    }
}

/// Convenience: stream content generation from a single text prompt.
pub fn generateContentStreamFromText(
    self: *Client,
    model: []const u8,
    prompt: []const u8,
    config: ?GenerationConfig,
    options: RequestOptions,
    context: anytype,
    callback: *const fn (@TypeOf(context), GenerateContentResponse) void,
) StreamError!void {
    const parts = [_]Part{.{ .text = prompt }};
    const contents = [_]Content{.{ .role = "user", .parts = &parts }};
    return self.generateContentStream(model, &contents, config, options, context, callback);
}

/// Reassembles a streamed generation into the same `GenerateContentResponse` the
/// non-streaming `generateContent` returns. Text parts are concatenated (and
/// forwarded to `on_text_fn`); function-call parts arrive complete per chunk;
/// the last `finishReason` and `usageMetadata` win. All storage lives in `arena`.
pub const StreamAccumulator = struct {
    arena: std.mem.Allocator,
    on_text_ctx: *anyopaque,
    on_text_fn: *const fn (*anyopaque, []const u8) void,
    text: std.ArrayListUnmanaged(u8) = .empty,
    fn_parts: std.ArrayListUnmanaged(Part) = .empty,
    finish_reason: ?types.FinishReason = null,
    usage: ?types.UsageMetadata = null,
    err: ?error{OutOfMemory} = null,

    pub fn init(
        arena: std.mem.Allocator,
        on_text_ctx: *anyopaque,
        on_text_fn: *const fn (*anyopaque, []const u8) void,
    ) StreamAccumulator {
        return .{ .arena = arena, .on_text_ctx = on_text_ctx, .on_text_fn = on_text_fn };
    }

    pub fn onEvent(self: *StreamAccumulator, chunk: GenerateContentResponse) void {
        self.handle(chunk) catch |e| {
            self.err = e;
        };
    }

    fn handle(self: *StreamAccumulator, chunk: GenerateContentResponse) error{OutOfMemory}!void {
        // `text()` already skips `thought` parts, so this never leaks reasoning.
        if (chunk.text()) |txt| {
            if (txt.len > 0) {
                try self.text.appendSlice(self.arena, txt);
                self.on_text_fn(self.on_text_ctx, txt);
            }
        }
        if (chunk.usageMetadata) |u| self.usage = u;
        const candidates = chunk.candidates orelse return;
        if (candidates.len == 0) return;
        if (candidates[0].finishReason) |fr| self.finish_reason = fr;
        const content = candidates[0].content orelse return;
        for (content.parts) |p| {
            const fc = p.functionCall orelse continue;
            // Strings/values point into the transient parsed chunk; dupe them.
            try self.fn_parts.append(self.arena, .{
                .functionCall = .{
                    .id = if (fc.id) |id| try self.arena.dupe(u8, id) else null,
                    .name = if (fc.name) |n| try self.arena.dupe(u8, n) else null,
                    .args = if (fc.args) |v| try json.dupeValue(self.arena, v) else null,
                },
                .thoughtSignature = if (p.thoughtSignature) |ts| try self.arena.dupe(u8, ts) else null,
            });
        }
    }

    /// Assemble the accumulated deltas into a `GenerateContentResponse`. Call
    /// once, after the stream completes and `err` is clear.
    pub fn response(self: *StreamAccumulator) error{OutOfMemory}!GenerateContentResponse {
        var parts: std.ArrayListUnmanaged(Part) = .empty;
        if (self.text.items.len > 0) {
            try parts.append(self.arena, .{ .text = self.text.items });
        }
        try parts.appendSlice(self.arena, self.fn_parts.items);
        const candidate = types.Candidate{
            .content = .{ .parts = parts.items },
            .finishReason = self.finish_reason,
        };
        const candidates = try self.arena.dupe(types.Candidate, &.{candidate});
        return .{ .candidates = candidates, .usageMetadata = self.usage };
    }
};

// --- Model Management ---

/// Get metadata about a specific model (token limits, supported methods, etc.).
pub fn getModel(self: *Client, model: []const u8) !Response(types.Model) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try self.modelUrl(model, .{ .project_scope = false });
    defer self.transport.allocator.free(url);
    return self.fetchGet(url, types.Model);
}

/// Whether `m` is a chat/text-generation model — i.e. it advertises
/// `generateContent` in `supportedGenerationMethods`. Drops embeddings
/// (embedContent), veo/imagen (predict), and aqa (generateAnswer).
///
/// Multimodal models that can also do images (e.g. `gemini-2.5-flash-image`,
/// "nano banana") are kept, since they support `generateContent` and can
/// read and generate text. Music-only (lyria) and speech-only (tts) variants
/// also pass this filter — Google's API doesn't distinguish output modality
/// from generation method, so the signal isn't fine-grained enough to drop
/// them without resorting to name heuristics.
pub fn isChatModel(m: types.Model) bool {
    const methods = m.supportedGenerationMethods orelse return false;
    for (methods) |meth| {
        if (std.mem.eql(u8, meth, "generateContent")) return true;
    }
    return false;
}

/// List available models. Use `ListOptions` to paginate through results.
/// On Vertex this lists Google publisher models; the entries carry no
/// `supportedGenerationMethods` and arrive under `publisherModels`. Google
/// rejects API keys for this endpoint (HTTP 401), so on Vertex it needs
/// project/location mode — express mode can generate but not list.
pub fn listModels(self: *Client, options: ListOptions) !Response(types.ListModelsResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    var base_buf: std.Io.Writer.Allocating = .init(self.transport.allocator);
    defer base_buf.deinit();
    self.writeHost(&base_buf.writer) catch return error.OutOfMemory;
    base_buf.writer.print("/{s}/{s}", .{
        self.api_version,
        if (self.vertex != null) "publishers/google/models" else "models",
    }) catch return error.OutOfMemory;
    const base = base_buf.written();
    const url = try http.appendListParams(self.transport.allocator, base, options);
    defer self.transport.allocator.free(url);
    return self.fetchGet(url, types.ListModelsResponse);
}

// --- Token Counting ---

/// Count the number of tokens in the provided contents.
pub fn countTokens(
    self: *Client,
    model: []const u8,
    contents: []const Content,
    config: ?GenerationConfig,
    options: RequestOptions,
) !Response(types.CountTokensResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try self.modelUrl(model, .{ .action = "countTokens" });
    defer self.transport.allocator.free(url);
    return self.fetchPost(url, types.CountTokensRequest{
        .contents = contents,
        .systemInstruction = options.systemInstruction,
        .tools = options.tools,
        .generationConfig = config,
    }, types.CountTokensResponse);
}

/// Convenience: count tokens for a single text string.
pub fn countTokensFromText(
    self: *Client,
    model: []const u8,
    prompt: []const u8,
) !Response(types.CountTokensResponse) {
    const parts = [_]Part{.{ .text = prompt }};
    const contents = [_]Content{.{ .role = "user", .parts = &parts }};
    return self.countTokens(model, &contents, null, .{});
}

// --- Embeddings ---

/// Configuration for embedding generation.
pub const EmbedConfig = struct {
    /// Type of task (e.g. "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY").
    taskType: ?[]const u8 = null,
    /// Title for the text. Only applicable when taskType is "RETRIEVAL_DOCUMENT".
    title: ?[]const u8 = null,
    /// Reduced dimension for the output embedding vector.
    outputDimensionality: ?i32 = null,
};

/// Generate an embedding for the provided content using the specified model.
pub fn embedContent(
    self: *Client,
    model: []const u8,
    content: Content,
    config: EmbedConfig,
) !Response(types.EmbedContentResponse) {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try self.transport.allocator.print("{s}/{s}/models/{s}:embedContent", .{ self.devBaseUrl(), self.api_version, model });
    defer self.transport.allocator.free(url);
    return self.fetchPost(url, types.EmbedContentRequest{
        .content = content,
        .taskType = config.taskType,
        .title = config.title,
        .outputDimensionality = config.outputDimensionality,
    }, types.EmbedContentResponse);
}

/// Convenience: generate an embedding for a single text string.
pub fn embedText(
    self: *Client,
    model: []const u8,
    text_content: []const u8,
) !Response(types.EmbedContentResponse) {
    const parts = [_]Part{.{ .text = text_content }};
    const content = Content{ .parts = &parts };
    return self.embedContent(model, content, .{});
}

// --- Decisions ---

/// Evaluate every question in `questions` against the shared `input` in one
/// call, getting a calibrated answer back under each question id. The input is
/// ingested once, so batch questions into one request. `model` may be bare
/// ("gemini-decision-1") or "models/…". Developer API only.
pub fn createDecision(
    self: *Client,
    model: []const u8,
    input: decisions.Input,
    questions: decisions.Questions,
) !Response(decisions.CreateDecisionResponse) {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try self.transport.allocator.print("{s}/{s}/decisions", .{ self.devBaseUrl(), self.api_version });
    defer self.transport.allocator.free(url);

    const full_model = if (std.mem.startsWith(u8, model, "models/"))
        try self.transport.allocator.dupe(u8, model)
    else
        try self.transport.allocator.print("models/{s}", .{model});
    defer self.transport.allocator.free(full_model);

    return self.fetchPost(url, decisions.CreateDecisionRequest{
        .input = input,
        .model = full_model,
        .questions = questions,
    }, decisions.CreateDecisionResponse);
}

test {
    _ = continuation;
    _ = decisions;
}

test "Client init and deinit" {
    var client = Client.init(std.testing.io, std.testing.allocator, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expect(client.base_url == null);
    try std.testing.expectEqualStrings("v1beta", client.api_version);
}

test "init derives the api version per backend, override wins" {
    var vertex = Client.init(std.testing.io, std.testing.allocator, "tok", .{ .vertex = .{} });
    defer vertex.deinit();
    try std.testing.expectEqualStrings("v1beta1", vertex.api_version);

    var pinned = Client.init(std.testing.io, std.testing.allocator, "tok", .{ .api_version = "v1", .vertex = .{} });
    defer pinned.deinit();
    try std.testing.expectEqualStrings("v1", pinned.api_version);
}

test "modelUrl: developer API" {
    var client = Client.init(std.testing.io, std.testing.allocator, "key", .{});
    defer client.deinit();
    const url = try client.modelUrl("gemini-2.5-flash", .{ .action = "generateContent" });
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        url,
    );
}

test "modelUrl: vertex express mode" {
    var client = Client.init(std.testing.io, std.testing.allocator, "key", .{ .vertex = .{} });
    defer client.deinit();
    const url = try client.modelUrl("gemini-2.5-flash", .{ .action = "generateContent" });
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings(
        "https://aiplatform.googleapis.com/v1beta1/publishers/google/models/gemini-2.5-flash:generateContent",
        url,
    );
}

test "modelUrl: vertex project mode, global and regional" {
    var global = Client.init(std.testing.io, std.testing.allocator, "tok", .{ .vertex = .{ .project = "my-proj" } });
    defer global.deinit();
    const global_url = try global.modelUrl("gemini-2.5-flash", .{ .action = "countTokens" });
    defer std.testing.allocator.free(global_url);
    try std.testing.expectEqualStrings(
        "https://aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/global/publishers/google/models/gemini-2.5-flash:countTokens",
        global_url,
    );

    var regional = Client.init(std.testing.io, std.testing.allocator, "tok", .{
        .vertex = .{ .project = "my-proj", .location = "us-central1" },
    });
    defer regional.deinit();
    const regional_url = try regional.modelUrl("gemini-2.5-flash", .{ .action = "generateContent" });
    defer std.testing.allocator.free(regional_url);
    try std.testing.expectEqualStrings(
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent",
        regional_url,
    );
}

test "modelUrl: vertex model name normalization" {
    var client = Client.init(std.testing.io, std.testing.allocator, "tok", .{ .vertex = .{ .project = "p" } });
    defer client.deinit();

    // Full resource names pass through with no project prefix added.
    const full = try client.modelUrl("projects/p/locations/global/publishers/google/models/m", .{ .action = "generateContent" });
    defer std.testing.allocator.free(full);
    try std.testing.expectEqualStrings(
        "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/publishers/google/models/m:generateContent",
        full,
    );

    // Publisher paths pass through (still project-scoped).
    const published = try client.modelUrl("publishers/google/models/m", .{ .action = "generateContent" });
    defer std.testing.allocator.free(published);
    try std.testing.expectEqualStrings(
        "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/publishers/google/models/m:generateContent",
        published,
    );

    // "org/name" maps to that org's publisher path.
    const org = try client.modelUrl("meta/llama-x", .{ .action = "generateContent" });
    defer std.testing.allocator.free(org);
    try std.testing.expectEqualStrings(
        "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/publishers/meta/models/llama-x:generateContent",
        org,
    );

    // Publisher-model GETs skip the project scope.
    const unscoped = try client.modelUrl("gemini-2.5-flash", .{ .project_scope = false });
    defer std.testing.allocator.free(unscoped);
    try std.testing.expectEqualStrings(
        "https://aiplatform.googleapis.com/v1beta1/publishers/google/models/gemini-2.5-flash",
        unscoped,
    );
}

test "modelUrl: explicit base_url overrides the backend host" {
    var client = Client.init(std.testing.io, std.testing.allocator, "tok", .{
        .base_url = "http://localhost:8080",
        .vertex = .{ .project = "p", .location = "us-central1" },
    });
    defer client.deinit();
    const url = try client.modelUrl("m", .{ .action = "generateContent" });
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings(
        "http://localhost:8080/v1beta1/projects/p/locations/us-central1/publishers/google/models/m:generateContent",
        url,
    );
}

test "authHeader: api key vs cached bearer token" {
    var dev = Client.init(std.testing.io, std.testing.allocator, "key", .{});
    defer dev.deinit();
    const dev_header = try dev.authHeader();
    try std.testing.expectEqualStrings("x-goog-api-key", dev_header.name);
    try std.testing.expectEqualStrings("key", dev_header.value);

    var express = Client.init(std.testing.io, std.testing.allocator, "key", .{ .vertex = .{} });
    defer express.deinit();
    try std.testing.expectEqualStrings("x-goog-api-key", (try express.authHeader()).name);

    var project = Client.init(std.testing.io, std.testing.allocator, "tok", .{ .vertex = .{ .project = "p" } });
    defer project.deinit();
    const first = try project.authHeader();
    try std.testing.expectEqualStrings("Authorization", first.name);
    try std.testing.expectEqualStrings("Bearer tok", first.value);
    // Built once and cached; deinit frees it (leak-checked by the test allocator).
    const second = try project.authHeader();
    try std.testing.expectEqual(first.value.ptr, second.value.ptr);
}

test "vertex: developer-only methods are unsupported" {
    var client = Client.init(std.testing.io, std.testing.allocator, "tok", .{ .vertex = .{ .project = "p" } });
    defer client.deinit();
    try std.testing.expectError(error.UnsupportedByBackend, client.embedText("m", "hi"));
    try std.testing.expectError(error.UnsupportedByBackend, client.createDecision("m", .{ .text = "hi" }, .{}));
}

test "isChatModel keeps text/chat models" {
    const T = struct {
        fn m(name: []const u8, methods: []const []const u8) types.Model {
            return .{ .name = name, .supportedGenerationMethods = methods };
        }
    };
    const generate = [_][]const u8{ "generateContent", "countTokens" };
    try std.testing.expect(isChatModel(T.m("models/gemini-2.5-flash", &generate)));
    try std.testing.expect(isChatModel(T.m("models/gemini-2.5-pro", &generate)));
    try std.testing.expect(isChatModel(T.m("models/gemma-4-31b-it", &generate)));
    try std.testing.expect(isChatModel(T.m("models/deep-research-preview-04-2026", &generate)));
}

test "isChatModel drops non-generateContent methods" {
    const T = struct {
        fn m(name: []const u8, methods: []const []const u8) types.Model {
            return .{ .name = name, .supportedGenerationMethods = methods };
        }
    };
    const embed = [_][]const u8{"embedContent"};
    const predict = [_][]const u8{"predict"};
    const generate_answer = [_][]const u8{"generateAnswer"};

    try std.testing.expect(!isChatModel(T.m("models/gemini-embedding-001", &embed)));
    try std.testing.expect(!isChatModel(T.m("models/imagen-4.0-generate-001", &predict)));
    try std.testing.expect(!isChatModel(T.m("models/veo-3.0-generate-001", &predict)));
    try std.testing.expect(!isChatModel(T.m("models/aqa", &generate_answer)));
}

test "isChatModel keeps multimodal text+image variants" {
    const T = struct {
        fn m(name: []const u8, methods: []const []const u8) types.Model {
            return .{ .name = name, .supportedGenerationMethods = methods };
        }
    };
    const generate = [_][]const u8{"generateContent"};
    // These can read and generate text, so they're agent-capable even
    // though they also handle images.
    try std.testing.expect(isChatModel(T.m("models/gemini-2.5-flash-image", &generate)));
    try std.testing.expect(isChatModel(T.m("models/nano-banana-pro-preview", &generate)));
}

const TextProbe = struct {
    count: usize = 0,
    text: std.ArrayListUnmanaged(u8) = .empty,

    fn cb(ptr: *anyopaque, delta: []const u8) void {
        const self: *TextProbe = @ptrCast(@alignCast(ptr));
        self.count += 1;
        self.text.appendSlice(std.testing.allocator, delta) catch {};
    }
};

test "StreamAccumulator reassembles text and a function call into a GenerateContentResponse" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var probe = TextProbe{};
    defer probe.text.deinit(std.testing.allocator);

    var args = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"q\":\"birds\"}", .{});
    defer args.deinit();

    var acc = StreamAccumulator.init(arena.allocator(), &probe, TextProbe.cb);
    const text_a = [_]Part{.{ .text = "Search" }};
    const text_b = [_]Part{.{ .text = "ing." }};
    const fc = [_]Part{.{ .functionCall = .{ .name = "search", .args = args.value } }};
    const chunks = [_]GenerateContentResponse{
        .{ .candidates = &.{.{ .content = .{ .parts = &text_a } }} },
        .{ .candidates = &.{.{ .content = .{ .parts = &text_b } }} },
        .{ .candidates = &.{.{ .content = .{ .parts = &fc }, .finishReason = .STOP }} },
        .{ .usageMetadata = .{ .promptTokenCount = 8, .candidatesTokenCount = 4, .totalTokenCount = 12 } },
    };
    for (chunks) |c| acc.onEvent(c);
    try std.testing.expect(acc.err == null);

    const resp = try acc.response();
    try std.testing.expectEqual(@as(usize, 2), probe.count);
    try std.testing.expectEqualStrings("Searching.", resp.text().?);
    try std.testing.expectEqual(@as(?i32, 4), resp.usageMetadata.?.candidatesTokenCount);

    const call = resp.firstFunctionCall().?;
    try std.testing.expectEqualStrings("search", call.name.?);
    try std.testing.expectEqualStrings("birds", call.args.?.object.get("q").?.string);
}

/// Test server for the continuation tests: answers one request per entry of
/// `bodies` in order, recording each request body, with keep-alive so the
/// client may reuse the connection.
const ContinuationServer = struct {
    server: std.Io.net.Server,
    bodies: []const []const u8,
    content_type: []const u8,
    requests: [4][]u8 = undefined,
    served: usize = 0,

    fn run(self: *ContinuationServer) void {
        const io = std.testing.io;
        while (self.served < self.bodies.len) {
            const stream = self.server.accept(io) catch return;
            defer stream.close(io);
            var read_buf: [8192]u8 = undefined;
            var write_buf: [1024]u8 = undefined;
            var reader = stream.reader(io, &read_buf);
            var writer = stream.writer(io, &write_buf);
            var http_server: std.http.Server = .init(&reader.interface, &writer.interface);
            while (self.served < self.bodies.len) {
                var request = http_server.receiveHead() catch break;
                var body_buf: [8192]u8 = undefined;
                const body_reader = request.readerExpectNone(&body_buf);
                const body = body_reader.allocRemaining(std.testing.allocator, .unlimited) catch return;
                self.requests[self.served] = body;
                const out = self.bodies[self.served];
                self.served += 1;
                request.respond(out, .{
                    .keep_alive = true,
                    .extra_headers = &.{.{ .name = "content-type", .value = self.content_type }},
                }) catch return;
            }
        }
    }

    fn deinit(self: *ContinuationServer) void {
        for (self.requests[0..self.served]) |r| std.testing.allocator.free(r);
        self.server.deinit(std.testing.io);
    }
};

fn continuationClient(server: *ContinuationServer) !Client {
    var buf: [64]u8 = undefined;
    const base = try std.fmt.bufPrint(&buf, "http://127.0.0.1:{d}", .{server.server.socket.address.getPort()});
    var client = Client.init(std.testing.io, std.testing.allocator, "key", .{ .retry_policy = .disabled, .request_timeout_ms = 2000 });
    client.base_url = try std.testing.allocator.dupe(u8, base);
    return client;
}

test "generateContent continues a CONTINUATION response and merges the hops" {
    const loopback: std.Io.net.IpAddress = try .parse("127.0.0.1", 0);
    var server: ContinuationServer = .{
        .server = try loopback.listen(std.testing.io, .{}),
        .content_type = "application/json",
        .bodies = &.{
            \\{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello, "}]},"finishReason":"CONTINUATION","continuationToken":"Y2sx"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"totalTokenCount":12}}
            ,
            \\{"candidates":[{"content":{"role":"model","parts":[{"text":"world."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":2,"totalTokenCount":14},"responseId":"r2"}
            ,
        },
    };
    defer server.deinit();
    const thread = try std.Thread.spawn(.{}, ContinuationServer.run, .{&server});
    defer thread.join();

    var client = try continuationClient(&server);
    defer std.testing.allocator.free(client.base_url.?);
    defer client.deinit();

    var response = try client.generateContentFromText("m", "hi", null, .{});
    defer response.deinit();

    try std.testing.expectEqual(@as(usize, 2), server.served);
    try std.testing.expect(std.mem.find(u8, server.requests[0], "continuationToken") == null);
    try std.testing.expect(std.mem.find(u8, server.requests[1], "\"continuationToken\":\"Y2sx\"") != null);

    const parts = response.value.parts().?;
    try std.testing.expectEqual(@as(usize, 2), parts.len);
    try std.testing.expectEqualStrings("Hello, ", parts[0].text.?);
    try std.testing.expectEqualStrings("world.", parts[1].text.?);
    try std.testing.expect(response.value.candidates.?[0].finishReason.? == .STOP);
    try std.testing.expectEqual(@as(?i32, 26), response.value.usageMetadata.?.totalTokenCount);
    try std.testing.expectEqualStrings("r2", response.value.responseId.?);
}

test "generateContent stops when a continuation returns no candidates" {
    const loopback: std.Io.net.IpAddress = try .parse("127.0.0.1", 0);
    var server: ContinuationServer = .{
        .server = try loopback.listen(std.testing.io, .{}),
        .content_type = "application/json",
        .bodies = &.{
            \\{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello, "}]},"finishReason":"CONTINUATION","continuationToken":"Y2sx"}]}
            ,
            \\{"promptFeedback":{"blockReason":"SAFETY"}}
            ,
        },
    };
    defer server.deinit();
    const thread = try std.Thread.spawn(.{}, ContinuationServer.run, .{&server});
    defer thread.join();

    var client = try continuationClient(&server);
    defer std.testing.allocator.free(client.base_url.?);
    defer client.deinit();

    var response = try client.generateContentFromText("m", "hi", null, .{});
    defer response.deinit();

    try std.testing.expectEqual(@as(usize, 2), server.served);
    const cand = response.value.candidates.?[0];
    try std.testing.expectEqualStrings("Hello, ", cand.content.?.parts[0].text.?);
    try std.testing.expect(cand.finishReason == null);
    try std.testing.expect(cand.continuationToken == null);
    try std.testing.expectEqualStrings("SAFETY", response.value.promptFeedback.?.blockReason.?);
}

test "generateContent with automatic_continuation off returns the first response" {
    const loopback: std.Io.net.IpAddress = try .parse("127.0.0.1", 0);
    var server: ContinuationServer = .{
        .server = try loopback.listen(std.testing.io, .{}),
        .content_type = "application/json",
        .bodies = &.{
            \\{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello, "}]},"finishReason":"CONTINUATION","continuationToken":"Y2sx"}]}
            ,
        },
    };
    defer server.deinit();
    const thread = try std.Thread.spawn(.{}, ContinuationServer.run, .{&server});
    defer thread.join();

    var client = try continuationClient(&server);
    defer std.testing.allocator.free(client.base_url.?);
    defer client.deinit();

    var response = try client.generateContentFromText("m", "hi", null, .{ .automatic_continuation = false });
    defer response.deinit();
    try std.testing.expect(response.value.candidates.?[0].finishReason.? == .CONTINUATION);
    try std.testing.expectEqualStrings("Y2sx", response.value.candidates.?[0].continuationToken.?);
}

test "generateContentStream continues across requests into one accumulated response" {
    const loopback: std.Io.net.IpAddress = try .parse("127.0.0.1", 0);
    var server: ContinuationServer = .{
        .server = try loopback.listen(std.testing.io, .{}),
        .content_type = "text/event-stream",
        .bodies = &.{
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello, \"}]},\"finishReason\":\"CONTINUATION\",\"continuationToken\":\"Y2sx\"}],\"usageMetadata\":{\"totalTokenCount\":12}}\n\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"world.\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"totalTokenCount\":14}}\n\n",
        },
    };
    defer server.deinit();
    const thread = try std.Thread.spawn(.{}, ContinuationServer.run, .{&server});
    defer thread.join();

    var client = try continuationClient(&server);
    defer std.testing.allocator.free(client.base_url.?);
    defer client.deinit();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var probe = TextProbe{};
    defer probe.text.deinit(std.testing.allocator);
    var acc = StreamAccumulator.init(arena.allocator(), &probe, TextProbe.cb);
    try client.generateContentStreamFromText("m", "hi", null, .{}, &acc, StreamAccumulator.onEvent);
    try std.testing.expect(acc.err == null);

    try std.testing.expectEqual(@as(usize, 2), server.served);
    try std.testing.expect(std.mem.find(u8, server.requests[1], "\"continuationToken\":\"Y2sx\"") != null);
    const resp = try acc.response();
    try std.testing.expectEqualStrings("Hello, world.", resp.text().?);
    try std.testing.expect(resp.candidates.?[0].finishReason.? == .STOP);
    try std.testing.expectEqual(@as(?i32, 26), resp.usageMetadata.?.totalTokenCount);
}

test "createDecision posts the questions and parses the answers" {
    const loopback: std.Io.net.IpAddress = try .parse("127.0.0.1", 0);
    var server: ContinuationServer = .{
        .server = try loopback.listen(std.testing.io, .{}),
        .content_type = "application/json",
        .bodies = &.{
            \\{"answers":{"urgent":{"type":"noul","noul":0.9,"confidence":0.9}},"usage":{"total_tokens":42}}
            ,
        },
    };
    defer server.deinit();
    const thread = try std.Thread.spawn(.{}, ContinuationServer.run, .{&server});
    defer thread.join();

    var client = try continuationClient(&server);
    defer std.testing.allocator.free(client.base_url.?);
    defer client.deinit();

    var response = try client.createDecision("gemini-decision-1", .{ .text = "Server is down!" }, .init(&.{
        .{ .key = "urgent", .value = .noulText("Is this urgent?") },
    }));
    defer response.deinit();

    try std.testing.expectEqualStrings(
        \\{"input":"Server is down!","model":"models/gemini-decision-1","questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"}}}
    , server.requests[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), response.value.noul("urgent").?, 1e-12);
    try std.testing.expectEqual(@as(?u32, 42), response.value.usage.?.total_tokens);
}
