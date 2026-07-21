const std = @import("std");
const types = @import("types.zig");
const http = @import("../http.zig");
const retry = @import("../retry.zig");
const json = @import("../json.zig");

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

allocator: std.mem.Allocator,
api_key: []const u8,
/// Base URL override; null derives the backend default (see `VertexConfig`).
base_url: ?[]const u8,
api_version: []const u8,
/// Vertex AI backend config; null targets the Gemini Developer API.
vertex: ?VertexConfig,
http_client: std.http.Client,
/// Retry policy applied to every non-streaming request.
retry_policy: RetryPolicy,
/// Cached "Bearer {token}" header value for Vertex project/location mode.
/// Built on first request; owned, freed in `deinit`.
bearer_value: ?[]u8 = null,
/// Human-readable message from the most recent API error, owned by the client
/// and freed on the next failure or `deinit`. Set on `error.ApiError`.
last_error_message: ?[]u8 = null,
last_error_status: ?u10 = null,
/// Set by the host so a SIGINT can abort an in-flight request mid-read.
interrupt: ?*http.Interrupt = null,

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
    /// Target Vertex AI instead of the Gemini Developer API.
    vertex: ?VertexConfig = null,
};

/// Create a new Gemini API client.
/// The `api_key` can be obtained from https://ai.google.dev/gemini-api/docs/api-key
/// In Vertex project/location mode, pass an OAuth access token instead — the
/// `MissingApiKey` guard then means "missing access token".
pub fn init(allocator: std.mem.Allocator, io: std.Io, api_key: []const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .api_version = options.api_version orelse
            (if (options.vertex != null) "v1beta1" else "v1beta"),
        .vertex = options.vertex,
        .http_client = .{ .allocator = allocator, .io = io },
        .retry_policy = options.retry_policy,
        .last_error_message = null,
        .last_error_status = null,
    };
}

/// Release all resources held by the client, including HTTP connections.
pub fn deinit(self: *Client) void {
    if (self.last_error_message) |m| self.allocator.free(m);
    if (self.bearer_value) |b| self.allocator.free(b);
    self.http_client.deinit();
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
};

/// Owns the parsed response and its backing memory.
/// Call `deinit()` when done to free all resources.
pub const Response = http.Response;

pub const ApiError = error{
    ApiError,
    MissingApiKey,
    EmptyResponse,
    UnsupportedByBackend,
} || std.http.Client.FetchError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

// --- Internal helpers ---

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error_status = status_code;
    if (self.last_error_message) |m| {
        self.allocator.free(m);
        self.last_error_message = null;
    }
    if (body.len > 0) {
        std.log.err("Gemini API error (HTTP {d}): {s}", .{ status_code, body });
        self.last_error_message = http.extractErrorMessage(self.allocator, body);
    }
}

const default_base_url = "https://generativelanguage.googleapis.com";

/// The auth header for this client's backend: `x-goog-api-key` for the
/// Developer API and Vertex express mode, or `Authorization: Bearer` in
/// Vertex project/location mode where `api_key` holds an OAuth access token.
/// The Bearer value is built once and cached on the client.
fn authHeader(self: *Client) error{OutOfMemory}!std.http.Header {
    if (self.vertex) |v| if (v.project != null) {
        const value = self.bearer_value orelse blk: {
            const b = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_key});
            self.bearer_value = b;
            break :blk b;
        };
        return .{ .name = "authorization", .value = value };
    };
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
    var buf: std.Io.Writer.Allocating = .init(self.allocator);
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
    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, .{
        .location = .{ .url = url },
        .extra_headers = &auth,
    }, T, self);
}

fn fetchPost(self: *Client, url: []const u8, body: anytype, comptime T: type) ApiError!Response(T) {
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    const auth = [1]std.http.Header{try self.authHeader()};
    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, .{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload_buf.written(),
        .extra_headers = &auth,
        .headers = .{ .content_type = .{ .override = "application/json" } },
    }, T, self);
}

/// Pagination options for list operations.
pub const ListOptions = http.ListOptions;

// --- Generate Content ---

/// Generate content from the model given a conversation history and configuration.
pub fn generateContent(
    self: *Client,
    model: []const u8,
    contents: []const Content,
    config: ?GenerationConfig,
    options: RequestOptions,
) ApiError!Response(GenerateContentResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try self.modelUrl(model, .{ .action = "generateContent" });
    defer self.allocator.free(url);

    return self.fetchPost(url, GenerateContentRequest{
        .contents = contents,
        .generationConfig = config,
        .systemInstruction = options.systemInstruction,
        .safetySettings = options.safetySettings,
        .tools = options.tools,
        .toolConfig = options.toolConfig,
        .modelArmorConfig = options.modelArmorConfig,
        .serviceTier = options.serviceTier,
    }, GenerateContentResponse);
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

pub const StreamError = error{
    ApiError,
    MissingApiKey,
    InvalidSseData,
} || std.http.Client.RequestError || std.http.Client.Request.ReceiveHeadError || std.Io.Writer.Error || std.Io.Reader.DelimiterError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

/// Generate content with streaming via Server-Sent Events.
/// The `callback` is invoked for each chunk as it arrives.
/// Pass a `context` value to carry state into the callback.
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
    defer self.allocator.free(url);

    const req_body = GenerateContentRequest{
        .contents = contents,
        .generationConfig = config,
        .systemInstruction = options.systemInstruction,
        .safetySettings = options.safetySettings,
        .tools = options.tools,
        .toolConfig = options.toolConfig,
        .modelArmorConfig = options.modelArmorConfig,
        .serviceTier = options.serviceTier,
    };
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(req_body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;

    const auth = [1]std.http.Header{try self.authHeader()};
    return http.streamSse(self.allocator, &self.http_client, url, &auth, payload_buf.written(), GenerateContentResponse, self, context, callback);
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
    defer self.allocator.free(url);
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
    var base_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer base_buf.deinit();
    self.writeHost(&base_buf.writer) catch return error.OutOfMemory;
    base_buf.writer.print("/{s}/{s}", .{
        self.api_version,
        if (self.vertex != null) "publishers/google/models" else "models",
    }) catch return error.OutOfMemory;
    const base = base_buf.written();
    const url = try http.appendListParams(self.allocator, base, options);
    defer self.allocator.free(url);
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
    defer self.allocator.free(url);
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
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/models/{s}:embedContent", .{ self.devBaseUrl(), self.api_version, model });
    defer self.allocator.free(url);
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

// --- Files ---

/// Configuration for file uploads.
pub const UploadFileConfig = struct {
    /// Optional resource name (e.g. "files/my-file").
    name: ?[]const u8 = null,
    /// Human-readable display name (max 512 characters).
    displayName: ?[]const u8 = null,
    /// MIME type of the file. Auto-detected if not provided.
    mimeType: ?[]const u8 = null,
};

/// Upload a file to the Gemini API using the resumable upload protocol.
/// Returns the uploaded file metadata. The caller should call `deinit()` on the response.
pub fn uploadFile(
    self: *Client,
    data: []const u8,
    config: UploadFileConfig,
) !Response(types.UploadFileResponse) {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;

    // Stage 1: Initialize resumable upload
    const create_url = try std.fmt.allocPrint(self.allocator, "{s}/upload/{s}/files", .{ self.devBaseUrl(), self.api_version });
    defer self.allocator.free(create_url);

    var metadata_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer metadata_buf.deinit();
    std.json.Stringify.value(types.UploadFileRequest{
        .file = .{
            .name = config.name,
            .displayName = config.displayName,
            .mimeType = config.mimeType,
        },
    }, .{ .emit_null_optional_fields = false }, &metadata_buf.writer) catch
        return error.OutOfMemory;
    const metadata = metadata_buf.written();

    const mime_type = config.mimeType orelse "application/octet-stream";

    const uri = try std.Uri.parse(create_url);
    var req = try self.http_client.request(.POST, uri, .{
        .extra_headers = &.{
            .{ .name = "x-goog-api-key", .value = self.api_key },
            .{ .name = "X-Goog-Upload-Protocol", .value = "resumable" },
            .{ .name = "X-Goog-Upload-Command", .value = "start" },
            .{ .name = "X-Goog-Upload-Header-Content-Type", .value = mime_type },
        },
        .headers = .{
            .content_type = .{ .override = "application/json" },
        },
        .redirect_behavior = .init(5),
    });
    defer req.deinit();

    req.transfer_encoding = .{ .content_length = metadata.len };
    var bw = try req.sendBodyUnflushed(&.{});
    try bw.writer.writeAll(metadata);
    try bw.end();
    try req.connection.?.flush();

    var redirect_buf: [0]u8 = undefined;
    var init_response = try req.receiveHead(&redirect_buf);

    const init_status: u10 = @intFromEnum(init_response.head.status);
    if (init_status < 200 or init_status >= 300) {
        self.setErrorDetail(init_status, "");
        return error.ApiError;
    }

    // Extract upload URL before reader() invalidates header strings
    const upload_url = blk: {
        var it = init_response.head.iterateHeaders();
        while (it.next()) |header| {
            if (std.ascii.eqlIgnoreCase(header.name, "x-goog-upload-url")) {
                break :blk try self.allocator.dupe(u8, header.value);
            }
        }
        return error.ApiError;
    };
    defer self.allocator.free(upload_url);

    var transfer_buf: [256]u8 = undefined;
    const init_reader = init_response.reader(&transfer_buf);
    _ = init_reader.discardRemaining() catch {};

    // Stage 2: Upload the data
    const upload_uri = try std.Uri.parse(upload_url);
    var upload_req = try self.http_client.request(.POST, upload_uri, .{
        .extra_headers = &.{
            .{ .name = "x-goog-api-key", .value = self.api_key },
            .{ .name = "X-Goog-Upload-Command", .value = "upload, finalize" },
            .{ .name = "X-Goog-Upload-Offset", .value = "0" },
        },
        .headers = .{
            .content_type = .{ .override = mime_type },
        },
        .redirect_behavior = .init(5),
    });
    defer upload_req.deinit();

    upload_req.transfer_encoding = .{ .content_length = data.len };
    var upload_bw = try upload_req.sendBodyUnflushed(&.{});
    try upload_bw.writer.writeAll(data);
    try upload_bw.end();
    try upload_req.connection.?.flush();

    var upload_redirect_buf: [0]u8 = undefined;
    var upload_response = try upload_req.receiveHead(&upload_redirect_buf);

    const upload_status: u10 = @intFromEnum(upload_response.head.status);
    if (upload_status < 200 or upload_status >= 300) {
        self.setErrorDetail(upload_status, "");
        return error.ApiError;
    }

    var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
    errdefer response_buf.deinit();

    var upload_transfer_buf: [4096]u8 = undefined;
    const body_reader = upload_response.reader(&upload_transfer_buf);
    _ = body_reader.streamRemaining(&response_buf.writer) catch return error.ApiError;

    const body = response_buf.written();
    if (body.len == 0) return error.EmptyResponse;

    const parsed = try std.json.parseFromSlice(types.UploadFileResponse, self.allocator, body, .{ .ignore_unknown_fields = true });
    return .{ .value = parsed.value, .json_buf = response_buf, .parsed = parsed };
}

/// Get metadata for an uploaded file by its resource name (e.g. "files/abc123").
pub fn getFile(self: *Client, name: []const u8) !Response(types.File) {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}", .{ self.devBaseUrl(), self.api_version, name });
    defer self.allocator.free(url);
    return self.fetchGet(url, types.File);
}

/// List uploaded files. Use `ListOptions` to paginate through results.
pub fn listFiles(self: *Client, options: ListOptions) !Response(types.ListFilesResponse) {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;
    const base = try std.fmt.allocPrint(self.allocator, "{s}/{s}/files", .{ self.devBaseUrl(), self.api_version });
    defer self.allocator.free(base);
    const url = try http.appendListParams(self.allocator, base, options);
    defer self.allocator.free(url);
    return self.fetchGet(url, types.ListFilesResponse);
}

/// Download file contents by URI. Returns owned bytes — caller must free with `allocator.free()`.
pub fn downloadFile(self: *Client, uri: []const u8) ![]u8 {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;

    var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
    errdefer response_buf.deinit();

    const result = try self.http_client.fetch(.{
        .location = .{ .url = uri },
        .extra_headers = &.{
            .{ .name = "x-goog-api-key", .value = self.api_key },
        },
        .response_writer = &response_buf.writer,
    });

    const status_code: u10 = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        self.setErrorDetail(status_code, response_buf.written());
        response_buf.deinit();
        return error.ApiError;
    }

    return response_buf.toOwnedSlice() catch return error.OutOfMemory;
}

/// Delete an uploaded file by its resource name.
pub fn deleteFile(self: *Client, name: []const u8) !void {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}", .{ self.devBaseUrl(), self.api_version, name });
    defer self.allocator.free(url);

    const result = try self.http_client.fetch(.{
        .location = .{ .url = url },
        .method = .DELETE,
        .extra_headers = &.{
            .{ .name = "x-goog-api-key", .value = self.api_key },
        },
    });

    const status_code: u10 = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        self.setErrorDetail(status_code, "");
        return error.ApiError;
    }
}

// --- Cached Content ---

/// Configuration for creating cached content.
pub const CreateCachedContentConfig = struct {
    /// The content to cache.
    contents: ?[]const Content = null,
    /// System instruction to cache alongside the content.
    systemInstruction: ?Content = null,
    /// Tools to cache alongside the content.
    tools: ?[]const Tool = null,
    /// Tool configuration to cache.
    toolConfig: ?ToolConfig = null,
    /// Human-readable display name for the cached content.
    displayName: ?[]const u8 = null,
    /// TTL duration string (e.g. "3600s" for 1 hour). Mutually exclusive with `expireTime`.
    ttl: ?[]const u8 = null,
    /// Expiration timestamp in RFC 3339 format. Mutually exclusive with `ttl`.
    expireTime: ?[]const u8 = null,
};

/// Create cached content for a model. Cached content reduces token usage and
/// latency for repeated context.
pub fn createCachedContent(
    self: *Client,
    model: []const u8,
    config: CreateCachedContentConfig,
) !Response(types.CachedContent) {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/cachedContents", .{ self.devBaseUrl(), self.api_version });
    defer self.allocator.free(url);

    const full_model = try std.fmt.allocPrint(self.allocator, "models/{s}", .{model});
    defer self.allocator.free(full_model);

    return self.fetchPost(url, types.CreateCachedContentRequest{
        .model = full_model,
        .contents = config.contents,
        .systemInstruction = config.systemInstruction,
        .tools = config.tools,
        .toolConfig = config.toolConfig,
        .displayName = config.displayName,
        .ttl = config.ttl,
        .expireTime = config.expireTime,
    }, types.CachedContent);
}

/// Get metadata for cached content by its resource name.
pub fn getCachedContent(self: *Client, name: []const u8) !Response(types.CachedContent) {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}", .{ self.devBaseUrl(), self.api_version, name });
    defer self.allocator.free(url);
    return self.fetchGet(url, types.CachedContent);
}

/// List cached contents. Use `ListOptions` to paginate through results.
pub fn listCachedContents(self: *Client, options: ListOptions) !Response(types.ListCachedContentsResponse) {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;
    const base = try std.fmt.allocPrint(self.allocator, "{s}/{s}/cachedContents", .{ self.devBaseUrl(), self.api_version });
    defer self.allocator.free(base);
    const url = try http.appendListParams(self.allocator, base, options);
    defer self.allocator.free(url);
    return self.fetchGet(url, types.ListCachedContentsResponse);
}

/// Update the expiration of cached content (TTL or explicit timestamp).
pub fn updateCachedContent(
    self: *Client,
    name: []const u8,
    config: struct { ttl: ?[]const u8 = null, expireTime: ?[]const u8 = null },
) !Response(types.CachedContent) {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;

    // Build updateMask from non-null fields
    const update_mask: []const u8 = if (config.ttl != null and config.expireTime != null)
        "ttl,expireTime"
    else if (config.ttl != null)
        "ttl"
    else if (config.expireTime != null)
        "expireTime"
    else
        "";

    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}?updateMask={s}", .{ self.devBaseUrl(), self.api_version, name, update_mask });
    defer self.allocator.free(url);

    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(types.UpdateCachedContentRequest{
        .ttl = config.ttl,
        .expireTime = config.expireTime,
    }, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;
    const payload = payload_buf.written();

    var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
    errdefer response_buf.deinit();

    const result = try self.http_client.fetch(.{
        .location = .{ .url = url },
        .method = .PATCH,
        .payload = payload,
        .extra_headers = &.{
            .{ .name = "x-goog-api-key", .value = self.api_key },
        },
        .headers = .{
            .content_type = .{ .override = "application/json" },
        },
        .response_writer = &response_buf.writer,
    });

    const body = response_buf.written();
    const status_code: u10 = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        self.setErrorDetail(status_code, body);
        return error.ApiError;
    }
    if (body.len == 0) return error.EmptyResponse;

    const parsed = try std.json.parseFromSlice(types.CachedContent, self.allocator, body, .{ .ignore_unknown_fields = true });
    return .{ .value = parsed.value, .json_buf = response_buf, .parsed = parsed };
}

/// Delete cached content by its resource name.
pub fn deleteCachedContent(self: *Client, name: []const u8) !void {
    try self.requireDeveloperApi();
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}", .{ self.devBaseUrl(), self.api_version, name });
    defer self.allocator.free(url);

    const result = try self.http_client.fetch(.{
        .location = .{ .url = url },
        .method = .DELETE,
        .extra_headers = &.{
            .{ .name = "x-goog-api-key", .value = self.api_key },
        },
    });

    const status_code: u10 = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        self.setErrorDetail(status_code, "");
        return error.ApiError;
    }
}

test "Client init and deinit" {
    var client = Client.init(std.testing.allocator, std.testing.io, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expect(client.base_url == null);
    try std.testing.expectEqualStrings("v1beta", client.api_version);
}

test "init derives the api version per backend, override wins" {
    var vertex = Client.init(std.testing.allocator, std.testing.io, "tok", .{ .vertex = .{} });
    defer vertex.deinit();
    try std.testing.expectEqualStrings("v1beta1", vertex.api_version);

    var pinned = Client.init(std.testing.allocator, std.testing.io, "tok", .{ .api_version = "v1", .vertex = .{} });
    defer pinned.deinit();
    try std.testing.expectEqualStrings("v1", pinned.api_version);
}

test "modelUrl: developer API" {
    var client = Client.init(std.testing.allocator, std.testing.io, "key", .{});
    defer client.deinit();
    const url = try client.modelUrl("gemini-2.5-flash", .{ .action = "generateContent" });
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        url,
    );
}

test "modelUrl: vertex express mode" {
    var client = Client.init(std.testing.allocator, std.testing.io, "key", .{ .vertex = .{} });
    defer client.deinit();
    const url = try client.modelUrl("gemini-2.5-flash", .{ .action = "generateContent" });
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings(
        "https://aiplatform.googleapis.com/v1beta1/publishers/google/models/gemini-2.5-flash:generateContent",
        url,
    );
}

test "modelUrl: vertex project mode, global and regional" {
    var global = Client.init(std.testing.allocator, std.testing.io, "tok", .{ .vertex = .{ .project = "my-proj" } });
    defer global.deinit();
    const global_url = try global.modelUrl("gemini-2.5-flash", .{ .action = "countTokens" });
    defer std.testing.allocator.free(global_url);
    try std.testing.expectEqualStrings(
        "https://aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/global/publishers/google/models/gemini-2.5-flash:countTokens",
        global_url,
    );

    var regional = Client.init(std.testing.allocator, std.testing.io, "tok", .{
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
    var client = Client.init(std.testing.allocator, std.testing.io, "tok", .{ .vertex = .{ .project = "p" } });
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
    var client = Client.init(std.testing.allocator, std.testing.io, "tok", .{
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
    var dev = Client.init(std.testing.allocator, std.testing.io, "key", .{});
    defer dev.deinit();
    const dev_header = try dev.authHeader();
    try std.testing.expectEqualStrings("x-goog-api-key", dev_header.name);
    try std.testing.expectEqualStrings("key", dev_header.value);

    var express = Client.init(std.testing.allocator, std.testing.io, "key", .{ .vertex = .{} });
    defer express.deinit();
    try std.testing.expectEqualStrings("x-goog-api-key", (try express.authHeader()).name);

    var project = Client.init(std.testing.allocator, std.testing.io, "tok", .{ .vertex = .{ .project = "p" } });
    defer project.deinit();
    const first = try project.authHeader();
    try std.testing.expectEqualStrings("authorization", first.name);
    try std.testing.expectEqualStrings("Bearer tok", first.value);
    // Built once and cached; deinit frees it (leak-checked by the test allocator).
    const second = try project.authHeader();
    try std.testing.expectEqual(first.value.ptr, second.value.ptr);
}

test "vertex: developer-only methods are unsupported" {
    var client = Client.init(std.testing.allocator, std.testing.io, "tok", .{ .vertex = .{ .project = "p" } });
    defer client.deinit();
    try std.testing.expectError(error.UnsupportedByBackend, client.embedText("m", "hi"));
    try std.testing.expectError(error.UnsupportedByBackend, client.getFile("files/abc"));
    try std.testing.expectError(error.UnsupportedByBackend, client.createCachedContent("m", .{}));
    try std.testing.expectError(error.UnsupportedByBackend, client.deleteFile("files/abc"));
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
