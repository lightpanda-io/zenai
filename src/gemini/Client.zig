const std = @import("std");
const types = @import("types.zig");
const http = @import("../http.zig");

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
base_url: []const u8,
api_version: []const u8,
http_client: std.http.Client,
/// The most recent API error detail, if any. Set on `error.ApiError`.
last_error: ?types.ApiErrorDetail = null,
last_error_status: ?u10 = null,

/// Options for customizing the API endpoint.
pub const InitOptions = struct {
    /// Base URL for the Gemini API.
    base_url: []const u8 = "https://generativelanguage.googleapis.com",
    /// API version prefix.
    api_version: []const u8 = "v1beta",
};

/// Create a new Gemini API client.
/// The `api_key` can be obtained from https://ai.google.dev/gemini-api/docs/api-key
pub fn init(allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .api_version = options.api_version,
        .http_client = .{ .allocator = allocator },
        .last_error = null,
        .last_error_status = null,
    };
}

/// Release all resources held by the client, including HTTP connections.
pub fn deinit(self: *Client) void {
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
} || std.http.Client.FetchError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

// --- Internal helpers ---

fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error_status = status_code;
    self.last_error = null;
    if (body.len > 0) {
        std.log.err("Gemini API error (HTTP {d}): {s}", .{ status_code, body });
        if (std.json.parseFromSlice(types.ApiErrorResponse, self.allocator, body, .{ .ignore_unknown_fields = true })) |parsed| {
            self.last_error = parsed.value.@"error";
            parsed.deinit();
        } else |_| {}
    }
}

fn fetchGet(self: *Client, url: []const u8, comptime T: type) ApiError!Response(T) {
    var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
    errdefer response_buf.deinit();

    const result = try self.http_client.fetch(.{
        .location = .{ .url = url },
        .extra_headers = &.{
            .{ .name = "x-goog-api-key", .value = self.api_key },
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

    const parsed = try std.json.parseFromSlice(T, self.allocator, body, .{ .ignore_unknown_fields = true });
    return .{ .value = parsed.value, .json_buf = response_buf, .parsed = parsed };
}

fn fetchPost(self: *Client, url: []const u8, body: anytype, comptime T: type) ApiError!Response(T) {
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;
    const payload = payload_buf.written();

    var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
    errdefer response_buf.deinit();

    const result = try self.http_client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = payload,
        .extra_headers = &.{
            .{ .name = "x-goog-api-key", .value = self.api_key },
        },
        .headers = .{
            .content_type = .{ .override = "application/json" },
        },
        .response_writer = &response_buf.writer,
    });

    const resp_body = response_buf.written();
    const status_code: u10 = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        self.setErrorDetail(status_code, resp_body);
        return error.ApiError;
    }
    if (resp_body.len == 0) return error.EmptyResponse;

    const parsed = try std.json.parseFromSlice(T, self.allocator, resp_body, .{ .ignore_unknown_fields = true });
    return .{ .value = parsed.value, .json_buf = response_buf, .parsed = parsed };
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
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/models/{s}:generateContent", .{ self.base_url, self.api_version, model });
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

    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/models/{s}:streamGenerateContent?alt=sse", .{ self.base_url, self.api_version, model });
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
    const payload = payload_buf.written();

    const uri = try std.Uri.parse(url);
    var req = try self.http_client.request(.POST, uri, .{
        .extra_headers = &.{
            .{ .name = "x-goog-api-key", .value = self.api_key },
        },
        .headers = .{
            .content_type = .{ .override = "application/json" },
        },
        .redirect_behavior = .init(5),
    });
    defer req.deinit();

    req.transfer_encoding = .{ .content_length = payload.len };
    var bw = try req.sendBodyUnflushed(&.{});
    try bw.writer.writeAll(payload);
    try bw.end();
    try req.connection.?.flush();

    var redirect_buf: [0]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);

    const status_code: u10 = @intFromEnum(response.head.status);
    if (status_code < 200 or status_code >= 300) {
        self.setErrorDetail(status_code, "");
        return error.ApiError;
    }

    const transfer_buf = try self.allocator.alloc(u8, 256 * 1024);
    defer self.allocator.free(transfer_buf);
    const reader = response.reader(transfer_buf);

    while (true) {
        const line = reader.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => return error.InvalidSseData,
            error.ReadFailed => return,
        } orelse return;

        const trimmed = std.mem.trimRight(u8, line, "\r");
        if (trimmed.len == 0) continue;

        if (std.mem.startsWith(u8, trimmed, "data: ")) {
            const json_data = trimmed["data: ".len..];
            const parsed = std.json.parseFromSlice(GenerateContentResponse, self.allocator, json_data, .{ .ignore_unknown_fields = true }) catch |err| {
                std.log.err("Gemini streaming: failed to parse SSE chunk: {}", .{err});
                return error.InvalidSseData;
            };
            defer parsed.deinit();
            callback(context, parsed.value);
        }
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

// --- Model Management ---

/// Get metadata about a specific model (token limits, supported methods, etc.).
pub fn getModel(self: *Client, model: []const u8) !Response(types.Model) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/models/{s}", .{ self.base_url, self.api_version, model });
    defer self.allocator.free(url);
    return self.fetchGet(url, types.Model);
}

/// List available models. Use `ListOptions` to paginate through results.
pub fn listModels(self: *Client, options: ListOptions) !Response(types.ListModelsResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const base = try std.fmt.allocPrint(self.allocator, "{s}/{s}/models", .{ self.base_url, self.api_version });
    defer self.allocator.free(base);
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
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/models/{s}:countTokens", .{ self.base_url, self.api_version, model });
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
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/models/{s}:embedContent", .{ self.base_url, self.api_version, model });
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
    if (self.api_key.len == 0) return error.MissingApiKey;

    // Stage 1: Initialize resumable upload
    const create_url = try std.fmt.allocPrint(self.allocator, "{s}/upload/{s}/files", .{ self.base_url, self.api_version });
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
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}", .{ self.base_url, self.api_version, name });
    defer self.allocator.free(url);
    return self.fetchGet(url, types.File);
}

/// List uploaded files. Use `ListOptions` to paginate through results.
pub fn listFiles(self: *Client, options: ListOptions) !Response(types.ListFilesResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const base = try std.fmt.allocPrint(self.allocator, "{s}/{s}/files", .{ self.base_url, self.api_version });
    defer self.allocator.free(base);
    const url = try http.appendListParams(self.allocator, base, options);
    defer self.allocator.free(url);
    return self.fetchGet(url, types.ListFilesResponse);
}

/// Download file contents by URI. Returns owned bytes — caller must free with `allocator.free()`.
pub fn downloadFile(self: *Client, uri: []const u8) ![]u8 {
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
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}", .{ self.base_url, self.api_version, name });
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
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/cachedContents", .{ self.base_url, self.api_version });
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
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}", .{ self.base_url, self.api_version, name });
    defer self.allocator.free(url);
    return self.fetchGet(url, types.CachedContent);
}

/// List cached contents. Use `ListOptions` to paginate through results.
pub fn listCachedContents(self: *Client, options: ListOptions) !Response(types.ListCachedContentsResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const base = try std.fmt.allocPrint(self.allocator, "{s}/{s}/cachedContents", .{ self.base_url, self.api_version });
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

    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}?updateMask={s}", .{ self.base_url, self.api_version, name, update_mask });
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
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}", .{ self.base_url, self.api_version, name });
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
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expectEqualStrings("https://generativelanguage.googleapis.com", client.base_url);
}
