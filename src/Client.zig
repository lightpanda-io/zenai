const std = @import("std");
const types = @import("types.zig");

const Content = types.Content;
const Part = types.Part;
const GenerationConfig = types.GenerationConfig;
const GenerateContentRequest = types.GenerateContentRequest;
const GenerateContentResponse = types.GenerateContentResponse;
const SafetySetting = types.SafetySetting;
const Tool = types.Tool;
const ToolConfig = types.ToolConfig;

const Client = @This();

allocator: std.mem.Allocator,
api_key: []const u8,
base_url: []const u8,
api_version: []const u8,
http_client: std.http.Client,

pub const InitOptions = struct {
    base_url: []const u8 = "https://generativelanguage.googleapis.com",
    api_version: []const u8 = "v1beta",
};

pub fn init(allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .api_version = options.api_version,
        .http_client = .{ .allocator = allocator },
    };
}

pub fn deinit(self: *Client) void {
    self.http_client.deinit();
}

pub const GenerateContentError = error{
    ApiError,
    MissingApiKey,
    EmptyResponse,
} || std.http.Client.FetchError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

pub const RequestOptions = struct {
    systemInstruction: ?Content = null,
    safetySettings: ?[]const SafetySetting = null,
    tools: ?[]const Tool = null,
    toolConfig: ?ToolConfig = null,
};

/// Owns the parsed response and its backing memory.
/// Call `deinit()` when done to free all resources.
pub const Response = struct {
    value: GenerateContentResponse,
    /// Owns the JSON bytes backing the parsed response.
    json_buf: std.Io.Writer.Allocating,
    /// Owns the parsed JSON arena (strings, slices, etc. point into json_buf).
    parsed: std.json.Parsed(GenerateContentResponse),

    pub fn deinit(self: *Response) void {
        self.parsed.deinit();
        self.json_buf.deinit();
    }

    pub fn text(self: Response) ?[]const u8 {
        return self.value.text();
    }

    pub fn firstFunctionCall(self: Response) ?types.FunctionCall {
        return self.value.firstFunctionCall();
    }
};

pub fn generateContent(
    self: *Client,
    model: []const u8,
    contents: []const Content,
    config: ?GenerationConfig,
    options: RequestOptions,
) GenerateContentError!Response {
    if (self.api_key.len == 0) return error.MissingApiKey;

    // Build URL
    const url = try std.fmt.allocPrint(
        self.allocator,
        "{s}/{s}/models/{s}:generateContent",
        .{ self.base_url, self.api_version, model },
    );
    defer self.allocator.free(url);

    // Build request body
    const req_body = GenerateContentRequest{
        .contents = contents,
        .generationConfig = config,
        .systemInstruction = options.systemInstruction,
        .safetySettings = options.safetySettings,
        .tools = options.tools,
        .toolConfig = options.toolConfig,
    };
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(req_body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;
    const payload = payload_buf.written();

    // Prepare response writer — ownership transfers to Response
    var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
    errdefer response_buf.deinit();

    // Make HTTP request
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

    const response_body = response_buf.written();

    // Check HTTP status
    const status_code = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        if (response_body.len > 0) {
            std.log.err("Gemini API error (HTTP {d}): {s}", .{ status_code, response_body });
        }
        return error.ApiError;
    }

    if (response_body.len == 0) return error.EmptyResponse;

    // Parse response — ownership transfers to Response
    const parsed = try std.json.parseFromSlice(
        GenerateContentResponse,
        self.allocator,
        response_body,
        .{ .ignore_unknown_fields = true },
    );

    return .{
        .value = parsed.value,
        .json_buf = response_buf,
        .parsed = parsed,
    };
}

pub fn generateContentFromText(
    self: *Client,
    model: []const u8,
    prompt: []const u8,
    config: ?GenerationConfig,
    options: RequestOptions,
) GenerateContentError!Response {
    const parts = [_]Part{.{ .text = prompt }};
    const contents = [_]Content{.{ .role = "user", .parts = &parts }};
    return self.generateContent(model, &contents, config, options);
}

pub const StreamError = error{
    ApiError,
    MissingApiKey,
    InvalidSseData,
} || std.http.Client.RequestError || std.http.Client.Request.ReceiveHeadError || std.Io.Writer.Error || std.Io.Reader.DelimiterError || std.json.ParseError(std.json.Scanner) || std.mem.Allocator.Error || std.Uri.ParseError;

pub fn generateContentStream(
    self: *Client,
    model: []const u8,
    contents: []const Content,
    config: ?GenerationConfig,
    options: RequestOptions,
    callback: *const fn (GenerateContentResponse) void,
) StreamError!void {
    if (self.api_key.len == 0) return error.MissingApiKey;

    // Build URL with streaming endpoint
    const url = try std.fmt.allocPrint(
        self.allocator,
        "{s}/{s}/models/{s}:streamGenerateContent?alt=sse",
        .{ self.base_url, self.api_version, model },
    );
    defer self.allocator.free(url);

    // Build request body
    const req_body = GenerateContentRequest{
        .contents = contents,
        .generationConfig = config,
        .systemInstruction = options.systemInstruction,
        .safetySettings = options.safetySettings,
        .tools = options.tools,
        .toolConfig = options.toolConfig,
    };
    var payload_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer payload_buf.deinit();
    std.json.Stringify.value(req_body, .{ .emit_null_optional_fields = false }, &payload_buf.writer) catch
        return error.OutOfMemory;
    const payload = payload_buf.written();

    // Create low-level request
    const uri = try std.Uri.parse(url);
    var req = try self.http_client.request(.POST, uri, .{
        .extra_headers = &.{
            .{ .name = "x-goog-api-key", .value = self.api_key },
        },
        .headers = .{
            .content_type = .{ .override = "application/json" },
        },
        .redirect_behavior = .unhandled,
    });
    defer req.deinit();

    // Send body
    req.transfer_encoding = .{ .content_length = payload.len };
    var bw = try req.sendBodyUnflushed(&.{});
    try bw.writer.writeAll(payload);
    try bw.end();
    try req.connection.?.flush();

    // Receive response headers
    var redirect_buf: [0]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);

    // Check HTTP status
    const status_code = @intFromEnum(response.head.status);
    if (status_code < 200 or status_code >= 300) {
        std.log.err("Gemini API streaming error (HTTP {d})", .{status_code});
        return error.ApiError;
    }

    // Read SSE lines — buffer must be large enough to hold a full SSE data line
    const transfer_buf = try self.allocator.alloc(u8, 256 * 1024);
    defer self.allocator.free(transfer_buf);
    const reader = response.reader(transfer_buf);

    while (true) {
        const line = reader.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => return error.InvalidSseData,
            error.ReadFailed => return, // connection closed
        } orelse return; // end of stream

        // Skip empty lines and carriage returns
        const trimmed = std.mem.trimRight(u8, line, "\r");
        if (trimmed.len == 0) continue;

        // Parse SSE data lines
        if (std.mem.startsWith(u8, trimmed, "data: ")) {
            const json_data = trimmed["data: ".len..];

            const parsed = std.json.parseFromSlice(
                GenerateContentResponse,
                self.allocator,
                json_data,
                .{ .ignore_unknown_fields = true },
            ) catch continue; // skip malformed chunks
            defer parsed.deinit();

            callback(parsed.value);
        }
    }
}

pub fn generateContentStreamFromText(
    self: *Client,
    model: []const u8,
    prompt: []const u8,
    config: ?GenerationConfig,
    options: RequestOptions,
    callback: *const fn (GenerateContentResponse) void,
) StreamError!void {
    const parts = [_]Part{.{ .text = prompt }};
    const contents = [_]Content{.{ .role = "user", .parts = &parts }};
    return self.generateContentStream(model, &contents, config, options, callback);
}

// --- Model Management ---

pub const GetError = error{
    ApiError,
    MissingApiKey,
    EmptyResponse,
} || std.http.Client.FetchError || std.mem.Allocator.Error || std.Uri.ParseError;

pub fn ParsedResponse(comptime T: type) type {
    return struct {
        value: T,
        json_buf: std.Io.Writer.Allocating,
        parsed: std.json.Parsed(T),

        pub fn deinit(self: *@This()) void {
            self.parsed.deinit();
            self.json_buf.deinit();
        }
    };
}

fn fetchGet(self: *Client, url: []const u8, comptime T: type) (GetError || std.json.ParseError(std.json.Scanner))!ParsedResponse(T) {
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
    const status_code = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        if (body.len > 0) {
            std.log.err("Gemini API error (HTTP {d}): {s}", .{ status_code, body });
        }
        return error.ApiError;
    }

    if (body.len == 0) return error.EmptyResponse;

    const parsed = try std.json.parseFromSlice(T, self.allocator, body, .{ .ignore_unknown_fields = true });

    return .{
        .value = parsed.value,
        .json_buf = response_buf,
        .parsed = parsed,
    };
}

fn fetchPost(self: *Client, url: []const u8, body: anytype, comptime T: type) (GetError || std.json.ParseError(std.json.Scanner))!ParsedResponse(T) {
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
    const status_code = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        if (resp_body.len > 0) {
            std.log.err("Gemini API error (HTTP {d}): {s}", .{ status_code, resp_body });
        }
        return error.ApiError;
    }

    if (resp_body.len == 0) return error.EmptyResponse;

    const parsed = try std.json.parseFromSlice(T, self.allocator, resp_body, .{ .ignore_unknown_fields = true });

    return .{
        .value = parsed.value,
        .json_buf = response_buf,
        .parsed = parsed,
    };
}

pub const ListOptions = struct {
    pageSize: ?i32 = null,
    pageToken: ?[]const u8 = null,
};

fn appendListParams(allocator: std.mem.Allocator, base_url: []const u8, options: ListOptions) ![]u8 {
    if (options.pageSize == null and options.pageToken == null) {
        return allocator.dupe(u8, base_url);
    }
    if (options.pageSize != null and options.pageToken != null) {
        return std.fmt.allocPrint(allocator, "{s}?pageSize={d}&pageToken={s}", .{ base_url, options.pageSize.?, options.pageToken.? });
    }
    if (options.pageSize) |ps| {
        return std.fmt.allocPrint(allocator, "{s}?pageSize={d}", .{ base_url, ps });
    }
    return std.fmt.allocPrint(allocator, "{s}?pageToken={s}", .{ base_url, options.pageToken.? });
}

pub fn getModel(self: *Client, model: []const u8) !ParsedResponse(types.Model) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/models/{s}", .{ self.base_url, self.api_version, model });
    defer self.allocator.free(url);
    return self.fetchGet(url, types.Model);
}

pub fn listModels(self: *Client, options: ListOptions) !ParsedResponse(types.ListModelsResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const base = try std.fmt.allocPrint(self.allocator, "{s}/{s}/models", .{ self.base_url, self.api_version });
    defer self.allocator.free(base);
    const url = try appendListParams(self.allocator, base, options);
    defer self.allocator.free(url);
    return self.fetchGet(url, types.ListModelsResponse);
}

// --- Token Counting ---

pub fn countTokens(
    self: *Client,
    model: []const u8,
    contents: []const Content,
    options: RequestOptions,
) !ParsedResponse(types.CountTokensResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/models/{s}:countTokens", .{ self.base_url, self.api_version, model });
    defer self.allocator.free(url);
    return self.fetchPost(url, types.CountTokensRequest{
        .contents = contents,
        .systemInstruction = options.systemInstruction,
        .tools = options.tools,
    }, types.CountTokensResponse);
}

pub fn countTokensFromText(
    self: *Client,
    model: []const u8,
    prompt: []const u8,
) !ParsedResponse(types.CountTokensResponse) {
    const parts = [_]Part{.{ .text = prompt }};
    const contents = [_]Content{.{ .role = "user", .parts = &parts }};
    return self.countTokens(model, &contents, .{});
}

// --- Embeddings ---

pub const EmbedConfig = struct {
    taskType: ?[]const u8 = null,
    title: ?[]const u8 = null,
    outputDimensionality: ?i32 = null,
};

pub fn embedContent(
    self: *Client,
    model: []const u8,
    content: Content,
    config: EmbedConfig,
) !ParsedResponse(types.EmbedContentResponse) {
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

pub fn embedText(
    self: *Client,
    model: []const u8,
    text: []const u8,
) !ParsedResponse(types.EmbedContentResponse) {
    const parts = [_]Part{.{ .text = text }};
    const content = Content{ .parts = &parts };
    return self.embedContent(model, content, .{});
}

// --- Files ---

pub const UploadFileConfig = struct {
    name: ?[]const u8 = null,
    displayName: ?[]const u8 = null,
    mimeType: ?[]const u8 = null,
};

pub fn uploadFile(
    self: *Client,
    data: []const u8,
    config: UploadFileConfig,
) !ParsedResponse(types.UploadFileResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;

    // Stage 1: Initialize resumable upload
    const create_url = try std.fmt.allocPrint(self.allocator, "{s}/upload/{s}/files", .{ self.base_url, self.api_version });
    defer self.allocator.free(create_url);

    // Build metadata JSON
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

    // Make the initialization request
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
        .redirect_behavior = .unhandled,
    });
    defer req.deinit();

    req.transfer_encoding = .{ .content_length = metadata.len };
    var bw = try req.sendBodyUnflushed(&.{});
    try bw.writer.writeAll(metadata);
    try bw.end();
    try req.connection.?.flush();

    var redirect_buf: [0]u8 = undefined;
    var init_response = try req.receiveHead(&redirect_buf);

    const init_status = @intFromEnum(init_response.head.status);
    if (init_status < 200 or init_status >= 300) {
        std.log.err("Gemini file upload init error (HTTP {d})", .{init_status});
        return error.ApiError;
    }

    // Find X-Goog-Upload-Url in response headers (before reader() invalidates strings)
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

    // Consume the init response body
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
        .redirect_behavior = .unhandled,
    });
    defer upload_req.deinit();

    upload_req.transfer_encoding = .{ .content_length = data.len };
    var upload_bw = try upload_req.sendBodyUnflushed(&.{});
    try upload_bw.writer.writeAll(data);
    try upload_bw.end();
    try upload_req.connection.?.flush();

    var upload_redirect_buf: [0]u8 = undefined;
    var upload_response = try upload_req.receiveHead(&upload_redirect_buf);

    const upload_status = @intFromEnum(upload_response.head.status);
    if (upload_status < 200 or upload_status >= 300) {
        std.log.err("Gemini file upload error (HTTP {d})", .{upload_status});
        return error.ApiError;
    }

    // Read and parse response body
    var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
    errdefer response_buf.deinit();

    var upload_transfer_buf: [4096]u8 = undefined;
    const body_reader = upload_response.reader(&upload_transfer_buf);
    _ = body_reader.streamRemaining(&response_buf.writer) catch return error.ApiError;

    const body = response_buf.written();
    if (body.len == 0) return error.EmptyResponse;

    const parsed = try std.json.parseFromSlice(types.UploadFileResponse, self.allocator, body, .{ .ignore_unknown_fields = true });

    return .{
        .value = parsed.value,
        .json_buf = response_buf,
        .parsed = parsed,
    };
}

pub fn getFile(self: *Client, name: []const u8) !ParsedResponse(types.File) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}", .{ self.base_url, self.api_version, name });
    defer self.allocator.free(url);
    return self.fetchGet(url, types.File);
}

pub fn listFiles(self: *Client, options: ListOptions) !ParsedResponse(types.ListFilesResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const base = try std.fmt.allocPrint(self.allocator, "{s}/{s}/files", .{ self.base_url, self.api_version });
    defer self.allocator.free(base);
    const url = try appendListParams(self.allocator, base, options);
    defer self.allocator.free(url);
    return self.fetchGet(url, types.ListFilesResponse);
}

pub fn downloadFile(self: *Client, uri: []const u8) ![]u8 {
    if (self.api_key.len == 0) return error.MissingApiKey;

    // Append API key as query parameter
    const sep: []const u8 = if (std.mem.indexOf(u8, uri, "?") != null) "&" else "?";
    const url = try std.fmt.allocPrint(self.allocator, "{s}{s}key={s}", .{ uri, sep, self.api_key });
    defer self.allocator.free(url);

    var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
    errdefer response_buf.deinit();

    const result = try self.http_client.fetch(.{
        .location = .{ .url = url },
        .response_writer = &response_buf.writer,
    });

    const status_code = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        response_buf.deinit();
        return error.ApiError;
    }

    return response_buf.toOwnedSlice() catch return error.OutOfMemory;
}

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

    const status_code = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        return error.ApiError;
    }
}

// --- Cached Content ---

pub const CreateCachedContentConfig = struct {
    contents: ?[]const Content = null,
    systemInstruction: ?Content = null,
    tools: ?[]const Tool = null,
    toolConfig: ?ToolConfig = null,
    displayName: ?[]const u8 = null,
    /// Duration string, e.g. "3600s" for 1 hour.
    ttl: ?[]const u8 = null,
    /// RFC 3339 timestamp, e.g. "2026-04-01T00:00:00Z".
    expireTime: ?[]const u8 = null,
};

pub fn createCachedContent(
    self: *Client,
    model: []const u8,
    config: CreateCachedContentConfig,
) !ParsedResponse(types.CachedContent) {
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

pub fn getCachedContent(self: *Client, name: []const u8) !ParsedResponse(types.CachedContent) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}", .{ self.base_url, self.api_version, name });
    defer self.allocator.free(url);
    return self.fetchGet(url, types.CachedContent);
}

pub fn listCachedContents(self: *Client, options: ListOptions) !ParsedResponse(types.ListCachedContentsResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const base = try std.fmt.allocPrint(self.allocator, "{s}/{s}/cachedContents", .{ self.base_url, self.api_version });
    defer self.allocator.free(base);
    const url = try appendListParams(self.allocator, base, options);
    defer self.allocator.free(url);
    return self.fetchGet(url, types.ListCachedContentsResponse);
}

pub fn updateCachedContent(
    self: *Client,
    name: []const u8,
    config: struct { ttl: ?[]const u8 = null, expireTime: ?[]const u8 = null },
) !ParsedResponse(types.CachedContent) {
    if (self.api_key.len == 0) return error.MissingApiKey;
    const url = try std.fmt.allocPrint(self.allocator, "{s}/{s}/{s}", .{ self.base_url, self.api_version, name });
    defer self.allocator.free(url);

    // PATCH request
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
    const status_code = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) {
        if (body.len > 0) std.log.err("Gemini API error (HTTP {d}): {s}", .{ status_code, body });
        return error.ApiError;
    }
    if (body.len == 0) return error.EmptyResponse;

    const parsed = try std.json.parseFromSlice(types.CachedContent, self.allocator, body, .{ .ignore_unknown_fields = true });
    return .{ .value = parsed.value, .json_buf = response_buf, .parsed = parsed };
}

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

    const status_code = @intFromEnum(result.status);
    if (status_code < 200 or status_code >= 300) return error.ApiError;
}

test "Client init and deinit" {
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expectEqualStrings("https://generativelanguage.googleapis.com", client.base_url);
}
