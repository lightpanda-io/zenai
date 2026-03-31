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

test "Client init and deinit" {
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expectEqualStrings("https://generativelanguage.googleapis.com", client.base_url);
}
