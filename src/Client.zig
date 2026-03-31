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

test "Client init and deinit" {
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expectEqualStrings("https://generativelanguage.googleapis.com", client.base_url);
}
