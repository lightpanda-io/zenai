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

pub fn generateContent(
    self: *Client,
    model: []const u8,
    contents: []const Content,
    config: ?GenerationConfig,
    options: RequestOptions,
) GenerateContentError!GenerateContentResponse {
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

    // Prepare response writer
    var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
    defer response_buf.deinit();

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

    // Parse response
    const parsed = try std.json.parseFromSlice(
        GenerateContentResponse,
        self.allocator,
        response_body,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    return dupeResponse(self.allocator, parsed.value);
}

pub fn generateContentFromText(
    self: *Client,
    model: []const u8,
    prompt: []const u8,
    config: ?GenerationConfig,
    options: RequestOptions,
) GenerateContentError!GenerateContentResponse {
    const parts = [_]Part{.{ .text = prompt }};
    const contents = [_]Content{.{ .role = "user", .parts = &parts }};
    return self.generateContent(model, &contents, config, options);
}

pub fn freeResponse(self: *Client, response: GenerateContentResponse) void {
    freeResponseAlloc(self.allocator, response);
}

fn freeResponseAlloc(allocator: std.mem.Allocator, response: GenerateContentResponse) void {
    if (response.modelVersion) |v| allocator.free(v);
    if (response.responseId) |v| allocator.free(v);

    if (response.candidates) |candidates| {
        for (candidates) |candidate| {
            if (candidate.content) |content| {
                freeParts(allocator, content.parts);
                allocator.free(content.parts);
                if (content.role) |r| allocator.free(r);
            }
            if (candidate.safetyRatings) |ratings| {
                allocator.free(ratings);
            }
            if (candidate.citationMetadata) |cm| {
                if (cm.citationSources) |sources| {
                    for (sources) |src| {
                        if (src.uri) |v| allocator.free(v);
                        if (src.license) |v| allocator.free(v);
                    }
                    allocator.free(sources);
                }
            }
        }
        allocator.free(candidates);
    }
}

fn freeParts(allocator: std.mem.Allocator, parts: []const Part) void {
    for (parts) |part| {
        if (part.text) |t| allocator.free(t);
        if (part.inlineData) |b| {
            if (b.data) |d| allocator.free(d);
            if (b.mimeType) |m| allocator.free(m);
            if (b.displayName) |n| allocator.free(n);
        }
        if (part.fileData) |f| {
            if (f.fileUri) |u| allocator.free(u);
            if (f.mimeType) |m| allocator.free(m);
        }
        if (part.functionCall) |fc| {
            if (fc.id) |v| allocator.free(v);
            if (fc.name) |v| allocator.free(v);
            // args is std.json.Value — owned by the parsed JSON, but since we dupe
            // strings manually, we don't free the Value tree here.
        }
        if (part.functionResponse) |fr| {
            if (fr.id) |v| allocator.free(v);
            if (fr.name) |v| allocator.free(v);
        }
        if (part.executableCode) |ec| {
            if (ec.code) |v| allocator.free(v);
        }
        if (part.codeExecutionResult) |cr| {
            if (cr.output) |v| allocator.free(v);
        }
    }
}

fn dupeResponse(allocator: std.mem.Allocator, resp: GenerateContentResponse) std.mem.Allocator.Error!GenerateContentResponse {
    var result: GenerateContentResponse = .{};

    result.modelVersion = if (resp.modelVersion) |v| try allocator.dupe(u8, v) else null;
    result.responseId = if (resp.responseId) |v| try allocator.dupe(u8, v) else null;
    result.usageMetadata = resp.usageMetadata;

    if (resp.candidates) |candidates| {
        const duped = try allocator.alloc(types.Candidate, candidates.len);
        errdefer allocator.free(duped);

        for (candidates, 0..) |candidate, i| {
            duped[i] = .{
                .finishReason = candidate.finishReason,
                .tokenCount = candidate.tokenCount,
                .avgLogprobs = candidate.avgLogprobs,
                .index = candidate.index,
            };

            if (candidate.content) |content| {
                duped[i].content = .{
                    .role = if (content.role) |r| try allocator.dupe(u8, r) else null,
                    .parts = try dupeParts(allocator, content.parts),
                };
            }

            if (candidate.safetyRatings) |ratings| {
                duped[i].safetyRatings = try allocator.dupe(types.SafetyRating, ratings);
            }

            if (candidate.citationMetadata) |cm| {
                var dcm: types.CitationMetadata = .{};
                if (cm.citationSources) |sources| {
                    const ds = try allocator.alloc(types.CitationSource, sources.len);
                    for (sources, 0..) |src, j| {
                        ds[j] = .{
                            .startIndex = src.startIndex,
                            .endIndex = src.endIndex,
                            .uri = if (src.uri) |v| try allocator.dupe(u8, v) else null,
                            .license = if (src.license) |v| try allocator.dupe(u8, v) else null,
                        };
                    }
                    dcm.citationSources = ds;
                }
                duped[i].citationMetadata = dcm;
            }
        }
        result.candidates = duped;
    }

    return result;
}

fn dupeParts(allocator: std.mem.Allocator, parts: []const Part) std.mem.Allocator.Error![]Part {
    const duped = try allocator.alloc(Part, parts.len);
    for (parts, 0..) |part, i| {
        duped[i] = .{};
        duped[i].text = if (part.text) |t| try allocator.dupe(u8, t) else null;
        duped[i].thought = part.thought;

        if (part.inlineData) |b| {
            duped[i].inlineData = .{
                .data = if (b.data) |d| try allocator.dupe(u8, d) else null,
                .mimeType = if (b.mimeType) |m| try allocator.dupe(u8, m) else null,
                .displayName = if (b.displayName) |n| try allocator.dupe(u8, n) else null,
            };
        }
        if (part.fileData) |f| {
            duped[i].fileData = .{
                .fileUri = if (f.fileUri) |u| try allocator.dupe(u8, u) else null,
                .mimeType = if (f.mimeType) |m| try allocator.dupe(u8, m) else null,
            };
        }
        if (part.functionCall) |fc| {
            duped[i].functionCall = .{
                .id = if (fc.id) |v| try allocator.dupe(u8, v) else null,
                .name = if (fc.name) |v| try allocator.dupe(u8, v) else null,
                .args = fc.args, // json Value — not deep-duped
            };
        }
        if (part.functionResponse) |fr| {
            duped[i].functionResponse = .{
                .id = if (fr.id) |v| try allocator.dupe(u8, v) else null,
                .name = if (fr.name) |v| try allocator.dupe(u8, v) else null,
                .response = fr.response,
            };
        }
        if (part.executableCode) |ec| {
            duped[i].executableCode = .{
                .code = if (ec.code) |v| try allocator.dupe(u8, v) else null,
                .language = ec.language,
            };
        }
        if (part.codeExecutionResult) |cr| {
            duped[i].codeExecutionResult = .{
                .outcome = cr.outcome,
                .output = if (cr.output) |v| try allocator.dupe(u8, v) else null,
            };
        }
    }
    return duped;
}

test "Client init and deinit" {
    var client = Client.init(std.testing.allocator, "test-key", .{});
    defer client.deinit();
    try std.testing.expectEqualStrings("test-key", client.api_key);
    try std.testing.expectEqualStrings("https://generativelanguage.googleapis.com", client.base_url);
}
