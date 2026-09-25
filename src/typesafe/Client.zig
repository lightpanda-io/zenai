//! TypeSafe System One API client. https://docs.typesafe.ai
//!
//! Post a `state` plus a map of typed questions about it; get one typed answer
//! per question. The state is ingested once, so batching several questions
//! into one request is much cheaper than one call each.

const std = @import("std");
const types = @import("types.zig");
const http = @import("../http.zig");
const retry = @import("../retry.zig");

pub const RetryPolicy = retry.RetryPolicy;

const AskOptions = types.AskOptions;
const AskRequest = types.AskRequest;
const AskResponse = types.AskResponse;
const Content = types.Content;
const ListModelsResponse = types.ListModelsResponse;
const Questions = types.Questions;

const Client = @This();

allocator: std.mem.Allocator,
api_key: []const u8,
base_url: []const u8,
http_client: std.http.Client,
retry_policy: RetryPolicy,
request_timeout_ms: ?u32,
last_error: http.ErrorDetail = .{},
/// `Bearer <api_key>`, built on first use.
authorization: ?[]const u8 = null,
/// Set by the host so a SIGINT can abort an in-flight request mid-read.
interrupt: ?*http.Interrupt = null,

/// Override to reach the same protocol elsewhere (see `channels`).
pub const default_base_url = "https://api.typesafe.ai";

pub const InitOptions = struct {
    base_url: []const u8 = default_base_url,
    retry_policy: RetryPolicy = .{},
    /// Per-attempt wall-clock bound on non-streaming requests, from an
    /// established connection to the end of the body (connect excluded);
    /// exceeding it fails with `error.Timeout`. `null` waits indefinitely.
    request_timeout_ms: ?u32 = null,
};

/// `api_key` and `options.base_url` are borrowed, not copied: they must
/// outlive the client.
pub fn init(io: std.Io, allocator: std.mem.Allocator, api_key: []const u8, options: InitOptions) Client {
    return .{
        .allocator = allocator,
        .api_key = api_key,
        .base_url = options.base_url,
        .http_client = .{ .allocator = allocator, .io = io },
        .retry_policy = options.retry_policy,
        .request_timeout_ms = options.request_timeout_ms,
    };
}

pub fn deinit(self: *Client) void {
    self.http_client.deinit();
    self.last_error.deinit(self.allocator);
    if (self.authorization) |a| self.allocator.free(a);
}

pub const Response = http.Response;

pub const ApiError = error{MissingApiKey} || http.FetchError;

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error.setLogged(self.allocator, status_code, body, "TypeSafe");
}

fn authHeader(self: *Client) std.mem.Allocator.Error![1]std.http.Header {
    if (self.authorization == null)
        self.authorization = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_key});
    return .{.{ .name = "Authorization", .value = self.authorization.? }};
}

/// Ask Jev one or more typed questions about `state`. Each question's key is an
/// id you choose, and the matching answer comes back under that same id
/// (`response.value.answer(id)`).
///
/// Validate a `choice` with `AskResponse.choice` before acting on it.
///
/// 401 (bad key) and 422 (malformed question, or a state over the 32k budget)
/// surface as `error.ApiError` with the detail in `last_error`; 429 and 529
/// are retried by `retry_policy`.
///
/// Caller owns the returned `Response` and must call `deinit()` — every string
/// in the answers borrows it.
pub fn ask(
    self: *Client,
    state: Content,
    questions: Questions,
    options: AskOptions,
) ApiError!Response(AskResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;

    const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/systemone", .{self.base_url});
    defer self.allocator.free(url);

    const request: AskRequest = .{ .state = state, .model = options.model, .questions = questions };
    const auth = try self.authHeader();

    return http.postJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, self.request_timeout_ms, url, &auth, request, AskResponse, self);
}

/// List the model names and aliases this key may send in `AskOptions.model`.
/// Caller owns the returned `Response` and must call `deinit()`.
pub fn listModels(self: *Client) ApiError!Response(ListModelsResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;

    const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/models", .{self.base_url});
    defer self.allocator.free(url);

    const auth = try self.authHeader();
    return http.fetchJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, self.request_timeout_ms, .{
        .location = .{ .url = url },
        .method = .GET,
        .extra_headers = &auth,
    }, ListModelsResponse, self);
}

test "ask rejects empty api key" {
    var client = init(std.testing.io, std.testing.allocator, "", .{});
    defer client.deinit();
    try std.testing.expectError(error.MissingApiKey, client.ask(
        .{ .text = "anything" },
        .init(&.{.{ .key = "q", .value = .noulText("Is this a complaint?") }}),
        .{},
    ));
}

test "listModels rejects empty api key" {
    var client = init(std.testing.io, std.testing.allocator, "", .{});
    defer client.deinit();
    try std.testing.expectError(error.MissingApiKey, client.listModels());
}
