//! TypeSafe System One API client. https://docs.typesafe.ai
//!
//! System One serves Jev, a judgement model rather than a chat model: post the
//! `state` under judgement plus a map of typed questions about it and get one
//! typed answer per question back — a probability (`noul`), a labelled
//! `choice` with per-option probabilities, or a `score` on an ordered scale.
//! There is no free-text completion and no tool calling, which is why this
//! client stays out of `provider.Client`.
//!
//! Jev ingests the `state` once and evaluates every question against it in
//! parallel, so packing several questions — including speculative ones the
//! caller may discard — into one request costs far less than one call each.

const std = @import("std");
const types = @import("types.zig");
const http = @import("../http.zig");
const retry = @import("../retry.zig");

pub const RetryPolicy = retry.RetryPolicy;

const AskOptions = types.AskOptions;
const AskResponse = types.AskResponse;
const Content = types.Content;
const ListModelsResponse = types.ListModelsResponse;
const QuestionEntry = types.QuestionEntry;

const Client = @This();

allocator: std.mem.Allocator,
api_key: []const u8,
base_url: []const u8,
http_client: std.http.Client,
retry_policy: RetryPolicy,
request_timeout_ms: ?u32,
last_error: http.ErrorDetail = .{},

/// TypeSafe's own endpoint. A caller routing through something that serves the
/// same protocol — Vercel AI Gateway does, at `/typesafe` — overrides it.
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
}

pub const Response = http.Response;

pub const ApiError = error{MissingApiKey} || http.FetchError;

pub fn setErrorDetail(self: *Client, status_code: u10, body: []const u8) void {
    self.last_error.setLogged(self.allocator, status_code, body, "TypeSafe");
}

/// Caller frees `[0].value`.
fn authHeader(self: *Client) std.mem.Allocator.Error![1]std.http.Header {
    return .{.{
        .name = "Authorization",
        .value = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_key}),
    }};
}

/// Ask Jev one or more typed questions about `state`. Each entry's `key` is an
/// id you choose, and the matching answer comes back under that same id
/// (`response.value.answer(id)`).
///
/// A `choice` answer is only safe to act on once `types.validateChoice` has
/// confirmed it stayed inside the option set it was given.
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
    questions: []const QuestionEntry,
    options: AskOptions,
) ApiError!Response(AskResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;

    const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/systemone", .{self.base_url});
    defer self.allocator.free(url);

    var request = options;
    request.state = state;
    request.questions = .init(questions);

    const auth = try self.authHeader();
    defer self.allocator.free(auth[0].value);

    return http.postJsonWithRetry(self.allocator, &self.http_client, self.retry_policy, self.request_timeout_ms, url, &auth, request, AskResponse, self);
}

/// List the model names and aliases this key may send in `AskOptions.model`.
/// Caller owns the returned `Response` and must call `deinit()`.
pub fn listModels(self: *Client) ApiError!Response(ListModelsResponse) {
    if (self.api_key.len == 0) return error.MissingApiKey;

    const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/models", .{self.base_url});
    defer self.allocator.free(url);

    const auth = try self.authHeader();
    defer self.allocator.free(auth[0].value);

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
        &.{.{ .key = "q", .value = .noulText("Is this a complaint?") }},
        .{},
    ));
}

test "listModels rejects empty api key" {
    var client = init(std.testing.io, std.testing.allocator, "", .{});
    defer client.deinit();
    try std.testing.expectError(error.MissingApiKey, client.listModels());
}
