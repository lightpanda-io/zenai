const std = @import("std");

/// Retry policy for HTTP requests. Transient failures (5xx statuses,
/// 429 rate limits, and certain network-level errors) are retried with
/// exponential backoff and optional jitter.
///
/// The default policy is active for every `fetchPost` / `fetchGet` call
/// across the provider clients. Callers that want to opt out can pass
/// `RetryPolicy.disabled` via `InitOptions.retry_policy`.
pub const RetryPolicy = struct {
    /// Total attempts (including the first). `1` disables retry.
    max_attempts: u8 = 4,
    /// Backoff for the first retry. Doubled on each subsequent retry.
    initial_backoff_ms: u32 = 1000,
    /// Hard cap on per-retry sleep.
    max_backoff_ms: u32 = 16_000,
    /// Apply ±25% jitter to each sleep to avoid thundering-herd on rate
    /// limits. Randomness comes from `std.crypto.random`.
    jitter: bool = true,

    /// No retry — return errors immediately on the first attempt.
    pub const disabled: RetryPolicy = .{ .max_attempts = 1 };

    /// Aggressive retry tuned for long-running agents that can afford to
    /// wait out per-minute rate-limit windows. Total possible wait is
    /// ~2+4+8+16+60+60 = 150 s across 5 retries, enough to ride out an
    /// Anthropic TPM window twice. Use when the calling context is
    /// already long-running (per-task timeouts in the minutes) and a
    /// transient 429 is cheaper to wait out than to surface.
    pub const long_running: RetryPolicy = .{
        .max_attempts = 6,
        .initial_backoff_ms = 2000,
        .max_backoff_ms = 60_000,
    };
};

/// HTTP status codes that are considered transient and worth retrying.
///   429 rate-limited
///   500 internal server error
///   502 bad gateway
///   503 service unavailable
///   504 gateway timeout
///   529 overloaded (Anthropic)
pub fn isRetryableStatus(code: u10) bool {
    return switch (code) {
        429, 500, 502, 503, 504, 529 => true,
        else => false,
    };
}

/// Whether a `std.http.Client.FetchError` is transient enough to be worth
/// retrying. Keep the list conservative: only errors that are clearly
/// network-level flakes. Permanent errors (unknown host, unsupported URI
/// scheme, allocator failure) fall through and bubble up unchanged.
pub fn isRetryableFetchError(err: anyerror) bool {
    return switch (err) {
        // Transient TCP/connection issues
        error.ConnectionRefused,
        error.ConnectionTimedOut,
        error.ConnectionResetByPeer,
        error.NetworkUnreachable,
        error.TemporaryNameServerFailure,
        error.TlsInitializationFailed,
        // Pooled keep-alive connection the server closed while idle; retrying
        // opens a fresh one. Common with local servers (llama.cpp, Ollama).
        error.HttpConnectionClosing,
        // HTTP chunk truncation is typically a flaky upstream
        error.HttpChunkTruncated,
        => true,
        else => false,
    };
}

/// Compute the backoff delay (ms) for a given 0-based attempt number.
/// `attempt == 0` is the delay before the *first retry* (after the
/// initial attempt fails), so callers pass the retry index, not the
/// attempt count. Saturates to `max_backoff_ms` for large attempts
/// (including ones whose doubled base would overflow u32).
pub fn backoffMs(attempt: u8, policy: RetryPolicy) u32 {
    if (policy.initial_backoff_ms == 0) return 0;
    const shift: u5 = @intCast(@min(attempt, 20));
    const base = std.math.shl(u32, policy.initial_backoff_ms, shift);
    // Once the shift saturates to 0 or past the ceiling, clamp.
    const capped = if (base == 0 or base > policy.max_backoff_ms)
        policy.max_backoff_ms
    else
        base;
    if (!policy.jitter) return capped;
    const jitter_range = capped / 4;
    if (jitter_range == 0) return capped;
    // Uniform ±25% around the base, via [capped - jitter_range, capped + jitter_range).
    const offset = std.crypto.random.uintLessThan(u32, jitter_range * 2);
    return capped - jitter_range + offset;
}

/// Thin wrapper over `std.Thread.sleep` in ms. Split out so tests can
/// fake it (the callers only use this one entry point).
pub fn sleepMs(ms: u32) void {
    std.Thread.sleep(@as(u64, ms) * std.time.ns_per_ms);
}

test "isRetryableStatus covers known transients" {
    try std.testing.expect(isRetryableStatus(429));
    try std.testing.expect(isRetryableStatus(500));
    try std.testing.expect(isRetryableStatus(502));
    try std.testing.expect(isRetryableStatus(503));
    try std.testing.expect(isRetryableStatus(504));
    try std.testing.expect(isRetryableStatus(529));
}

test "isRetryableStatus rejects non-retryable codes" {
    try std.testing.expect(!isRetryableStatus(200));
    try std.testing.expect(!isRetryableStatus(301));
    try std.testing.expect(!isRetryableStatus(400));
    try std.testing.expect(!isRetryableStatus(401));
    try std.testing.expect(!isRetryableStatus(403));
    try std.testing.expect(!isRetryableStatus(404));
    try std.testing.expect(!isRetryableStatus(422));
    try std.testing.expect(!isRetryableStatus(501)); // Not implemented — permanent
}

test "isRetryableFetchError picks known transients" {
    try std.testing.expect(isRetryableFetchError(error.ConnectionRefused));
    try std.testing.expect(isRetryableFetchError(error.ConnectionTimedOut));
    try std.testing.expect(isRetryableFetchError(error.ConnectionResetByPeer));
    try std.testing.expect(isRetryableFetchError(error.NetworkUnreachable));
    try std.testing.expect(isRetryableFetchError(error.TemporaryNameServerFailure));
    try std.testing.expect(isRetryableFetchError(error.TlsInitializationFailed));
    try std.testing.expect(isRetryableFetchError(error.HttpChunkTruncated));
    try std.testing.expect(isRetryableFetchError(error.HttpConnectionClosing));
}

test "isRetryableFetchError rejects permanent errors" {
    try std.testing.expect(!isRetryableFetchError(error.UnknownHostName));
    try std.testing.expect(!isRetryableFetchError(error.UnsupportedUriScheme));
    try std.testing.expect(!isRetryableFetchError(error.HostLacksNetworkAddresses));
    try std.testing.expect(!isRetryableFetchError(error.OutOfMemory));
    try std.testing.expect(!isRetryableFetchError(error.CertificateBundleLoadFailure));
}

test "backoffMs without jitter doubles and caps" {
    const policy: RetryPolicy = .{ .initial_backoff_ms = 1000, .max_backoff_ms = 16_000, .jitter = false };
    try std.testing.expectEqual(@as(u32, 1000), backoffMs(0, policy));
    try std.testing.expectEqual(@as(u32, 2000), backoffMs(1, policy));
    try std.testing.expectEqual(@as(u32, 4000), backoffMs(2, policy));
    try std.testing.expectEqual(@as(u32, 8000), backoffMs(3, policy));
    try std.testing.expectEqual(@as(u32, 16_000), backoffMs(4, policy));
    try std.testing.expectEqual(@as(u32, 16_000), backoffMs(5, policy));
    try std.testing.expectEqual(@as(u32, 16_000), backoffMs(50, policy)); // No overflow
}

test "backoffMs with jitter stays within ±25% of the capped base" {
    const policy: RetryPolicy = .{ .initial_backoff_ms = 1000, .max_backoff_ms = 16_000, .jitter = true };
    for (0..64) |_| {
        const ms0 = backoffMs(0, policy);
        try std.testing.expect(ms0 >= 750);
        try std.testing.expect(ms0 < 1250);
        const ms3 = backoffMs(3, policy);
        try std.testing.expect(ms3 >= 6000);
        try std.testing.expect(ms3 < 10000);
    }
}

test "RetryPolicy.disabled gives a single attempt" {
    try std.testing.expectEqual(@as(u8, 1), RetryPolicy.disabled.max_attempts);
}
