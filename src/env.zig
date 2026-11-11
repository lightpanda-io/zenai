//! Environment variable lookup that works the same on every OS.

const std = @import("std");
const builtin = @import("builtin");

/// `Environ.getPosix` that also works on Windows, where std only offers a
/// WTF-16 lookup. There, values come from a WTF-8 copy of the process
/// environment made once on first use and kept for the process lifetime,
/// the same lifetime getPosix values have, so lookups never allocate.
/// Returns null if the variable is unset, or on Windows if that one-time
/// copy cannot be allocated.
pub fn get(environ: std.process.Environ, key: []const u8) ?[:0]const u8 {
    if (comptime builtin.os.tag != .windows) return environ.getPosix(key);
    if (!environ.block.use_global) return null;
    const block = @atomicLoad(?[*:0]const u8, &windows_env.block, .acquire) orelse
        windows_env.init() orelse return null;
    return windows_env.lookup(block, key);
}

const windows_env = struct {
    var block: ?[*:0]const u8 = null;

    fn init() ?[*:0]const u8 {
        const windows = std.os.windows;
        const peb = windows.peb();
        const copy = copy: {
            std.debug.assert(windows.ntdll.RtlEnterCriticalSection(peb.FastPebLock) == .SUCCESS);
            defer std.debug.assert(windows.ntdll.RtlLeaveCriticalSection(peb.FastPebLock) == .SUCCESS);
            break :copy toWtf8(std.heap.page_allocator, peb.ProcessParameters.Environment) catch return null;
        };
        // Another thread may have won the race to publish its copy.
        if (@cmpxchgStrong(?[*:0]const u8, &block, null, copy.ptr, .acq_rel, .acquire)) |winner| {
            std.heap.page_allocator.free(copy);
            return winner;
        }
        return copy.ptr;
    }

    /// Transcodes a Windows environment block ("k=v\0k=v\0\0") to WTF-8,
    /// keeping the NUL separators and the final empty entry.
    fn toWtf8(allocator: std.mem.Allocator, wblock: [*:0]const u16) ![:0]u8 {
        var len: usize = 0;
        while (wblock[len] != 0) len += std.mem.len(wblock + len) + 1;
        const wtf16 = wblock[0..len];
        const out = try allocator.allocSentinel(u8, std.unicode.calcWtf8Len(wtf16), 0);
        _ = std.unicode.wtf16LeToWtf8(out, wtf16);
        return out;
    }

    /// Case-insensitive, like getWindows. Entries may start with '=' (the
    /// hidden per-drive "=C:" variables), so the separator search starts
    /// at index 1.
    fn lookup(wtf8_block: [*:0]const u8, key: []const u8) ?[:0]const u8 {
        if (key.len == 0 or std.mem.findScalar(u8, key[1..], '=') != null) return null;
        if (!std.unicode.wtf8ValidateSlice(key)) return null;
        var i: usize = 0;
        while (wtf8_block[i] != 0) {
            const entry = std.mem.span(wtf8_block + i);
            i += entry.len + 1;
            const eq = std.mem.findScalarPos(u8, entry, 1, '=') orelse continue;
            if (std.os.windows.eqlIgnoreCaseWtf8(key, entry[0..eq])) return entry[eq + 1 ..];
        }
        return null;
    }
};

test "get: windows environment block lookup" {
    // The Windows path reads the live PEB, so exercise the transcode and
    // lookup on a synthetic block, which runs on every target.
    const wblock = std.unicode.wtf8ToWtf16LeStringLiteral(
        "=C:=C:\\work\x00" ++
            "Path=C:\\bin\x00" ++
            "OPENAI_API_KEY=sk-\xc3\xa9\x00" ++
            "LONE=\xed\xa0\x80\x00" ++
            "EMPTY=\x00",
    );
    const block = try windows_env.toWtf8(std.testing.allocator, wblock);
    defer std.testing.allocator.free(block);

    try std.testing.expectEqualStrings("C:\\bin", windows_env.lookup(block, "PATH").?);
    try std.testing.expectEqualStrings("sk-\xc3\xa9", windows_env.lookup(block, "openai_api_key").?);
    // An unpaired surrogate is valid WTF-16 and comes back as WTF-8.
    try std.testing.expectEqualStrings("\xed\xa0\x80", windows_env.lookup(block, "LONE").?);
    try std.testing.expectEqualStrings("", windows_env.lookup(block, "EMPTY").?);
    try std.testing.expectEqualStrings("C:\\work", windows_env.lookup(block, "=C:").?);
    try std.testing.expect(windows_env.lookup(block, "MISSING") == null);
    try std.testing.expect(windows_env.lookup(block, "Pa=th") == null);
    try std.testing.expect(windows_env.lookup(block, "") == null);
}

test "get: windows reads the live process environment" {
    if (comptime builtin.os.tag != .windows) return error.SkipZigTest;
    // PATH is set in every process. The second lookup must return the same
    // memory: values point into the one-time copy, nothing per call.
    const value = get(.{ .block = .global }, "PATH") orelse return error.SkipZigTest;
    try std.testing.expect(value.len > 0);
    try std.testing.expectEqual(value.ptr, get(.{ .block = .global }, "path").?.ptr);
    try std.testing.expect(get(.empty, "PATH") == null);
}
