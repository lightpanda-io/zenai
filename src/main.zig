const std = @import("std");
const zenai = @import("zenai");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const api_key = std.posix.getenv("GOOGLE_API_KEY") orelse
        std.posix.getenv("GEMINI_API_KEY") orelse
        {
            std.debug.print("Error: set GOOGLE_API_KEY or GEMINI_API_KEY environment variable\n", .{});
            std.process.exit(1);
        };

    var client = zenai.Client.init(allocator, api_key, .{});
    defer client.deinit();

    const response = try client.generateContentFromText(
        "gemini-2.5-flash",
        "What is your name?",
        .{ .temperature = 0 },
        .{},
    );
    defer client.freeResponse(response);

    if (response.text()) |text| {
        std.debug.print("{s}\n", .{text});
    } else {
        std.debug.print("No response text received\n", .{});
    }
}
