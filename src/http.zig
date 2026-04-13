const std = @import("std");

/// Owns the parsed response and its backing memory.
/// Call `deinit()` when done to free all resources.
pub fn Response(comptime T: type) type {
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

/// Pagination options for list operations.
pub const ListOptions = struct {
    /// Maximum number of items to return per page.
    pageSize: ?i32 = null,
    /// Token from a previous response's `nextPageToken` to fetch the next page.
    pageToken: ?[]const u8 = null,
};

/// Deep-copy a `std.json.Value`, duplicating all owned strings and containers.
pub fn dupeJsonValue(a: std.mem.Allocator, value: std.json.Value) std.mem.Allocator.Error!std.json.Value {
    return switch (value) {
        .null, .bool, .integer, .float => value,
        .number_string => |s| .{ .number_string = try a.dupe(u8, s) },
        .string => |s| .{ .string = try a.dupe(u8, s) },
        .array => |arr| blk: {
            var new_arr = try std.json.Array.initCapacity(a, arr.items.len);
            for (arr.items) |item| {
                new_arr.appendAssumeCapacity(try dupeJsonValue(a, item));
            }
            break :blk .{ .array = new_arr };
        },
        .object => |obj| blk: {
            var new_obj = std.json.ObjectMap.init(a);
            try new_obj.ensureTotalCapacity(@intCast(obj.count()));
            var it = obj.iterator();
            while (it.next()) |entry| {
                new_obj.putAssumeCapacity(try a.dupe(u8, entry.key_ptr.*), try dupeJsonValue(a, entry.value_ptr.*));
            }
            break :blk .{ .object = new_obj };
        },
    };
}

/// Serialize a `std.json.Value` to a JSON string, allocated with `a`.
pub fn jsonValueToString(a: std.mem.Allocator, val: std.json.Value) std.mem.Allocator.Error![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(a);
    std.json.Stringify.value(val, .{}, &aw.writer) catch return error.OutOfMemory;
    return aw.written();
}

pub fn appendListParams(allocator: std.mem.Allocator, base_url: []const u8, options: ListOptions) ![]u8 {
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
