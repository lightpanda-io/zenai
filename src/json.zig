const std = @import("std");

/// Parse a JSON string into tagged union `U`, whose known wire values are void
/// tags and whose catch-all is `unknown: []const u8`. An unrecognized value is
/// duped into `allocator` and preserved instead of failing the parse, matching
/// the Go SDKs' string-backed enums. Pair with `stringifyStringUnion`.
pub fn parseStringUnion(
    comptime U: type,
    allocator: std.mem.Allocator,
    source: anytype,
    options: std.json.ParseOptions,
) !U {
    comptime if (!@hasField(U, "unknown"))
        @compileError(@typeName(U) ++ " needs an `unknown: []const u8` field for parseStringUnion");
    const token = try source.nextAllocMax(allocator, .alloc_if_needed, options.max_value_len.?);
    defer switch (token) {
        .allocated_string => |s| allocator.free(s),
        else => {},
    };
    const slice = switch (token) {
        inline .string, .allocated_string => |s| s,
        else => return error.UnexpectedToken,
    };
    inline for (@typeInfo(U).@"union".fields) |f| {
        if (f.type == void and std.mem.eql(u8, f.name, slice)) return @unionInit(U, f.name, {});
    }
    return .{ .unknown = try allocator.dupe(u8, slice) };
}

/// Serialize a `parseStringUnion`-style union to its wire string: the tag name
/// for a known value, or the raw payload for `unknown`.
pub fn stringifyStringUnion(value: anytype, jws: anytype) !void {
    switch (value) {
        .unknown => |s| try jws.write(s),
        else => try jws.write(@tagName(value)),
    }
}

/// Returns a namespace whose `jsonParse`/`jsonStringify` are bound to the
/// string-backed union `U`, so each such union wires up its JSON hooks in two
/// name-free lines instead of re-pasting the forwarder bodies:
///
///     pub const jsonParse = jsonutil.StringUnionMethods(@This()).jsonParse;
///     pub const jsonStringify = jsonutil.StringUnionMethods(@This()).jsonStringify;
pub fn StringUnionMethods(comptime U: type) type {
    return struct {
        pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !U {
            return parseStringUnion(U, allocator, source, options);
        }
        pub fn jsonStringify(self: U, jws: anytype) !void {
            return stringifyStringUnion(self, jws);
        }
    };
}

/// Deep-copy a `std.json.Value`, duplicating all owned strings and containers.
pub fn dupeValue(a: std.mem.Allocator, value: std.json.Value) std.mem.Allocator.Error!std.json.Value {
    return switch (value) {
        .null, .bool, .integer, .float => value,
        .number_string => |s| .{ .number_string = try a.dupe(u8, s) },
        .string => |s| .{ .string = try a.dupe(u8, s) },
        .array => |arr| blk: {
            var new_arr = try std.json.Array.initCapacity(a, arr.items.len);
            for (arr.items) |item| {
                new_arr.appendAssumeCapacity(try dupeValue(a, item));
            }
            break :blk .{ .array = new_arr };
        },
        .object => |obj| blk: {
            var new_obj = std.json.ObjectMap.init(a);
            try new_obj.ensureTotalCapacity(@intCast(obj.count()));
            var it = obj.iterator();
            while (it.next()) |entry| {
                new_obj.putAssumeCapacity(try a.dupe(u8, entry.key_ptr.*), try dupeValue(a, entry.value_ptr.*));
            }
            break :blk .{ .object = new_obj };
        },
    };
}

/// Serialize a `std.json.Value` to a JSON string, allocated with `a`.
pub fn valueToString(a: std.mem.Allocator, val: std.json.Value) std.mem.Allocator.Error![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(a);
    std.json.Stringify.value(val, .{}, &aw.writer) catch return error.OutOfMemory;
    return aw.written();
}
