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

/// For a union whose active payload *is* the JSON value (text or a list):
///
///     pub const jsonStringify = jsonutil.PayloadUnionMethods(@This()).jsonStringify;
pub fn PayloadUnionMethods(comptime U: type) type {
    return struct {
        pub fn jsonStringify(self: U, jws: anytype) !void {
            switch (self) {
                inline else => |v| try jws.write(v),
            }
        }
    };
}

/// A JSON object whose keys are chosen at runtime rather than declared as
/// struct fields — TypeSafe's `questions`/`answers`, Gemini's schema
/// `properties`. Zig's std has no `json.ArrayHashMap`, so the entries are kept
/// as an ordered slice; wire order survives both parse and stringify, and
/// lookup is a linear scan (these maps are small).
///
/// Parsed keys and values borrow the parse arena and the response body, so the
/// owning `Parsed`/`Response` must outlive the map.
pub fn StringMap(comptime V: type) type {
    return struct {
        entries: []const Entry = &.{},

        pub const Entry = struct {
            key: []const u8,
            value: V,
        };

        const Self = @This();

        pub fn init(entries: []const Entry) Self {
            return .{ .entries = entries };
        }

        /// The first value stored under `key`, or null.
        pub fn get(self: Self, key: []const u8) ?V {
            for (self.entries) |entry| {
                if (std.mem.eql(u8, entry.key, key)) return entry.value;
            }
            return null;
        }

        pub fn count(self: Self) usize {
            return self.entries.len;
        }

        pub fn has(self: Self, key: []const u8) bool {
            for (self.entries) |entry| {
                if (std.mem.eql(u8, entry.key, key)) return true;
            }
            return false;
        }

        pub fn jsonParse(
            allocator: std.mem.Allocator,
            source: anytype,
            options: std.json.ParseOptions,
        ) std.json.ParseError(@TypeOf(source.*))!Self {
            if (.object_begin != try source.next()) return error.UnexpectedToken;

            var list: std.ArrayList(Entry) = .empty;
            while (true) {
                // Unlike std's struct parser the key is kept, not freed: it is
                // the entry's own data.
                const name_token = try source.nextAllocMax(allocator, options.allocate.?, options.max_value_len.?);
                const key = switch (name_token) {
                    inline .string, .allocated_string => |slice| slice,
                    .object_end => break,
                    else => return error.UnexpectedToken,
                };
                try list.append(allocator, .{
                    .key = key,
                    .value = try std.json.innerParse(V, allocator, source, options),
                });
            }
            return .{ .entries = try list.toOwnedSlice(allocator) };
        }

        /// Needed alongside `jsonParse`: a containing type whose own `jsonParse`
        /// buffers into a `std.json.Value` first reaches this map through the
        /// value path instead of the token path.
        pub fn jsonParseFromValue(
            allocator: std.mem.Allocator,
            source: std.json.Value,
            options: std.json.ParseOptions,
        ) std.json.ParseFromValueError!Self {
            const object = switch (source) {
                .object => |o| o,
                else => return error.UnexpectedToken,
            };
            const entries = try allocator.alloc(Entry, object.count());
            var i: usize = 0;
            var it = object.iterator();
            while (it.next()) |kv| : (i += 1) {
                entries[i] = .{
                    .key = kv.key_ptr.*,
                    .value = try std.json.innerParseFromValue(V, allocator, kv.value_ptr.*, options),
                };
            }
            return .{ .entries = entries };
        }

        pub fn jsonStringify(self: Self, jw: *std.json.Stringify) !void {
            try jw.beginObject();
            for (self.entries) |entry| {
                try jw.objectField(entry.key);
                try jw.write(entry.value);
            }
            try jw.endObject();
        }
    };
}

/// Write a struct's fields into an already-open JSON object, honouring
/// `emit_null_optional_fields` the way std does for a plain struct. For hooks
/// that emit something around a struct and cannot delegate to `jws.write`.
pub fn writeStructFields(payload: anytype, jw: *std.json.Stringify) !void {
    inline for (@typeInfo(@TypeOf(payload)).@"struct".fields) |field| {
        const value = @field(payload, field.name);
        if (comptime @typeInfo(field.type) == .optional) {
            if (value) |unwrapped| {
                try jw.objectField(field.name);
                try jw.write(unwrapped);
            } else if (jw.options.emit_null_optional_fields) {
                try jw.objectField(field.name);
                try jw.write(null);
            }
        } else {
            try jw.objectField(field.name);
            try jw.write(value);
        }
    }
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
            var new_obj: std.json.ObjectMap = .empty;
            try new_obj.ensureTotalCapacity(a, @intCast(obj.count()));
            var it = obj.iterator();
            while (it.next()) |entry| {
                new_obj.putAssumeCapacity(try a.dupe(u8, entry.key_ptr.*), try dupeValue(a, entry.value_ptr.*));
            }
            break :blk .{ .object = new_obj };
        },
    };
}

/// Serialize any value to a JSON string, allocated with `allocator`.
pub fn stringifyAlloc(allocator: std.mem.Allocator, value: anytype, options: std.json.Stringify.Options) std.mem.Allocator.Error![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    std.json.Stringify.value(value, options, &aw.writer) catch return error.OutOfMemory;
    return aw.toOwnedSlice();
}

/// Serialize a `std.json.Value` to a JSON string, allocated with `a`.
pub fn valueToString(a: std.mem.Allocator, val: std.json.Value) std.mem.Allocator.Error![]const u8 {
    return stringifyAlloc(a, val, .{});
}

test "StringMap parses an object with dynamic keys" {
    const parsed = try std.json.parseFromSlice(StringMap(f64), std.testing.allocator,
        \\{"billing":0.88,"delivery":0.1,"other":0}
    , .{});
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 3), parsed.value.count());
    try std.testing.expectEqualStrings("billing", parsed.value.entries[0].key);
    try std.testing.expectEqual(@as(?f64, 0.88), parsed.value.get("billing"));
    // An integer token still lands in an f64 value.
    try std.testing.expectEqual(@as(?f64, 0), parsed.value.get("other"));
    try std.testing.expectEqual(@as(?f64, null), parsed.value.get("absent"));
}

test "StringMap parses an empty object" {
    const parsed = try std.json.parseFromSlice(StringMap(f64), std.testing.allocator, "{}", .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 0), parsed.value.count());
}

test "StringMap stringifies entries in order" {
    const map: StringMap([]const u8) = .init(&.{
        .{ .key = "0", .value = "Calm" },
        .{ .key = "1", .value = "Furious" },
    });
    const out = try stringifyAlloc(std.testing.allocator, map, .{});
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings(
        \\{"0":"Calm","1":"Furious"}
    , out);
}
