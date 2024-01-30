const std = @import("std");
const Allocator = std.mem.Allocator;

file: ?std.fs.File,
size: u64,
lines: std.MultiArrayList(Line),
cursors: std.ArrayListUnmanaged(Cursor),
scroll_pos: [2]u32,

pub const Line = struct {
    data: std.ArrayListUnmanaged(u8),
};

// todo: cursor should be file relative
pub const Cursor = struct {
    pos: [2]u32 = .{ 0, 0 },
    x: u32 = 0,
};

pub fn deinit(self: *@This(), allocator: Allocator) void {
    self.clear(allocator);
    self.lines.deinit(allocator);
    self.cursors.deinit(allocator);
}

pub fn clear(self: *@This(), allocator: Allocator) void {
    for (self.lines.items(.data)) |*data| {
        data.deinit(allocator);
    }
    self.lines.len = 0;

    // required because clear is also called when open a file
    if (self.cursors.items.len > 0) {
        self.cursors.items.len = 1;
        self.cursors.items[0] = .{};
    }
}

pub fn open(allocator: Allocator, path: []const u8) !@This() {
    const realpath = try std.fs.cwd().realpathAlloc(allocator, path);
    defer allocator.free(realpath);

    const file = try std.fs.openFileAbsolute(realpath, .{ .mode = .read_write });
    const size: u64 = @intCast(try file.getEndPos());

    var self = @This(){
        .file = file,
        .size = size,
        .lines = .{},
        .cursors = .{},
        .scroll_pos = .{ 0, 0 },
    };

    try self.readFile(allocator);

    try self.cursors.append(allocator, .{});

    return self;
}

pub fn readFile(self: *@This(), allocator: Allocator) !void {
    if (self.file) |file| {
        try file.seekTo(0);

        const buffer = try allocator.alloc(u8, @intCast(self.size));
        defer allocator.free(buffer);

        self.clear(allocator);

        _ = try file.readAll(buffer);

        // break down the file into multiple lines
        var buffer_view = buffer;
        while (std.mem.indexOfScalar(u8, buffer_view, '\n')) |line_end| {
            // account for the \r\n line end style
            var len = line_end;
            if (line_end > 0 and buffer_view[line_end - 1] == '\r') {
                len -= 1;
            }

            // todo: handle the case where the line is very long

            var line = Line{ .data = .{} };
            try line.data.ensureTotalCapacity(allocator, len);
            line.data.appendSliceAssumeCapacity(buffer_view[0..len]);
            try self.lines.append(allocator, line);

            buffer_view = buffer_view[line_end + 1 ..];

            // insert an empty last line
            if (buffer_view.len == 0) {
                try self.lines.append(allocator, Line{ .data = .{} });
                return;
            }
        }

        if (buffer_view.len > 0) {
            var line = Line{ .data = .{} };
            try line.data.appendSlice(allocator, buffer_view);
            try self.lines.append(allocator, line);
        }
    }
}

pub fn flush(self: *@This()) !void {
    _ = self; // autofix
}
