const std = @import("std");
const Allocator = std.mem.Allocator;

file: ?std.fs.File,
size: u64,
lines: std.MultiArrayList(Line),
cursors: std.ArrayListUnmanaged(Cursor),
// kinda odd, but the scroll is a per file thing but
// is also associated with the `TextEditor.View`
scroll_pos: [2]u32,

pub const Line = struct {
    data: std.ArrayListUnmanaged(u8) = .{},
};

// todo: cursor should be file relative
pub const Cursor = struct {
    pos: [2]u32 = .{ 0, 0 },
    x: u32 = 0,
    // todo: selection
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

pub fn enter(self: *@This(), allocator: Allocator) !void {
    for (0..self.cursors.items.len) |i| {
        const cursor = &self.cursors.items[i];

        var new_line = Line{};

        const curr_line = &self.lines.items(.data)[cursor.pos[1]];
        // todo: respect tabulation

        if (cursor.x < curr_line.items.len) {
            try new_line.data.appendSlice(allocator, curr_line.items[cursor.x..]);
            curr_line.items.len = cursor.x;
        }

        // todo: proper cursor movement
        cursor.pos[1] += 1;
        cursor.pos[0] = 0;
        cursor.x = 0;

        try self.lines.insert(allocator, cursor.pos[1], new_line);
    }
}

pub fn insertChar(self: *@This(), allocator: Allocator, unicode: u32) !void {
    var buffer: [4]u8 = undefined;
    const len = try std.unicode.utf8Encode(@truncate(unicode), &buffer);

    // todo: ignore back space
    if (len == 1 and std.ascii.isControl(buffer[0])) insert: {
        if (buffer[0] == '\t') {
            break :insert;
        } else if (buffer[0] == std.ascii.control_code.bs) {
            try self.backspace(allocator);
        } else if (buffer[0] == '\r' or buffer[0] == '\n') {
            try self.enter(allocator);
        }
        return;
    }

    for (0..self.cursors.items.len) |i| {
        const cursor = &self.cursors.items[i];
        const line = &self.lines.items(.data)[cursor.pos[1]];
        try line.insertSlice(allocator, cursor.x, buffer[0..len]);

        // todo: proper cursor movement
        cursor.x += 1;
        cursor.pos[0] = cursor.x;
    }
}

pub fn delete(self: *@This(), allocator: Allocator) !void {
    for (0..self.cursors.items.len) |i| {
        const cursor = &self.cursors.items[i];
        const line_number = cursor.pos[1];
        const line = &self.lines.items(.data)[line_number];
        if (cursor.x == line.items.len or line.items.len == 0) {
            if (line_number < self.lines.len) {
                const next_line_number = line_number + 1;
                var next_line = self.lines.items(.data)[next_line_number];
                _ = try line.appendSlice(allocator, next_line.items);
                self.lines.orderedRemove(next_line_number);
                next_line.deinit(allocator);
            }
        } else {
            _ = line.orderedRemove(cursor.x);
        }
    }
}

pub fn backspace(self: *@This(), allocator: Allocator) !void {
    // todo: proper curso moviment
    for (0..self.cursors.items.len) |i| {
        const cursor = &self.cursors.items[i];
        const line_number = cursor.pos[1];
        const line = &self.lines.items(.data)[line_number];
        if (cursor.x == 0) {
            if (line_number > 0) {
                // remember the current line that will be deallocated
                var curr_line = line.*;
                cursor.pos[1] -= 1;
                if (line.items.len > 0) {
                    const prev_line = &self.lines.items(.data)[cursor.pos[1]];
                    cursor.x = @intCast(prev_line.items.len);
                    cursor.pos[0] = cursor.x;
                    try prev_line.appendSlice(allocator, line.items);
                    self.lines.orderedRemove(line_number);
                }
                curr_line.deinit(allocator);
            }
        } else {
            cursor.x -= 1;
            cursor.pos[0] = cursor.x;
            _ = line.orderedRemove(cursor.x);
        }
    }
}

pub fn flush(self: *@This()) !void {
    _ = self; // autofix
}
