const std = @import("std");
const Allocator = std.mem.Allocator;

const assets = @import("assets.zig");
const Res = assets.Res;
const math = @import("../math.zig");
const rect_pack = @import("../rect_pack.zig");
const util = @import("../util.zig");
const Mesh = @import("Mesh.zig");
const Texture = @import("Texture.zig");
const StaticMesh = @import("./StaticMesh.zig");

pub const Error = error{BadData};

pub const FontInfo = extern struct {
    pub const Buffer = extern struct {
        data: [*c]u8,
        cursor: c_int,
        size: c_int,
    };

    pub const VMetrics = struct {
        ascent: c_int,
        descent: c_int,
        line_gap: c_int,
    };

    pub const HMetrics = struct {
        advance_width: c_int,
        left_side_bearing: c_int,
    };

    pub const Box = struct {
        x0: c_int,
        y0: c_int,
        x1: c_int,
        y1: c_int,
    };

    userdata: ?*anyopaque,
    data: [*]u8,
    fontstart: c_int,
    numGlyphs: c_int,
    loca: c_int,
    head: c_int,
    glyf: c_int,
    hhea: c_int,
    hmtx: c_int,
    kern: c_int,
    gpos: c_int,
    svg: c_int,
    index_map: c_int,
    indexToLocFormat: c_int,
    cff: Buffer,
    charstrings: Buffer,
    gsubrs: Buffer,
    subrs: Buffer,
    fontdicts: Buffer,
    fdselect: Buffer,

    pub fn init(data: []const u8, offset: c_int) Error!FontInfo {
        var font_info: FontInfo = undefined;
        if (fii.stbtt_InitFont(&font_info, data.ptr, offset) == 0) {
            return Error.BadData;
        }
        return font_info;
    }

    pub inline fn scaleForPixelHeight(self: *const FontInfo, pixels: f32) f32 {
        return fii.stbtt_ScaleForPixelHeight(self, pixels);
    }

    pub inline fn getFontVMetrics(self: *const FontInfo) VMetrics {
        var v_metrics: VMetrics = undefined;
        fii.stbtt_GetFontVMetrics(self, &v_metrics.ascent, &v_metrics.descent, &v_metrics.line_gap);
        return v_metrics;
    }

    pub inline fn findCodepointIndex(self: *const FontInfo, codepoint: c_int) c_int {
        return fii.stbtt_FindGlyphIndex(self, codepoint);
    }

    pub inline fn getHMetrics(self: *const FontInfo, index: c_int) HMetrics {
        var h_metrics: HMetrics = undefined;
        fii.stbtt_GetGlyphHMetrics(self, index, &h_metrics.advance_width, &h_metrics.left_side_bearing);
        return h_metrics;
    }

    pub inline fn getBitmapBox(self: *const FontInfo, index: c_int, scale_x: f32, scale_y: f32) Box {
        var box: Box = undefined;
        fii.stbtt_GetGlyphBitmapBox(self, index, scale_x, scale_y, &box.x0, &box.y0, &box.x1, &box.y1);
        return box;
    }

    pub inline fn makeBitmap(self: *const FontInfo, output: *u8, out_w: c_int, out_h: c_int, out_stride: c_int, scale_x: f32, scale_y: f32, index: c_int) void {
        fii.stbtt_MakeGlyphBitmap(self, output, out_w, out_h, out_stride, scale_x, scale_y, index);
    }
};

pub const Font = struct {
    pub const Ref = assets.AssetRef(@This());

    pub const tag = @import("../meta.zig").assets.tagOf(@This());

    pub const Char = packed struct {
        size: u11,
        codepoint: u21,
    };

    pub const CharInfo = struct {
        size: @Vector(2, f32),
        uv: @Vector(4, f16),
        offset: @Vector(2, f32),
        xadvance: f32,
    };

    pub const CharHashMapUnmanaged = std.HashMapUnmanaged(Char, CharInfo, util.SplittableHash(Char), std.hash_map.default_max_load_percentage);

    // pub const CharPair = packed struct {
    //     a: c_int,
    //     b: c_int,
    // };

    allocator: Allocator,
    info: FontInfo,
    tex: Texture.Ref = .{},
    packer: rect_pack.RectPacker,
    cache: CharHashMapUnmanaged = .{},

    /// owned font data
    data: ?[]const u8 = null,

    pub fn init(allocator: Allocator, data: []const u8) !Font {
        return .{
            .info = try FontInfo.init(data, 0),
            .packer = try rect_pack.RectPacker.init(allocator, .{ .allow_flipping = false }),
            .data = data,
        };
    }

    // todo: preCacheCharRange, preCacheChars

    pub fn getCharInfo(self: *Font, codepoint: u21, size: u11, res: *Res) !CharInfo {
        const entry = try self.cache.getOrPut(res.allocator, Char{
            .size = size,
            .codepoint = codepoint,
        });

        if (!entry.found_existing) {
            // slow path
            if (!self.tex.inner.isAlive()) {
                // todo:
                unreachable;
                // // owned texture
                // const tmp = try Texture.init(
                //     res.allocator,
                //     .r8_unorm,
                //     .{ self.packer.config.max_width, self.packer.config.max_height },
                //     false,
                // );
                // @memset(tmp.data, 0);
                // self.tex = try res.textures.append(tmp);
            }
            const tex: *Texture = &res.textures.entries.items[self.tex.index].occupied;

            const i = self.info.findCodepointIndex(@intCast(codepoint));
            const scale = self.info.scaleForPixelHeight(@floatFromInt(size));
            const box = self.info.getBitmapBox(i, scale, scale);

            const w = box.x1 - box.x0;
            const h = box.y1 - box.y0;
            var r = try self.packer.insert(@intCast(w + 2), @intCast(h + 2));
            std.debug.assert(!r.is_flipped); // flip not supported

            // remove padding
            r.rect.x += 1;
            r.rect.y += 1;
            r.rect.w -= 2;
            r.rect.h -= 2;

            const stride = tex.size[0];
            self.info.makeBitmap(&tex.data[stride * r.rect.y + r.rect.x], w, h, @intCast(stride), scale, scale, i);

            const h_metrics = self.info.getHMetrics(i);

            var info: CharInfo = undefined;
            info.size[0] = @floatFromInt(w);
            info.size[1] = @floatFromInt(h);
            const n: math.F32x2 = math.recip(@as(math.F32x2, @floatFromInt(tex.size)));
            info.uv[0] = @floatFromInt(r.rect.y + r.rect.h);
            info.uv[1] = @floatFromInt(r.rect.y);
            info.uv[2] = @floatFromInt(r.rect.x);
            info.uv[3] = @floatFromInt(r.rect.x + r.rect.w);
            info.uv *= @floatCast(math.f32x4(n[0], n[1], n[0], n[1]));
            info.offset[0] = @floatFromInt(box.x0);
            info.offset[1] = @floatFromInt(box.y0);
            info.xadvance = @as(f32, @floatFromInt(h_metrics.advance_width)) * scale;

            entry.value_ptr.* = info;
        }

        return entry.value_ptr.*;
    }

    pub fn deinit(self: *Font) void {
        if (self.data) |data| {
            self.allocator.free(data);
        }
        self.tex.deinit();
        self.packer.deinit();
        self.cache.deinit(self.allocator);
    }

    pub fn loadFromPath(uri: [:0]const u8, res: *Res) !Ref {
        var group = res.group(@This());
        if (res.findAsset(uri)) |link| {
            std.debug.assert(tag.eq(link.tag));
            if (group.get(link.id)) |asset| {
                // todo: duplicate reference, but it has to deal with the AssetId thing
                return asset.cloneRef();
            }
        }

        const data = try res.packs.loadFile(res.allocator, uri);
        defer res.allocator.free(data);

        // create asset
        var font_asset = try group.create();
        const font = font_asset.as(@This());
        try res.linkAsset(uri, font_asset);

        font.* = try Font.init(res.allocator, data);

        return font_asset.ref();
    }
};

pub const Text = struct {
    font_id: Font.Id = .{},
    size: u11 = 14,

    renderer: StaticMesh = .{
        .shader_id = .r_mask,
        .tex_id = Texture.Id.DEFAULT,
    },

    // mesh buffers
    allocator: Allocator,
    verts: []Mesh.Vert = &.{},
    indices: []u32 = &.{},

    // todo:
    // /// bound box is a min max rectangle stored as `{ x0, y0, x1, y1, }`
    // bound_box: @Vector(4, f32),

    pub fn init(allocator: Allocator) Text {
        return .{ .allocator = allocator };
    }

    pub fn update(self: *Text, text: []const u8, res: *assets.Res) !void {
        if (!res.meshes.isAlive(self.renderer.mesh)) {
            // owned mesh
            self.renderer.mesh = try res.meshes.append(Mesh{
                .read_only = false,
                // data is owned by the line it self because the mesh
                // may only have a sub-slice of the acctual data
                .allocator = null,
            });
        }
        var mesh: *Mesh = &res.meshes.entries.items[self.renderer.mesh.index].occupied;

        var v: u32 = 0;
        var i: u32 = 0;

        if (res.fonts.getPtr(self.font_id)) |font| {
            // resize mesh buffers
            const vc = text.len * 4;
            if (self.verts.len < vc) try util.reallocNoCopy(Mesh.Vert, self.allocator, &self.verts, vc);

            const ic = text.len * 6;
            if (self.indices.len < ic) try util.reallocNoCopy(u32, self.allocator, &self.indices, ic);

            const color: @Vector(4, u8) = @splat(255);

            var cursor: @Vector(2, f32) = @splat(0.0);
            var chars = std.unicode.Utf8View.initUnchecked(text).iterator();
            while (chars.nextCodepoint()) |codepoint| {
                const info: Font.CharInfo = try font.getCharInfo(codepoint, self.size, res);
                // todo: Kerning

                // todo: enter
                // todo: space and other blank characters

                // push quads
                self.indices[i + 0] = v + 2;
                self.indices[i + 1] = v + 1;
                self.indices[i + 2] = v + 0;
                self.indices[i + 3] = v + 2;
                self.indices[i + 4] = v + 3;
                self.indices[i + 5] = v + 1;
                i += 6;

                const offset = cursor + info.offset;
                const r = math.f32x4(info.size[1], 0.0, 0.0, info.size[0]) + math.f32x4(offset[1], offset[1], offset[0], offset[0]);
                self.verts[v + 0] = .{ .pos = .{ r[2], r[0] }, .uv0 = .{ info.uv[2], info.uv[0] }, .color = color };
                self.verts[v + 1] = .{ .pos = .{ r[3], r[0] }, .uv0 = .{ info.uv[3], info.uv[0] }, .color = color };
                self.verts[v + 2] = .{ .pos = .{ r[2], r[1] }, .uv0 = .{ info.uv[2], info.uv[1] }, .color = color };
                self.verts[v + 3] = .{ .pos = .{ r[3], r[1] }, .uv0 = .{ info.uv[3], info.uv[1] }, .color = color };
                v += 4;

                cursor[0] += info.xadvance;
            }

            // assign texture at the end
            self.renderer.tex = font.tex_id;
        }

        mesh.dirty = true;
        mesh.read_only = false;
        mesh.verts = self.verts[0..v];
        mesh.indices = self.indices[0..i];
    }

    pub fn deinit(self: Text, res: *assets.Res) void {
        if (self.verts.len > 0) self.allocator.free(self.verts);
        if (self.indices.len > 0) self.allocator.free(self.indices);

        const mesh_id = self.renderer.mesh;
        res.meshes.remove(mesh_id);
    }
};

const fii = struct {
    pub extern fn stbtt_GetNumberOfFonts(data: [*c]const u8) c_int;
    pub extern fn stbtt_GetFontOffsetForIndex(data: [*c]const u8, index: c_int) c_int;
    pub extern fn stbtt_InitFont(info: [*c]FontInfo, data: [*c]const u8, offset: c_int) c_int;
    pub extern fn stbtt_ScaleForPixelHeight(info: [*c]const FontInfo, pixels: f32) f32;
    pub extern fn stbtt_GetFontVMetrics(info: [*c]const FontInfo, ascent: [*c]c_int, descent: [*c]c_int, lineGap: [*c]c_int) void;
    pub extern fn stbtt_FindGlyphIndex(info: [*c]const FontInfo, unicode_codepoint: c_int) c_int;
    pub extern fn stbtt_GetGlyphHMetrics(info: [*c]const FontInfo, glyph_index: c_int, advanceWidth: [*c]c_int, leftSideBearing: [*c]c_int) void;
    // pub extern fn stbtt_GetGlyphKernAdvance(info: [*c]const FontInfo, glyph1: c_int, glyph2: c_int) c_int;
    // pub extern fn stbtt_GetGlyphBox(info: [*c]const FontInfo, glyph_index: c_int, x0: [*c]c_int, y0: [*c]c_int, x1: [*c]c_int, y1: [*c]c_int) c_int;
    pub extern fn stbtt_GetGlyphBitmapBox(font: [*c]const FontInfo, glyph: c_int, scale_x: f32, scale_y: f32, ix0: [*c]c_int, iy0: [*c]c_int, ix1: [*c]c_int, iy1: [*c]c_int) void;
    pub extern fn stbtt_MakeGlyphBitmap(info: [*c]const FontInfo, output: [*c]u8, out_w: c_int, out_h: c_int, out_stride: c_int, scale_x: f32, scale_y: f32, glyph: c_int) void;
};
