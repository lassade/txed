const std = @import("std");
const Allocator = std.mem.Allocator;

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

    pub inline fn makeBitmap(self: *const FontInfo, output: [*]u8, out_w: c_int, out_h: c_int, out_stride: c_int, scale_x: f32, scale_y: f32, index: c_int) void {
        fii.stbtt_MakeGlyphBitmap(self, output, out_w, out_h, out_stride, scale_x, scale_y, index);
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
