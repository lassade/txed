//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (29)
//--------------------------------------------------------------------------------
pub const D2D_COLOR_F = extern struct {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
};

pub const D2D1_COLOR_F = extern struct {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
};

pub const D2D1_ALPHA_MODE = enum(u32) {
    UNKNOWN = 0,
    PREMULTIPLIED = 1,
    STRAIGHT = 2,
    IGNORE = 3,
    FORCE_DWORD = 4294967295,
};
pub const D2D1_ALPHA_MODE_UNKNOWN = D2D1_ALPHA_MODE.UNKNOWN;
pub const D2D1_ALPHA_MODE_PREMULTIPLIED = D2D1_ALPHA_MODE.PREMULTIPLIED;
pub const D2D1_ALPHA_MODE_STRAIGHT = D2D1_ALPHA_MODE.STRAIGHT;
pub const D2D1_ALPHA_MODE_IGNORE = D2D1_ALPHA_MODE.IGNORE;
pub const D2D1_ALPHA_MODE_FORCE_DWORD = D2D1_ALPHA_MODE.FORCE_DWORD;

pub const D2D1_PIXEL_FORMAT = extern struct {
    format: DXGI_FORMAT,
    alphaMode: D2D1_ALPHA_MODE,
};

pub const D2D_POINT_2U = extern struct {
    x: u32,
    y: u32,
};

pub const D2D_POINT_2F = extern struct {
    x: f32,
    y: f32,
};

pub const D2D_VECTOR_2F = extern struct {
    x: f32,
    y: f32,
};

pub const D2D_VECTOR_3F = extern struct {
    x: f32,
    y: f32,
    z: f32,
};

pub const D2D_VECTOR_4F = extern struct {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
};

pub const D2D_RECT_F = extern struct {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
};

pub const D2D_RECT_U = extern struct {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
};

pub const D2D_SIZE_F = extern struct {
    width: f32,
    height: f32,
};

pub const D2D_SIZE_U = extern struct {
    width: u32,
    height: u32,
};

pub const D2D_MATRIX_3X2_F = extern struct {
    Anonymous: extern union {
        Anonymous1: extern struct {
            m11: f32,
            m12: f32,
            m21: f32,
            m22: f32,
            dx: f32,
            dy: f32,
        },
        Anonymous2: extern struct {
            _11: f32,
            _12: f32,
            _21: f32,
            _22: f32,
            _31: f32,
            _32: f32,
        },
        m: [6]f32,
    },
};

pub const D2D_MATRIX_4X3_F = extern struct {
    Anonymous: extern union {
        Anonymous: extern struct {
            _11: f32,
            _12: f32,
            _13: f32,
            _21: f32,
            _22: f32,
            _23: f32,
            _31: f32,
            _32: f32,
            _33: f32,
            _41: f32,
            _42: f32,
            _43: f32,
        },
        m: [12]f32,
    },
};

pub const D2D_MATRIX_4X4_F = extern struct {
    Anonymous: extern union {
        Anonymous: extern struct {
            _11: f32,
            _12: f32,
            _13: f32,
            _14: f32,
            _21: f32,
            _22: f32,
            _23: f32,
            _24: f32,
            _31: f32,
            _32: f32,
            _33: f32,
            _34: f32,
            _41: f32,
            _42: f32,
            _43: f32,
            _44: f32,
        },
        m: [16]f32,
    },
};

pub const D2D_MATRIX_5X4_F = extern struct {
    Anonymous: extern union {
        Anonymous: extern struct {
            _11: f32,
            _12: f32,
            _13: f32,
            _14: f32,
            _21: f32,
            _22: f32,
            _23: f32,
            _24: f32,
            _31: f32,
            _32: f32,
            _33: f32,
            _34: f32,
            _41: f32,
            _42: f32,
            _43: f32,
            _44: f32,
            _51: f32,
            _52: f32,
            _53: f32,
            _54: f32,
        },
        m: [20]f32,
    },
};

pub const D2D1_FIGURE_BEGIN = enum(u32) {
    FILLED = 0,
    HOLLOW = 1,
    FORCE_DWORD = 4294967295,
};
pub const D2D1_FIGURE_BEGIN_FILLED = D2D1_FIGURE_BEGIN.FILLED;
pub const D2D1_FIGURE_BEGIN_HOLLOW = D2D1_FIGURE_BEGIN.HOLLOW;
pub const D2D1_FIGURE_BEGIN_FORCE_DWORD = D2D1_FIGURE_BEGIN.FORCE_DWORD;

pub const D2D1_FIGURE_END = enum(u32) {
    OPEN = 0,
    CLOSED = 1,
    FORCE_DWORD = 4294967295,
};
pub const D2D1_FIGURE_END_OPEN = D2D1_FIGURE_END.OPEN;
pub const D2D1_FIGURE_END_CLOSED = D2D1_FIGURE_END.CLOSED;
pub const D2D1_FIGURE_END_FORCE_DWORD = D2D1_FIGURE_END.FORCE_DWORD;

pub const D2D1_BEZIER_SEGMENT = extern struct {
    point1: D2D_POINT_2F,
    point2: D2D_POINT_2F,
    point3: D2D_POINT_2F,
};

pub const D2D1_PATH_SEGMENT = enum(u32) {
    NONE = 0,
    FORCE_UNSTROKED = 1,
    FORCE_ROUND_LINE_JOIN = 2,
    FORCE_DWORD = 4294967295,
    _,
    pub fn initFlags(o: struct {
        NONE: u1 = 0,
        FORCE_UNSTROKED: u1 = 0,
        FORCE_ROUND_LINE_JOIN: u1 = 0,
        FORCE_DWORD: u1 = 0,
    }) D2D1_PATH_SEGMENT {
        return @as(D2D1_PATH_SEGMENT, @enumFromInt((if (o.NONE == 1) @intFromEnum(D2D1_PATH_SEGMENT.NONE) else 0) | (if (o.FORCE_UNSTROKED == 1) @intFromEnum(D2D1_PATH_SEGMENT.FORCE_UNSTROKED) else 0) | (if (o.FORCE_ROUND_LINE_JOIN == 1) @intFromEnum(D2D1_PATH_SEGMENT.FORCE_ROUND_LINE_JOIN) else 0) | (if (o.FORCE_DWORD == 1) @intFromEnum(D2D1_PATH_SEGMENT.FORCE_DWORD) else 0)));
    }
};
pub const D2D1_PATH_SEGMENT_NONE = D2D1_PATH_SEGMENT.NONE;
pub const D2D1_PATH_SEGMENT_FORCE_UNSTROKED = D2D1_PATH_SEGMENT.FORCE_UNSTROKED;
pub const D2D1_PATH_SEGMENT_FORCE_ROUND_LINE_JOIN = D2D1_PATH_SEGMENT.FORCE_ROUND_LINE_JOIN;
pub const D2D1_PATH_SEGMENT_FORCE_DWORD = D2D1_PATH_SEGMENT.FORCE_DWORD;

pub const D2D1_FILL_MODE = enum(u32) {
    ALTERNATE = 0,
    WINDING = 1,
    FORCE_DWORD = 4294967295,
};
pub const D2D1_FILL_MODE_ALTERNATE = D2D1_FILL_MODE.ALTERNATE;
pub const D2D1_FILL_MODE_WINDING = D2D1_FILL_MODE.WINDING;
pub const D2D1_FILL_MODE_FORCE_DWORD = D2D1_FILL_MODE.FORCE_DWORD;

// TODO: this type is limited to platform 'windows6.1'
const IID_ID2D1SimplifiedGeometrySink_Value = Guid.initString("2cd9069e-12e2-11dc-9fed-001143a055f9");
pub const IID_ID2D1SimplifiedGeometrySink = &IID_ID2D1SimplifiedGeometrySink_Value;
pub const ID2D1SimplifiedGeometrySink = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        SetFillMode: *const fn (
            self: *const ID2D1SimplifiedGeometrySink,
            fill_mode: D2D1_FILL_MODE,
        ) callconv(@import("std").os.windows.WINAPI) void,
        SetSegmentFlags: *const fn (
            self: *const ID2D1SimplifiedGeometrySink,
            vertex_flags: D2D1_PATH_SEGMENT,
        ) callconv(@import("std").os.windows.WINAPI) void,
        BeginFigure: *const fn (
            self: *const ID2D1SimplifiedGeometrySink,
            start_point: D2D_POINT_2F,
            figure_begin: D2D1_FIGURE_BEGIN,
        ) callconv(@import("std").os.windows.WINAPI) void,
        AddLines: *const fn (
            self: *const ID2D1SimplifiedGeometrySink,
            points: [*]const D2D_POINT_2F,
            points_count: u32,
        ) callconv(@import("std").os.windows.WINAPI) void,
        AddBeziers: *const fn (
            self: *const ID2D1SimplifiedGeometrySink,
            beziers: [*]const D2D1_BEZIER_SEGMENT,
            beziers_count: u32,
        ) callconv(@import("std").os.windows.WINAPI) void,
        EndFigure: *const fn (
            self: *const ID2D1SimplifiedGeometrySink,
            figure_end: D2D1_FIGURE_END,
        ) callconv(@import("std").os.windows.WINAPI) void,
        Close: *const fn (
            self: *const ID2D1SimplifiedGeometrySink,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn setFillMode(self: *const T, fill_mode_: D2D1_FILL_MODE) void {
                return @as(*const ID2D1SimplifiedGeometrySink.VTable, @ptrCast(self.vtable)).SetFillMode(@as(*const ID2D1SimplifiedGeometrySink, @ptrCast(self)), fill_mode_);
            }
            pub inline fn setSegmentFlags(self: *const T, vertex_flags_: D2D1_PATH_SEGMENT) void {
                return @as(*const ID2D1SimplifiedGeometrySink.VTable, @ptrCast(self.vtable)).SetSegmentFlags(@as(*const ID2D1SimplifiedGeometrySink, @ptrCast(self)), vertex_flags_);
            }
            pub inline fn beginFigure(self: *const T, start_point_: D2D_POINT_2F, figure_begin_: D2D1_FIGURE_BEGIN) void {
                return @as(*const ID2D1SimplifiedGeometrySink.VTable, @ptrCast(self.vtable)).BeginFigure(@as(*const ID2D1SimplifiedGeometrySink, @ptrCast(self)), start_point_, figure_begin_);
            }
            pub inline fn addLines(self: *const T, points_: [*]const D2D_POINT_2F, points_count_: u32) void {
                return @as(*const ID2D1SimplifiedGeometrySink.VTable, @ptrCast(self.vtable)).AddLines(@as(*const ID2D1SimplifiedGeometrySink, @ptrCast(self)), points_, points_count_);
            }
            pub inline fn addBeziers(self: *const T, beziers_: [*]const D2D1_BEZIER_SEGMENT, beziers_count_: u32) void {
                return @as(*const ID2D1SimplifiedGeometrySink.VTable, @ptrCast(self.vtable)).AddBeziers(@as(*const ID2D1SimplifiedGeometrySink, @ptrCast(self)), beziers_, beziers_count_);
            }
            pub inline fn endFigure(self: *const T, figure_end_: D2D1_FIGURE_END) void {
                return @as(*const ID2D1SimplifiedGeometrySink.VTable, @ptrCast(self.vtable)).EndFigure(@as(*const ID2D1SimplifiedGeometrySink, @ptrCast(self)), figure_end_);
            }
            pub inline fn close(self: *const T) HRESULT {
                return @as(*const ID2D1SimplifiedGeometrySink.VTable, @ptrCast(self.vtable)).Close(@as(*const ID2D1SimplifiedGeometrySink, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

pub const D2D1_BORDER_MODE = enum(u32) {
    SOFT = 0,
    HARD = 1,
    FORCE_DWORD = 4294967295,
};
pub const D2D1_BORDER_MODE_SOFT = D2D1_BORDER_MODE.SOFT;
pub const D2D1_BORDER_MODE_HARD = D2D1_BORDER_MODE.HARD;
pub const D2D1_BORDER_MODE_FORCE_DWORD = D2D1_BORDER_MODE.FORCE_DWORD;

pub const D2D1_BLEND_MODE = enum(u32) {
    MULTIPLY = 0,
    SCREEN = 1,
    DARKEN = 2,
    LIGHTEN = 3,
    DISSOLVE = 4,
    COLOR_BURN = 5,
    LINEAR_BURN = 6,
    DARKER_COLOR = 7,
    LIGHTER_COLOR = 8,
    COLOR_DODGE = 9,
    LINEAR_DODGE = 10,
    OVERLAY = 11,
    SOFT_LIGHT = 12,
    HARD_LIGHT = 13,
    VIVID_LIGHT = 14,
    LINEAR_LIGHT = 15,
    PIN_LIGHT = 16,
    HARD_MIX = 17,
    DIFFERENCE = 18,
    EXCLUSION = 19,
    HUE = 20,
    SATURATION = 21,
    COLOR = 22,
    LUMINOSITY = 23,
    SUBTRACT = 24,
    DIVISION = 25,
    FORCE_DWORD = 4294967295,
};
pub const D2D1_BLEND_MODE_MULTIPLY = D2D1_BLEND_MODE.MULTIPLY;
pub const D2D1_BLEND_MODE_SCREEN = D2D1_BLEND_MODE.SCREEN;
pub const D2D1_BLEND_MODE_DARKEN = D2D1_BLEND_MODE.DARKEN;
pub const D2D1_BLEND_MODE_LIGHTEN = D2D1_BLEND_MODE.LIGHTEN;
pub const D2D1_BLEND_MODE_DISSOLVE = D2D1_BLEND_MODE.DISSOLVE;
pub const D2D1_BLEND_MODE_COLOR_BURN = D2D1_BLEND_MODE.COLOR_BURN;
pub const D2D1_BLEND_MODE_LINEAR_BURN = D2D1_BLEND_MODE.LINEAR_BURN;
pub const D2D1_BLEND_MODE_DARKER_COLOR = D2D1_BLEND_MODE.DARKER_COLOR;
pub const D2D1_BLEND_MODE_LIGHTER_COLOR = D2D1_BLEND_MODE.LIGHTER_COLOR;
pub const D2D1_BLEND_MODE_COLOR_DODGE = D2D1_BLEND_MODE.COLOR_DODGE;
pub const D2D1_BLEND_MODE_LINEAR_DODGE = D2D1_BLEND_MODE.LINEAR_DODGE;
pub const D2D1_BLEND_MODE_OVERLAY = D2D1_BLEND_MODE.OVERLAY;
pub const D2D1_BLEND_MODE_SOFT_LIGHT = D2D1_BLEND_MODE.SOFT_LIGHT;
pub const D2D1_BLEND_MODE_HARD_LIGHT = D2D1_BLEND_MODE.HARD_LIGHT;
pub const D2D1_BLEND_MODE_VIVID_LIGHT = D2D1_BLEND_MODE.VIVID_LIGHT;
pub const D2D1_BLEND_MODE_LINEAR_LIGHT = D2D1_BLEND_MODE.LINEAR_LIGHT;
pub const D2D1_BLEND_MODE_PIN_LIGHT = D2D1_BLEND_MODE.PIN_LIGHT;
pub const D2D1_BLEND_MODE_HARD_MIX = D2D1_BLEND_MODE.HARD_MIX;
pub const D2D1_BLEND_MODE_DIFFERENCE = D2D1_BLEND_MODE.DIFFERENCE;
pub const D2D1_BLEND_MODE_EXCLUSION = D2D1_BLEND_MODE.EXCLUSION;
pub const D2D1_BLEND_MODE_HUE = D2D1_BLEND_MODE.HUE;
pub const D2D1_BLEND_MODE_SATURATION = D2D1_BLEND_MODE.SATURATION;
pub const D2D1_BLEND_MODE_COLOR = D2D1_BLEND_MODE.COLOR;
pub const D2D1_BLEND_MODE_LUMINOSITY = D2D1_BLEND_MODE.LUMINOSITY;
pub const D2D1_BLEND_MODE_SUBTRACT = D2D1_BLEND_MODE.SUBTRACT;
pub const D2D1_BLEND_MODE_DIVISION = D2D1_BLEND_MODE.DIVISION;
pub const D2D1_BLEND_MODE_FORCE_DWORD = D2D1_BLEND_MODE.FORCE_DWORD;

pub const D2D1_COLORMATRIX_ALPHA_MODE = enum(u32) {
    PREMULTIPLIED = 1,
    STRAIGHT = 2,
    FORCE_DWORD = 4294967295,
};
pub const D2D1_COLORMATRIX_ALPHA_MODE_PREMULTIPLIED = D2D1_COLORMATRIX_ALPHA_MODE.PREMULTIPLIED;
pub const D2D1_COLORMATRIX_ALPHA_MODE_STRAIGHT = D2D1_COLORMATRIX_ALPHA_MODE.STRAIGHT;
pub const D2D1_COLORMATRIX_ALPHA_MODE_FORCE_DWORD = D2D1_COLORMATRIX_ALPHA_MODE.FORCE_DWORD;

pub const D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE = enum(u32) {
    NEAREST_NEIGHBOR = 0,
    LINEAR = 1,
    CUBIC = 2,
    MULTI_SAMPLE_LINEAR = 3,
    ANISOTROPIC = 4,
    HIGH_QUALITY_CUBIC = 5,
    FORCE_DWORD = 4294967295,
};
pub const D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE_NEAREST_NEIGHBOR = D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE.NEAREST_NEIGHBOR;
pub const D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE_LINEAR = D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE.LINEAR;
pub const D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE_CUBIC = D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE.CUBIC;
pub const D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE_MULTI_SAMPLE_LINEAR = D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE.MULTI_SAMPLE_LINEAR;
pub const D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE_ANISOTROPIC = D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE.ANISOTROPIC;
pub const D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE_HIGH_QUALITY_CUBIC = D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE.HIGH_QUALITY_CUBIC;
pub const D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE_FORCE_DWORD = D2D1_2DAFFINETRANSFORM_INTERPOLATION_MODE.FORCE_DWORD;

pub const D2D1_TURBULENCE_NOISE = enum(u32) {
    FRACTAL_SUM = 0,
    TURBULENCE = 1,
    FORCE_DWORD = 4294967295,
};
pub const D2D1_TURBULENCE_NOISE_FRACTAL_SUM = D2D1_TURBULENCE_NOISE.FRACTAL_SUM;
pub const D2D1_TURBULENCE_NOISE_TURBULENCE = D2D1_TURBULENCE_NOISE.TURBULENCE;
pub const D2D1_TURBULENCE_NOISE_FORCE_DWORD = D2D1_TURBULENCE_NOISE.FORCE_DWORD;

pub const D2D1_COMPOSITE_MODE = enum(u32) {
    SOURCE_OVER = 0,
    DESTINATION_OVER = 1,
    SOURCE_IN = 2,
    DESTINATION_IN = 3,
    SOURCE_OUT = 4,
    DESTINATION_OUT = 5,
    SOURCE_ATOP = 6,
    DESTINATION_ATOP = 7,
    XOR = 8,
    PLUS = 9,
    SOURCE_COPY = 10,
    BOUNDED_SOURCE_COPY = 11,
    MASK_INVERT = 12,
    FORCE_DWORD = 4294967295,
};
pub const D2D1_COMPOSITE_MODE_SOURCE_OVER = D2D1_COMPOSITE_MODE.SOURCE_OVER;
pub const D2D1_COMPOSITE_MODE_DESTINATION_OVER = D2D1_COMPOSITE_MODE.DESTINATION_OVER;
pub const D2D1_COMPOSITE_MODE_SOURCE_IN = D2D1_COMPOSITE_MODE.SOURCE_IN;
pub const D2D1_COMPOSITE_MODE_DESTINATION_IN = D2D1_COMPOSITE_MODE.DESTINATION_IN;
pub const D2D1_COMPOSITE_MODE_SOURCE_OUT = D2D1_COMPOSITE_MODE.SOURCE_OUT;
pub const D2D1_COMPOSITE_MODE_DESTINATION_OUT = D2D1_COMPOSITE_MODE.DESTINATION_OUT;
pub const D2D1_COMPOSITE_MODE_SOURCE_ATOP = D2D1_COMPOSITE_MODE.SOURCE_ATOP;
pub const D2D1_COMPOSITE_MODE_DESTINATION_ATOP = D2D1_COMPOSITE_MODE.DESTINATION_ATOP;
pub const D2D1_COMPOSITE_MODE_XOR = D2D1_COMPOSITE_MODE.XOR;
pub const D2D1_COMPOSITE_MODE_PLUS = D2D1_COMPOSITE_MODE.PLUS;
pub const D2D1_COMPOSITE_MODE_SOURCE_COPY = D2D1_COMPOSITE_MODE.SOURCE_COPY;
pub const D2D1_COMPOSITE_MODE_BOUNDED_SOURCE_COPY = D2D1_COMPOSITE_MODE.BOUNDED_SOURCE_COPY;
pub const D2D1_COMPOSITE_MODE_MASK_INVERT = D2D1_COMPOSITE_MODE.MASK_INVERT;
pub const D2D1_COMPOSITE_MODE_FORCE_DWORD = D2D1_COMPOSITE_MODE.FORCE_DWORD;

//--------------------------------------------------------------------------------
// Section: Functions (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Unicode Aliases (0)
//--------------------------------------------------------------------------------
const thismodule = @This();
pub usingnamespace switch (@import("../../zig.zig").unicode_mode) {
    .ansi => struct {},
    .wide => struct {},
    .unspecified => if (@import("builtin").is_test) struct {} else struct {},
};
//--------------------------------------------------------------------------------
// Section: Imports (4)
//--------------------------------------------------------------------------------
const Guid = @import("../../zig.zig").Guid;
const DXGI_FORMAT = @import("../../graphics/dxgi/common.zig").DXGI_FORMAT;
const HRESULT = @import("../../foundation.zig").HRESULT;
const IUnknown = @import("../../system/com.zig").IUnknown;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
