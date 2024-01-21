//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (1)
//--------------------------------------------------------------------------------
pub const CLSID_SoftwareBitmapNativeFactory = Guid.initString("84e65691-8602-4a84-be46-708be9cd4b74");

//--------------------------------------------------------------------------------
// Section: Types (2)
//--------------------------------------------------------------------------------
const IID_ISoftwareBitmapNative_Value = Guid.initString("94bc8415-04ea-4b2e-af13-4de95aa898eb");
pub const IID_ISoftwareBitmapNative = &IID_ISoftwareBitmapNative_Value;
pub const ISoftwareBitmapNative = extern struct {
    pub const VTable = extern struct {
        base: IInspectable.VTable,
        GetData: *const fn (
            self: *const ISoftwareBitmapNative,
            riid: ?*const Guid,
            ppv: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IInspectable.MethodMixin(T);
            pub inline fn getData(self: *const T, riid_: ?*const Guid, ppv_: ?*?*anyopaque) HRESULT {
                return @as(*const ISoftwareBitmapNative.VTable, @ptrCast(self.vtable)).GetData(@as(*const ISoftwareBitmapNative, @ptrCast(self)), riid_, ppv_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISoftwareBitmapNativeFactory_Value = Guid.initString("c3c181ec-2914-4791-af02-02d224a10b43");
pub const IID_ISoftwareBitmapNativeFactory = &IID_ISoftwareBitmapNativeFactory_Value;
pub const ISoftwareBitmapNativeFactory = extern struct {
    pub const VTable = extern struct {
        base: IInspectable.VTable,
        CreateFromWICBitmap: *const fn (
            self: *const ISoftwareBitmapNativeFactory,
            data: ?*IWICBitmap,
            force_read_only: BOOL,
            riid: ?*const Guid,
            ppv: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        CreateFromMF2DBuffer2: *const fn (
            self: *const ISoftwareBitmapNativeFactory,
            data: ?*IMF2DBuffer2,
            subtype: ?*const Guid,
            width: u32,
            height: u32,
            force_read_only: BOOL,
            min_display_aperture: ?*const MFVideoArea,
            riid: ?*const Guid,
            ppv: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IInspectable.MethodMixin(T);
            pub inline fn createFromWICBitmap(self: *const T, data_: ?*IWICBitmap, force_read_only_: BOOL, riid_: ?*const Guid, ppv_: ?*?*anyopaque) HRESULT {
                return @as(*const ISoftwareBitmapNativeFactory.VTable, @ptrCast(self.vtable)).CreateFromWICBitmap(@as(*const ISoftwareBitmapNativeFactory, @ptrCast(self)), data_, force_read_only_, riid_, ppv_);
            }
            pub inline fn createFromMF2DBuffer2(self: *const T, data_: ?*IMF2DBuffer2, subtype_: ?*const Guid, width_: u32, height_: u32, force_read_only_: BOOL, min_display_aperture_: ?*const MFVideoArea, riid_: ?*const Guid, ppv_: ?*?*anyopaque) HRESULT {
                return @as(*const ISoftwareBitmapNativeFactory.VTable, @ptrCast(self.vtable)).CreateFromMF2DBuffer2(@as(*const ISoftwareBitmapNativeFactory, @ptrCast(self)), data_, subtype_, width_, height_, force_read_only_, min_display_aperture_, riid_, ppv_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

//--------------------------------------------------------------------------------
// Section: Functions (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Unicode Aliases (0)
//--------------------------------------------------------------------------------
const thismodule = @This();
pub usingnamespace switch (@import("../../../zig.zig").unicode_mode) {
    .ansi => struct {},
    .wide => struct {},
    .unspecified => if (@import("builtin").is_test) struct {} else struct {},
};
//--------------------------------------------------------------------------------
// Section: Imports (7)
//--------------------------------------------------------------------------------
const Guid = @import("../../../zig.zig").Guid;
const BOOL = @import("../../../foundation.zig").BOOL;
const HRESULT = @import("../../../foundation.zig").HRESULT;
const IInspectable = @import("../../../system/win_rt.zig").IInspectable;
const IMF2DBuffer2 = @import("../../../media/media_foundation.zig").IMF2DBuffer2;
const IWICBitmap = @import("../../../graphics/imaging.zig").IWICBitmap;
const MFVideoArea = @import("../../../media/media_foundation.zig").MFVideoArea;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}