//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (2)
//--------------------------------------------------------------------------------
pub const CLSID_AudioFrameNativeFactory = Guid.initString("16a0a3b9-9f65-4102-9367-2cda3a4f372a");
pub const CLSID_VideoFrameNativeFactory = Guid.initString("d194386a-04e3-4814-8100-b2b0ae6d78c7");

//--------------------------------------------------------------------------------
// Section: Types (4)
//--------------------------------------------------------------------------------
const IID_IAudioFrameNative_Value = Guid.initString("20be1e2e-930f-4746-9335-3c332f255093");
pub const IID_IAudioFrameNative = &IID_IAudioFrameNative_Value;
pub const IAudioFrameNative = extern struct {
    pub const VTable = extern struct {
        base: IInspectable.VTable,
        GetData: *const fn (
            self: *const IAudioFrameNative,
            riid: ?*const Guid,
            ppv: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IInspectable.MethodMixin(T);
            pub inline fn getData(self: *const T, riid_: ?*const Guid, ppv_: ?*?*anyopaque) HRESULT {
                return @as(*const IAudioFrameNative.VTable, @ptrCast(self.vtable)).GetData(@as(*const IAudioFrameNative, @ptrCast(self)), riid_, ppv_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IVideoFrameNative_Value = Guid.initString("26ba702b-314a-4620-aaf6-7a51aa58fa18");
pub const IID_IVideoFrameNative = &IID_IVideoFrameNative_Value;
pub const IVideoFrameNative = extern struct {
    pub const VTable = extern struct {
        base: IInspectable.VTable,
        GetData: *const fn (
            self: *const IVideoFrameNative,
            riid: ?*const Guid,
            ppv: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetDevice: *const fn (
            self: *const IVideoFrameNative,
            riid: ?*const Guid,
            ppv: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IInspectable.MethodMixin(T);
            pub inline fn getData(self: *const T, riid_: ?*const Guid, ppv_: ?*?*anyopaque) HRESULT {
                return @as(*const IVideoFrameNative.VTable, @ptrCast(self.vtable)).GetData(@as(*const IVideoFrameNative, @ptrCast(self)), riid_, ppv_);
            }
            pub inline fn getDevice(self: *const T, riid_: ?*const Guid, ppv_: ?*?*anyopaque) HRESULT {
                return @as(*const IVideoFrameNative.VTable, @ptrCast(self.vtable)).GetDevice(@as(*const IVideoFrameNative, @ptrCast(self)), riid_, ppv_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IAudioFrameNativeFactory_Value = Guid.initString("7bd67cf8-bf7d-43e6-af8d-b170ee0c0110");
pub const IID_IAudioFrameNativeFactory = &IID_IAudioFrameNativeFactory_Value;
pub const IAudioFrameNativeFactory = extern struct {
    pub const VTable = extern struct {
        base: IInspectable.VTable,
        CreateFromMFSample: *const fn (
            self: *const IAudioFrameNativeFactory,
            data: ?*IMFSample,
            force_read_only: BOOL,
            riid: ?*const Guid,
            ppv: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IInspectable.MethodMixin(T);
            pub inline fn createFromMFSample(self: *const T, data_: ?*IMFSample, force_read_only_: BOOL, riid_: ?*const Guid, ppv_: ?*?*anyopaque) HRESULT {
                return @as(*const IAudioFrameNativeFactory.VTable, @ptrCast(self.vtable)).CreateFromMFSample(@as(*const IAudioFrameNativeFactory, @ptrCast(self)), data_, force_read_only_, riid_, ppv_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IVideoFrameNativeFactory_Value = Guid.initString("69e3693e-8e1e-4e63-ac4c-7fdc21d9731d");
pub const IID_IVideoFrameNativeFactory = &IID_IVideoFrameNativeFactory_Value;
pub const IVideoFrameNativeFactory = extern struct {
    pub const VTable = extern struct {
        base: IInspectable.VTable,
        CreateFromMFSample: *const fn (
            self: *const IVideoFrameNativeFactory,
            data: ?*IMFSample,
            subtype: ?*const Guid,
            width: u32,
            height: u32,
            force_read_only: BOOL,
            min_display_aperture: ?*const MFVideoArea,
            device: ?*IMFDXGIDeviceManager,
            riid: ?*const Guid,
            ppv: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IInspectable.MethodMixin(T);
            pub inline fn createFromMFSample(self: *const T, data_: ?*IMFSample, subtype_: ?*const Guid, width_: u32, height_: u32, force_read_only_: BOOL, min_display_aperture_: ?*const MFVideoArea, device_: ?*IMFDXGIDeviceManager, riid_: ?*const Guid, ppv_: ?*?*anyopaque) HRESULT {
                return @as(*const IVideoFrameNativeFactory.VTable, @ptrCast(self.vtable)).CreateFromMFSample(@as(*const IVideoFrameNativeFactory, @ptrCast(self)), data_, subtype_, width_, height_, force_read_only_, min_display_aperture_, device_, riid_, ppv_);
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
pub usingnamespace switch (@import("../../zig.zig").unicode_mode) {
    .ansi => struct {},
    .wide => struct {},
    .unspecified => if (@import("builtin").is_test) struct {} else struct {},
};
//--------------------------------------------------------------------------------
// Section: Imports (7)
//--------------------------------------------------------------------------------
const Guid = @import("../../zig.zig").Guid;
const BOOL = @import("../../foundation.zig").BOOL;
const HRESULT = @import("../../foundation.zig").HRESULT;
const IInspectable = @import("../../system/win_rt.zig").IInspectable;
const IMFDXGIDeviceManager = @import("../../media/media_foundation.zig").IMFDXGIDeviceManager;
const IMFSample = @import("../../media/media_foundation.zig").IMFSample;
const MFVideoArea = @import("../../media/media_foundation.zig").MFVideoArea;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}