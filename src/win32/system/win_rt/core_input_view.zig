//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (1)
//--------------------------------------------------------------------------------
const IID_ICoreFrameworkInputViewInterop_Value = Guid.initString("0e3da342-b11c-484b-9c1c-be0d61c2f6c5");
pub const IID_ICoreFrameworkInputViewInterop = &IID_ICoreFrameworkInputViewInterop_Value;
pub const ICoreFrameworkInputViewInterop = extern struct {
    pub const VTable = extern struct {
        base: IInspectable.VTable,
        GetForWindow: *const fn (
            self: *const ICoreFrameworkInputViewInterop,
            app_window: ?HWND,
            riid: ?*const Guid,
            core_framework_input_view: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IInspectable.MethodMixin(T);
            pub inline fn getForWindow(self: *const T, app_window_: ?HWND, riid_: ?*const Guid, core_framework_input_view_: ?*?*anyopaque) HRESULT {
                return @as(*const ICoreFrameworkInputViewInterop.VTable, @ptrCast(self.vtable)).GetForWindow(@as(*const ICoreFrameworkInputViewInterop, @ptrCast(self)), app_window_, riid_, core_framework_input_view_);
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
// Section: Imports (4)
//--------------------------------------------------------------------------------
const Guid = @import("../../zig.zig").Guid;
const HRESULT = @import("../../foundation.zig").HRESULT;
const HWND = @import("../../foundation.zig").HWND;
const IInspectable = @import("../../system/win_rt.zig").IInspectable;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
