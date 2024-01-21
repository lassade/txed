//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (1)
//--------------------------------------------------------------------------------
pub const MAX_D3D9ON12_QUEUES = @as(u32, 2);

//--------------------------------------------------------------------------------
// Section: Types (4)
//--------------------------------------------------------------------------------
pub const D3D9ON12_ARGS = extern struct {
    Enable9On12: BOOL,
    pD3D12Device: ?*IUnknown,
    ppD3D12Queues: [2]?*IUnknown,
    NumQueues: u32,
    NodeMask: u32,
};

pub const PFN_Direct3DCreate9On12Ex = *const fn (
    s_d_k_version: u32,
    p_override_list: ?*D3D9ON12_ARGS,
    num_override_entries: u32,
    pp_output_interface: ?*?*IDirect3D9Ex,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub const PFN_Direct3DCreate9On12 = *const fn (
    s_d_k_version: u32,
    p_override_list: ?*D3D9ON12_ARGS,
    num_override_entries: u32,
    retval: *?*IDirect3D9,
) callconv(@import("std").os.windows.WINAPI) void;

const IID_IDirect3DDevice9On12_Value = Guid.initString("e7fda234-b589-4049-940d-8878977531c8");
pub const IID_IDirect3DDevice9On12 = &IID_IDirect3DDevice9On12_Value;
pub const IDirect3DDevice9On12 = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        GetD3D12Device: *const fn (
            self: *const IDirect3DDevice9On12,
            riid: ?*const Guid,
            ppv_device: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        UnwrapUnderlyingResource: *const fn (
            self: *const IDirect3DDevice9On12,
            p_resource: ?*IDirect3DResource9,
            p_command_queue: ?*ID3D12CommandQueue,
            riid: ?*const Guid,
            ppv_resource12: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ReturnUnderlyingResource: *const fn (
            self: *const IDirect3DDevice9On12,
            p_resource: ?*IDirect3DResource9,
            num_sync: u32,
            p_signal_values: ?*u64,
            pp_fences: ?*?*ID3D12Fence,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getD3D12Device(self: *const T, riid_: ?*const Guid, ppv_device_: ?*?*anyopaque) HRESULT {
                return @as(*const IDirect3DDevice9On12.VTable, @ptrCast(self.vtable)).GetD3D12Device(@as(*const IDirect3DDevice9On12, @ptrCast(self)), riid_, ppv_device_);
            }
            pub inline fn unwrapUnderlyingResource(self: *const T, p_resource_: ?*IDirect3DResource9, p_command_queue_: ?*ID3D12CommandQueue, riid_: ?*const Guid, ppv_resource12_: ?*?*anyopaque) HRESULT {
                return @as(*const IDirect3DDevice9On12.VTable, @ptrCast(self.vtable)).UnwrapUnderlyingResource(@as(*const IDirect3DDevice9On12, @ptrCast(self)), p_resource_, p_command_queue_, riid_, ppv_resource12_);
            }
            pub inline fn returnUnderlyingResource(self: *const T, p_resource_: ?*IDirect3DResource9, num_sync_: u32, p_signal_values_: ?*u64, pp_fences_: ?*?*ID3D12Fence) HRESULT {
                return @as(*const IDirect3DDevice9On12.VTable, @ptrCast(self.vtable)).ReturnUnderlyingResource(@as(*const IDirect3DDevice9On12, @ptrCast(self)), p_resource_, num_sync_, p_signal_values_, pp_fences_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

//--------------------------------------------------------------------------------
// Section: Functions (2)
//--------------------------------------------------------------------------------
pub extern "d3d9" fn Direct3DCreate9On12Ex(
    s_d_k_version: u32,
    p_override_list: ?*D3D9ON12_ARGS,
    num_override_entries: u32,
    pp_output_interface: ?*?*IDirect3D9Ex,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "d3d9" fn Direct3DCreate9On12(
    s_d_k_version: u32,
    p_override_list: ?*D3D9ON12_ARGS,
    num_override_entries: u32,
    retval: *?*IDirect3D9,
) callconv(@import("std").os.windows.WINAPI) void;

//--------------------------------------------------------------------------------
// Section: Unicode Aliases (0)
//--------------------------------------------------------------------------------
const thismodule = @This();
pub usingnamespace switch (@import("../zig.zig").unicode_mode) {
    .ansi => struct {},
    .wide => struct {},
    .unspecified => if (@import("builtin").is_test) struct {} else struct {},
};
//--------------------------------------------------------------------------------
// Section: Imports (9)
//--------------------------------------------------------------------------------
const Guid = @import("../zig.zig").Guid;
const BOOL = @import("../foundation.zig").BOOL;
const HRESULT = @import("../foundation.zig").HRESULT;
const ID3D12CommandQueue = @import("../graphics/direct3d12.zig").ID3D12CommandQueue;
const ID3D12Fence = @import("../graphics/direct3d12.zig").ID3D12Fence;
const IDirect3D9 = @import("../graphics/direct3d9.zig").IDirect3D9;
const IDirect3D9Ex = @import("../graphics/direct3d9.zig").IDirect3D9Ex;
const IDirect3DResource9 = @import("../graphics/direct3d9.zig").IDirect3DResource9;
const IUnknown = @import("../system/com.zig").IUnknown;

test {
    // The following '_ = <FuncPtrType>' lines are a workaround for https://github.com/ziglang/zig/issues/4476
    if (@hasDecl(@This(), "PFN_Direct3DCreate9On12Ex")) {
        _ = PFN_Direct3DCreate9On12Ex;
    }
    if (@hasDecl(@This(), "PFN_Direct3DCreate9On12")) {
        _ = PFN_Direct3DCreate9On12;
    }

    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}