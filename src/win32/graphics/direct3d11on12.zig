//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (5)
//--------------------------------------------------------------------------------
pub const PFN_D3D11ON12_CREATE_DEVICE = *const fn (
    param0: ?*IUnknown,
    param1: u32,
    param2: ?[*]const D3D_FEATURE_LEVEL,
    feature_levels: u32,
    param4: ?[*]?*IUnknown,
    num_queues: u32,
    param6: u32,
    param7: ?*?*ID3D11Device,
    param8: ?*?*ID3D11DeviceContext,
    param9: ?*D3D_FEATURE_LEVEL,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub const D3D11_RESOURCE_FLAGS = extern struct {
    BindFlags: u32,
    MiscFlags: u32,
    CPUAccessFlags: u32,
    StructureByteStride: u32,
};

// This COM type is Agile, not sure what that means
const IID_ID3D11On12Device_Value = Guid.initString("85611e73-70a9-490e-9614-a9e302777904");
pub const IID_ID3D11On12Device = &IID_ID3D11On12Device_Value;
pub const ID3D11On12Device = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        CreateWrappedResource: *const fn (
            self: *const ID3D11On12Device,
            p_resource12: ?*IUnknown,
            p_flags11: ?*const D3D11_RESOURCE_FLAGS,
            in_state: D3D12_RESOURCE_STATES,
            out_state: D3D12_RESOURCE_STATES,
            riid: ?*const Guid,
            pp_resource11: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ReleaseWrappedResources: *const fn (
            self: *const ID3D11On12Device,
            pp_resources: [*]?*ID3D11Resource,
            num_resources: u32,
        ) callconv(@import("std").os.windows.WINAPI) void,
        AcquireWrappedResources: *const fn (
            self: *const ID3D11On12Device,
            pp_resources: [*]?*ID3D11Resource,
            num_resources: u32,
        ) callconv(@import("std").os.windows.WINAPI) void,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn createWrappedResource(self: *const T, p_resource12_: ?*IUnknown, p_flags11_: ?*const D3D11_RESOURCE_FLAGS, in_state_: D3D12_RESOURCE_STATES, out_state_: D3D12_RESOURCE_STATES, riid_: ?*const Guid, pp_resource11_: ?*?*anyopaque) HRESULT {
                return @as(*const ID3D11On12Device.VTable, @ptrCast(self.vtable)).CreateWrappedResource(@as(*const ID3D11On12Device, @ptrCast(self)), p_resource12_, p_flags11_, in_state_, out_state_, riid_, pp_resource11_);
            }
            pub inline fn releaseWrappedResources(self: *const T, pp_resources_: [*]?*ID3D11Resource, num_resources_: u32) void {
                return @as(*const ID3D11On12Device.VTable, @ptrCast(self.vtable)).ReleaseWrappedResources(@as(*const ID3D11On12Device, @ptrCast(self)), pp_resources_, num_resources_);
            }
            pub inline fn acquireWrappedResources(self: *const T, pp_resources_: [*]?*ID3D11Resource, num_resources_: u32) void {
                return @as(*const ID3D11On12Device.VTable, @ptrCast(self.vtable)).AcquireWrappedResources(@as(*const ID3D11On12Device, @ptrCast(self)), pp_resources_, num_resources_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows10.0.18362'
// This COM type is Agile, not sure what that means
const IID_ID3D11On12Device1_Value = Guid.initString("bdb64df4-ea2f-4c70-b861-aaab1258bb5d");
pub const IID_ID3D11On12Device1 = &IID_ID3D11On12Device1_Value;
pub const ID3D11On12Device1 = extern struct {
    pub const VTable = extern struct {
        base: ID3D11On12Device.VTable,
        GetD3D12Device: *const fn (
            self: *const ID3D11On12Device1,
            riid: ?*const Guid,
            ppv_device: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace ID3D11On12Device.MethodMixin(T);
            pub inline fn getD3D12Device(self: *const T, riid_: ?*const Guid, ppv_device_: ?*?*anyopaque) HRESULT {
                return @as(*const ID3D11On12Device1.VTable, @ptrCast(self.vtable)).GetD3D12Device(@as(*const ID3D11On12Device1, @ptrCast(self)), riid_, ppv_device_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows10.0.19041'
// This COM type is Agile, not sure what that means
const IID_ID3D11On12Device2_Value = Guid.initString("dc90f331-4740-43fa-866e-67f12cb58223");
pub const IID_ID3D11On12Device2 = &IID_ID3D11On12Device2_Value;
pub const ID3D11On12Device2 = extern struct {
    pub const VTable = extern struct {
        base: ID3D11On12Device1.VTable,
        UnwrapUnderlyingResource: *const fn (
            self: *const ID3D11On12Device2,
            p_resource11: ?*ID3D11Resource,
            p_command_queue: ?*ID3D12CommandQueue,
            riid: ?*const Guid,
            ppv_resource12: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ReturnUnderlyingResource: *const fn (
            self: *const ID3D11On12Device2,
            p_resource11: ?*ID3D11Resource,
            num_sync: u32,
            p_signal_values: [*]u64,
            pp_fences: [*]?*ID3D12Fence,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace ID3D11On12Device1.MethodMixin(T);
            pub inline fn unwrapUnderlyingResource(self: *const T, p_resource11_: ?*ID3D11Resource, p_command_queue_: ?*ID3D12CommandQueue, riid_: ?*const Guid, ppv_resource12_: ?*?*anyopaque) HRESULT {
                return @as(*const ID3D11On12Device2.VTable, @ptrCast(self.vtable)).UnwrapUnderlyingResource(@as(*const ID3D11On12Device2, @ptrCast(self)), p_resource11_, p_command_queue_, riid_, ppv_resource12_);
            }
            pub inline fn returnUnderlyingResource(self: *const T, p_resource11_: ?*ID3D11Resource, num_sync_: u32, p_signal_values_: [*]u64, pp_fences_: [*]?*ID3D12Fence) HRESULT {
                return @as(*const ID3D11On12Device2.VTable, @ptrCast(self.vtable)).ReturnUnderlyingResource(@as(*const ID3D11On12Device2, @ptrCast(self)), p_resource11_, num_sync_, p_signal_values_, pp_fences_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

//--------------------------------------------------------------------------------
// Section: Functions (1)
//--------------------------------------------------------------------------------
pub extern "d3d11" fn D3D11On12CreateDevice(
    p_device: ?*IUnknown,
    flags: u32,
    p_feature_levels: ?[*]const D3D_FEATURE_LEVEL,
    feature_levels: u32,
    pp_command_queues: ?[*]?*IUnknown,
    num_queues: u32,
    node_mask: u32,
    pp_device: ?*?*ID3D11Device,
    pp_immediate_context: ?*?*ID3D11DeviceContext,
    p_chosen_feature_level: ?*D3D_FEATURE_LEVEL,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

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
// Section: Imports (10)
//--------------------------------------------------------------------------------
const Guid = @import("../zig.zig").Guid;
const D3D12_RESOURCE_STATES = @import("../graphics/direct3d12.zig").D3D12_RESOURCE_STATES;
const D3D_FEATURE_LEVEL = @import("../graphics/direct3d.zig").D3D_FEATURE_LEVEL;
const HRESULT = @import("../foundation.zig").HRESULT;
const ID3D11Device = @import("../graphics/direct3d11.zig").ID3D11Device;
const ID3D11DeviceContext = @import("../graphics/direct3d11.zig").ID3D11DeviceContext;
const ID3D11Resource = @import("../graphics/direct3d11.zig").ID3D11Resource;
const ID3D12CommandQueue = @import("../graphics/direct3d12.zig").ID3D12CommandQueue;
const ID3D12Fence = @import("../graphics/direct3d12.zig").ID3D12Fence;
const IUnknown = @import("../system/com.zig").IUnknown;

test {
    // The following '_ = <FuncPtrType>' lines are a workaround for https://github.com/ziglang/zig/issues/4476
    if (@hasDecl(@This(), "PFN_D3D11ON12_CREATE_DEVICE")) {
        _ = PFN_D3D11ON12_CREATE_DEVICE;
    }

    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
