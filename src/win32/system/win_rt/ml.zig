//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (5)
//--------------------------------------------------------------------------------
const IID_ILearningModelOperatorProviderNative_Value = Guid.initString("1adaa23a-eb67-41f3-aad8-5d984e9bacd4");
pub const IID_ILearningModelOperatorProviderNative = &IID_ILearningModelOperatorProviderNative_Value;
pub const ILearningModelOperatorProviderNative = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        GetRegistry: *const fn (
            self: *const ILearningModelOperatorProviderNative,
            pp_operator_registry: ?*?*IMLOperatorRegistry,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getRegistry(self: *const T, pp_operator_registry_: ?*?*IMLOperatorRegistry) HRESULT {
                return @as(*const ILearningModelOperatorProviderNative.VTable, @ptrCast(self.vtable)).GetRegistry(@as(*const ILearningModelOperatorProviderNative, @ptrCast(self)), pp_operator_registry_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ITensorNative_Value = Guid.initString("52f547ef-5b03-49b5-82d6-565f1ee0dd49");
pub const IID_ITensorNative = &IID_ITensorNative_Value;
pub const ITensorNative = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        GetBuffer: *const fn (
            self: *const ITensorNative,
            value: [*]?*u8,
            capacity: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetD3D12Resource: *const fn (
            self: *const ITensorNative,
            result: ?*?*ID3D12Resource,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getBuffer(self: *const T, value_: [*]?*u8, capacity_: ?*u32) HRESULT {
                return @as(*const ITensorNative.VTable, @ptrCast(self.vtable)).GetBuffer(@as(*const ITensorNative, @ptrCast(self)), value_, capacity_);
            }
            pub inline fn getD3D12Resource(self: *const T, result_: ?*?*ID3D12Resource) HRESULT {
                return @as(*const ITensorNative.VTable, @ptrCast(self.vtable)).GetD3D12Resource(@as(*const ITensorNative, @ptrCast(self)), result_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ITensorStaticsNative_Value = Guid.initString("39d055a4-66f6-4ebc-95d9-7a29ebe7690a");
pub const IID_ITensorStaticsNative = &IID_ITensorStaticsNative_Value;
pub const ITensorStaticsNative = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        CreateFromD3D12Resource: *const fn (
            self: *const ITensorStaticsNative,
            value: ?*ID3D12Resource,
            shape: ?*i64,
            shape_count: i32,
            result: ?*?*IUnknown,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn createFromD3D12Resource(self: *const T, value_: ?*ID3D12Resource, shape_: ?*i64, shape_count_: i32, result_: ?*?*IUnknown) HRESULT {
                return @as(*const ITensorStaticsNative.VTable, @ptrCast(self.vtable)).CreateFromD3D12Resource(@as(*const ITensorStaticsNative, @ptrCast(self)), value_, shape_, shape_count_, result_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ILearningModelDeviceFactoryNative_Value = Guid.initString("1e9b31a1-662e-4ae0-af67-f63bb337e634");
pub const IID_ILearningModelDeviceFactoryNative = &IID_ILearningModelDeviceFactoryNative_Value;
pub const ILearningModelDeviceFactoryNative = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        CreateFromD3D12CommandQueue: *const fn (
            self: *const ILearningModelDeviceFactoryNative,
            value: ?*ID3D12CommandQueue,
            result: ?*?*IUnknown,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn createFromD3D12CommandQueue(self: *const T, value_: ?*ID3D12CommandQueue, result_: ?*?*IUnknown) HRESULT {
                return @as(*const ILearningModelDeviceFactoryNative.VTable, @ptrCast(self.vtable)).CreateFromD3D12CommandQueue(@as(*const ILearningModelDeviceFactoryNative, @ptrCast(self)), value_, result_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ILearningModelSessionOptionsNative_Value = Guid.initString("c71e953f-37b4-4564-8658-d8396866db0d");
pub const IID_ILearningModelSessionOptionsNative = &IID_ILearningModelSessionOptionsNative_Value;
pub const ILearningModelSessionOptionsNative = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        SetIntraOpNumThreadsOverride: *const fn (
            self: *const ILearningModelSessionOptionsNative,
            intra_op_num_threads: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn setIntraOpNumThreadsOverride(self: *const T, intra_op_num_threads_: u32) HRESULT {
                return @as(*const ILearningModelSessionOptionsNative.VTable, @ptrCast(self.vtable)).SetIntraOpNumThreadsOverride(@as(*const ILearningModelSessionOptionsNative, @ptrCast(self)), intra_op_num_threads_);
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
// Section: Imports (6)
//--------------------------------------------------------------------------------
const Guid = @import("../../zig.zig").Guid;
const HRESULT = @import("../../foundation.zig").HRESULT;
const ID3D12CommandQueue = @import("../../graphics/direct3d12.zig").ID3D12CommandQueue;
const ID3D12Resource = @import("../../graphics/direct3d12.zig").ID3D12Resource;
const IMLOperatorRegistry = @import("../../ai/machine_learning/win_ml.zig").IMLOperatorRegistry;
const IUnknown = @import("../../system/com.zig").IUnknown;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
