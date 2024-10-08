//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (30)
//--------------------------------------------------------------------------------
pub const NDF_ERROR_START = @as(u32, 63744);
pub const NDF_E_LENGTH_EXCEEDED = @import("../zig.zig").typedConst(HRESULT, @as(i32, -2146895616));
pub const NDF_E_NOHELPERCLASS = @import("../zig.zig").typedConst(HRESULT, @as(i32, -2146895615));
pub const NDF_E_CANCELLED = @import("../zig.zig").typedConst(HRESULT, @as(i32, -2146895614));
pub const NDF_E_DISABLED = @import("../zig.zig").typedConst(HRESULT, @as(i32, -2146895612));
pub const NDF_E_BAD_PARAM = @import("../zig.zig").typedConst(HRESULT, @as(i32, -2146895611));
pub const NDF_E_VALIDATION = @import("../zig.zig").typedConst(HRESULT, @as(i32, -2146895610));
pub const NDF_E_UNKNOWN = @import("../zig.zig").typedConst(HRESULT, @as(i32, -2146895609));
pub const NDF_E_PROBLEM_PRESENT = @import("../zig.zig").typedConst(HRESULT, @as(i32, -2146895608));
pub const RF_WORKAROUND = @as(u32, 536870912);
pub const RF_USER_ACTION = @as(u32, 268435456);
pub const RF_USER_CONFIRMATION = @as(u32, 134217728);
pub const RF_INFORMATION_ONLY = @as(u32, 33554432);
pub const RF_UI_ONLY = @as(u32, 16777216);
pub const RF_SHOW_EVENTS = @as(u32, 8388608);
pub const RF_VALIDATE_HELPTOPIC = @as(u32, 4194304);
pub const RF_REPRO = @as(u32, 2097152);
pub const RF_CONTACT_ADMIN = @as(u32, 131072);
pub const RF_RESERVED = @as(u32, 1073741824);
pub const RF_RESERVED_CA = @as(u32, 2147483648);
pub const RF_RESERVED_LNI = @as(u32, 65536);
pub const RCF_ISLEAF = @as(u32, 1);
pub const RCF_ISCONFIRMED = @as(u32, 2);
pub const RCF_ISTHIRDPARTY = @as(u32, 4);
pub const DF_IMPERSONATION = @as(u32, 2147483648);
pub const DF_TRACELESS = @as(u32, 1073741824);
pub const NDF_INBOUND_FLAG_EDGETRAVERSAL = @as(u32, 1);
pub const NDF_INBOUND_FLAG_HEALTHCHECK = @as(u32, 2);
pub const NDF_ADD_CAPTURE_TRACE = @as(u32, 1);
pub const NDF_APPLY_INCLUSION_LIST_FILTER = @as(u32, 2);

//--------------------------------------------------------------------------------
// Section: Types (25)
//--------------------------------------------------------------------------------
pub const ATTRIBUTE_TYPE = enum(i32) {
    INVALID = 0,
    BOOLEAN = 1,
    INT8 = 2,
    UINT8 = 3,
    INT16 = 4,
    UINT16 = 5,
    INT32 = 6,
    UINT32 = 7,
    INT64 = 8,
    UINT64 = 9,
    STRING = 10,
    GUID = 11,
    LIFE_TIME = 12,
    SOCKADDR = 13,
    OCTET_STRING = 14,
};
pub const AT_INVALID = ATTRIBUTE_TYPE.INVALID;
pub const AT_BOOLEAN = ATTRIBUTE_TYPE.BOOLEAN;
pub const AT_INT8 = ATTRIBUTE_TYPE.INT8;
pub const AT_UINT8 = ATTRIBUTE_TYPE.UINT8;
pub const AT_INT16 = ATTRIBUTE_TYPE.INT16;
pub const AT_UINT16 = ATTRIBUTE_TYPE.UINT16;
pub const AT_INT32 = ATTRIBUTE_TYPE.INT32;
pub const AT_UINT32 = ATTRIBUTE_TYPE.UINT32;
pub const AT_INT64 = ATTRIBUTE_TYPE.INT64;
pub const AT_UINT64 = ATTRIBUTE_TYPE.UINT64;
pub const AT_STRING = ATTRIBUTE_TYPE.STRING;
pub const AT_GUID = ATTRIBUTE_TYPE.GUID;
pub const AT_LIFE_TIME = ATTRIBUTE_TYPE.LIFE_TIME;
pub const AT_SOCKADDR = ATTRIBUTE_TYPE.SOCKADDR;
pub const AT_OCTET_STRING = ATTRIBUTE_TYPE.OCTET_STRING;

pub const OCTET_STRING = extern struct {
    dwLength: u32,
    lpValue: ?*u8,
};

pub const LIFE_TIME = extern struct {
    startTime: FILETIME,
    endTime: FILETIME,
};

pub const DIAG_SOCKADDR = extern struct {
    family: u16,
    data: [126]CHAR,
};

pub const HELPER_ATTRIBUTE = extern struct {
    pwszName: ?PWSTR,
    type: ATTRIBUTE_TYPE,
    Anonymous: extern union {
        Boolean: BOOL,
        Char: u8,
        Byte: u8,
        Short: i16,
        Word: u16,
        Int: i32,
        DWord: u32,
        Int64: i64,
        UInt64: u64,
        PWStr: ?PWSTR,
        Guid: Guid,
        LifeTime: LIFE_TIME,
        Address: DIAG_SOCKADDR,
        OctetString: OCTET_STRING,
    },
};

pub const REPAIR_SCOPE = enum(i32) {
    SYSTEM = 0,
    USER = 1,
    APPLICATION = 2,
    PROCESS = 3,
};
pub const RS_SYSTEM = REPAIR_SCOPE.SYSTEM;
pub const RS_USER = REPAIR_SCOPE.USER;
pub const RS_APPLICATION = REPAIR_SCOPE.APPLICATION;
pub const RS_PROCESS = REPAIR_SCOPE.PROCESS;

pub const REPAIR_RISK = enum(i32) {
    NOROLLBACK = 0,
    ROLLBACK = 1,
    NORISK = 2,
};
pub const RR_NOROLLBACK = REPAIR_RISK.NOROLLBACK;
pub const RR_ROLLBACK = REPAIR_RISK.ROLLBACK;
pub const RR_NORISK = REPAIR_RISK.NORISK;

pub const UI_INFO_TYPE = enum(i32) {
    INVALID = 0,
    NONE = 1,
    SHELL_COMMAND = 2,
    HELP_PANE = 3,
    DUI = 4,
};
pub const UIT_INVALID = UI_INFO_TYPE.INVALID;
pub const UIT_NONE = UI_INFO_TYPE.NONE;
pub const UIT_SHELL_COMMAND = UI_INFO_TYPE.SHELL_COMMAND;
pub const UIT_HELP_PANE = UI_INFO_TYPE.HELP_PANE;
pub const UIT_DUI = UI_INFO_TYPE.DUI;

pub const ShellCommandInfo = extern struct {
    pwszOperation: ?PWSTR,
    pwszFile: ?PWSTR,
    pwszParameters: ?PWSTR,
    pwszDirectory: ?PWSTR,
    nShowCmd: u32,
};

pub const UiInfo = extern struct {
    type: UI_INFO_TYPE,
    Anonymous: extern union {
        pwzNull: ?PWSTR,
        ShellInfo: ShellCommandInfo,
        pwzHelpUrl: ?PWSTR,
        pwzDui: ?PWSTR,
    },
};

pub const RepairInfo = extern struct {
    guid: Guid,
    pwszClassName: ?PWSTR,
    pwszDescription: ?PWSTR,
    sidType: u32,
    cost: i32,
    flags: u32,
    scope: REPAIR_SCOPE,
    risk: REPAIR_RISK,
    UiInfo: UiInfo,
    rootCauseIndex: i32,
};

pub const RepairInfoEx = extern struct {
    repair: RepairInfo,
    repairRank: u16,
};

pub const RootCauseInfo = extern struct {
    pwszDescription: ?PWSTR,
    rootCauseID: Guid,
    rootCauseFlags: u32,
    networkInterfaceID: Guid,
    pRepairs: ?*RepairInfoEx,
    repairCount: u16,
};

pub const DIAGNOSIS_STATUS = enum(i32) {
    NOT_IMPLEMENTED = 0,
    CONFIRMED = 1,
    REJECTED = 2,
    INDETERMINATE = 3,
    DEFERRED = 4,
    PASSTHROUGH = 5,
};
pub const DS_NOT_IMPLEMENTED = DIAGNOSIS_STATUS.NOT_IMPLEMENTED;
pub const DS_CONFIRMED = DIAGNOSIS_STATUS.CONFIRMED;
pub const DS_REJECTED = DIAGNOSIS_STATUS.REJECTED;
pub const DS_INDETERMINATE = DIAGNOSIS_STATUS.INDETERMINATE;
pub const DS_DEFERRED = DIAGNOSIS_STATUS.DEFERRED;
pub const DS_PASSTHROUGH = DIAGNOSIS_STATUS.PASSTHROUGH;

pub const REPAIR_STATUS = enum(i32) {
    NOT_IMPLEMENTED = 0,
    REPAIRED = 1,
    UNREPAIRED = 2,
    DEFERRED = 3,
    USER_ACTION = 4,
};
pub const RS_NOT_IMPLEMENTED = REPAIR_STATUS.NOT_IMPLEMENTED;
pub const RS_REPAIRED = REPAIR_STATUS.REPAIRED;
pub const RS_UNREPAIRED = REPAIR_STATUS.UNREPAIRED;
pub const RS_DEFERRED = REPAIR_STATUS.DEFERRED;
pub const RS_USER_ACTION = REPAIR_STATUS.USER_ACTION;

pub const PROBLEM_TYPE = enum(i32) {
    INVALID = 0,
    LOW_HEALTH = 1,
    LOWER_HEALTH = 2,
    DOWN_STREAM_HEALTH = 4,
    HIGH_UTILIZATION = 8,
    HIGHER_UTILIZATION = 16,
    UP_STREAM_UTILIZATION = 32,
};
pub const PT_INVALID = PROBLEM_TYPE.INVALID;
pub const PT_LOW_HEALTH = PROBLEM_TYPE.LOW_HEALTH;
pub const PT_LOWER_HEALTH = PROBLEM_TYPE.LOWER_HEALTH;
pub const PT_DOWN_STREAM_HEALTH = PROBLEM_TYPE.DOWN_STREAM_HEALTH;
pub const PT_HIGH_UTILIZATION = PROBLEM_TYPE.HIGH_UTILIZATION;
pub const PT_HIGHER_UTILIZATION = PROBLEM_TYPE.HIGHER_UTILIZATION;
pub const PT_UP_STREAM_UTILIZATION = PROBLEM_TYPE.UP_STREAM_UTILIZATION;

pub const HYPOTHESIS = extern struct {
    pwszClassName: ?PWSTR,
    pwszDescription: ?PWSTR,
    celt: u32,
    rgAttributes: ?*HELPER_ATTRIBUTE,
};

pub const HelperAttributeInfo = extern struct {
    pwszName: ?PWSTR,
    type: ATTRIBUTE_TYPE,
};

pub const DiagnosticsInfo = extern struct {
    cost: i32,
    flags: u32,
};

// TODO: this type is limited to platform 'windows6.0.6000'
const IID_INetDiagHelper_Value = Guid.initString("c0b35746-ebf5-11d8-bbe9-505054503030");
pub const IID_INetDiagHelper = &IID_INetDiagHelper_Value;
pub const INetDiagHelper = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        Initialize: *const fn (
            self: *const INetDiagHelper,
            celt: u32,
            rg_attributes: [*]HELPER_ATTRIBUTE,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetDiagnosticsInfo: *const fn (
            self: *const INetDiagHelper,
            pp_info: ?*?*DiagnosticsInfo,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetKeyAttributes: *const fn (
            self: *const INetDiagHelper,
            pcelt: ?*u32,
            pprg_attributes: [*]?*HELPER_ATTRIBUTE,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        LowHealth: *const fn (
            self: *const INetDiagHelper,
            pwsz_instance_description: ?[*:0]const u16,
            ppwsz_description: ?*?PWSTR,
            p_deferred_time: ?*i32,
            p_status: ?*DIAGNOSIS_STATUS,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        HighUtilization: *const fn (
            self: *const INetDiagHelper,
            pwsz_instance_description: ?[*:0]const u16,
            ppwsz_description: ?*?PWSTR,
            p_deferred_time: ?*i32,
            p_status: ?*DIAGNOSIS_STATUS,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetLowerHypotheses: *const fn (
            self: *const INetDiagHelper,
            pcelt: ?*u32,
            pprg_hypotheses: [*]?*HYPOTHESIS,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetDownStreamHypotheses: *const fn (
            self: *const INetDiagHelper,
            pcelt: ?*u32,
            pprg_hypotheses: [*]?*HYPOTHESIS,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetHigherHypotheses: *const fn (
            self: *const INetDiagHelper,
            pcelt: ?*u32,
            pprg_hypotheses: [*]?*HYPOTHESIS,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetUpStreamHypotheses: *const fn (
            self: *const INetDiagHelper,
            pcelt: ?*u32,
            pprg_hypotheses: [*]?*HYPOTHESIS,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Repair: *const fn (
            self: *const INetDiagHelper,
            p_info: ?*RepairInfo,
            p_deferred_time: ?*i32,
            p_status: ?*REPAIR_STATUS,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Validate: *const fn (
            self: *const INetDiagHelper,
            problem: PROBLEM_TYPE,
            p_deferred_time: ?*i32,
            p_status: ?*REPAIR_STATUS,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetRepairInfo: *const fn (
            self: *const INetDiagHelper,
            problem: PROBLEM_TYPE,
            pcelt: ?*u32,
            pp_info: [*]?*RepairInfo,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetLifeTime: *const fn (
            self: *const INetDiagHelper,
            p_life_time: ?*LIFE_TIME,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetLifeTime: *const fn (
            self: *const INetDiagHelper,
            life_time: LIFE_TIME,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetCacheTime: *const fn (
            self: *const INetDiagHelper,
            p_cache_time: ?*FILETIME,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetAttributes: *const fn (
            self: *const INetDiagHelper,
            pcelt: ?*u32,
            pprg_attributes: [*]?*HELPER_ATTRIBUTE,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Cancel: *const fn (
            self: *const INetDiagHelper,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Cleanup: *const fn (
            self: *const INetDiagHelper,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn initialize(self: *const T, celt_: u32, rg_attributes_: [*]HELPER_ATTRIBUTE) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).Initialize(@as(*const INetDiagHelper, @ptrCast(self)), celt_, rg_attributes_);
            }
            pub inline fn getDiagnosticsInfo(self: *const T, pp_info_: ?*?*DiagnosticsInfo) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).GetDiagnosticsInfo(@as(*const INetDiagHelper, @ptrCast(self)), pp_info_);
            }
            pub inline fn getKeyAttributes(self: *const T, pcelt_: ?*u32, pprg_attributes_: [*]?*HELPER_ATTRIBUTE) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).GetKeyAttributes(@as(*const INetDiagHelper, @ptrCast(self)), pcelt_, pprg_attributes_);
            }
            pub inline fn lowHealth(self: *const T, pwsz_instance_description_: ?[*:0]const u16, ppwsz_description_: ?*?PWSTR, p_deferred_time_: ?*i32, p_status_: ?*DIAGNOSIS_STATUS) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).LowHealth(@as(*const INetDiagHelper, @ptrCast(self)), pwsz_instance_description_, ppwsz_description_, p_deferred_time_, p_status_);
            }
            pub inline fn highUtilization(self: *const T, pwsz_instance_description_: ?[*:0]const u16, ppwsz_description_: ?*?PWSTR, p_deferred_time_: ?*i32, p_status_: ?*DIAGNOSIS_STATUS) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).HighUtilization(@as(*const INetDiagHelper, @ptrCast(self)), pwsz_instance_description_, ppwsz_description_, p_deferred_time_, p_status_);
            }
            pub inline fn getLowerHypotheses(self: *const T, pcelt_: ?*u32, pprg_hypotheses_: [*]?*HYPOTHESIS) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).GetLowerHypotheses(@as(*const INetDiagHelper, @ptrCast(self)), pcelt_, pprg_hypotheses_);
            }
            pub inline fn getDownStreamHypotheses(self: *const T, pcelt_: ?*u32, pprg_hypotheses_: [*]?*HYPOTHESIS) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).GetDownStreamHypotheses(@as(*const INetDiagHelper, @ptrCast(self)), pcelt_, pprg_hypotheses_);
            }
            pub inline fn getHigherHypotheses(self: *const T, pcelt_: ?*u32, pprg_hypotheses_: [*]?*HYPOTHESIS) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).GetHigherHypotheses(@as(*const INetDiagHelper, @ptrCast(self)), pcelt_, pprg_hypotheses_);
            }
            pub inline fn getUpStreamHypotheses(self: *const T, pcelt_: ?*u32, pprg_hypotheses_: [*]?*HYPOTHESIS) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).GetUpStreamHypotheses(@as(*const INetDiagHelper, @ptrCast(self)), pcelt_, pprg_hypotheses_);
            }
            pub inline fn repair(self: *const T, p_info_: ?*RepairInfo, p_deferred_time_: ?*i32, p_status_: ?*REPAIR_STATUS) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).Repair(@as(*const INetDiagHelper, @ptrCast(self)), p_info_, p_deferred_time_, p_status_);
            }
            pub inline fn validate(self: *const T, problem_: PROBLEM_TYPE, p_deferred_time_: ?*i32, p_status_: ?*REPAIR_STATUS) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).Validate(@as(*const INetDiagHelper, @ptrCast(self)), problem_, p_deferred_time_, p_status_);
            }
            pub inline fn getRepairInfo(self: *const T, problem_: PROBLEM_TYPE, pcelt_: ?*u32, pp_info_: [*]?*RepairInfo) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).GetRepairInfo(@as(*const INetDiagHelper, @ptrCast(self)), problem_, pcelt_, pp_info_);
            }
            pub inline fn getLifeTime(self: *const T, p_life_time_: ?*LIFE_TIME) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).GetLifeTime(@as(*const INetDiagHelper, @ptrCast(self)), p_life_time_);
            }
            pub inline fn setLifeTime(self: *const T, life_time_: LIFE_TIME) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).SetLifeTime(@as(*const INetDiagHelper, @ptrCast(self)), life_time_);
            }
            pub inline fn getCacheTime(self: *const T, p_cache_time_: ?*FILETIME) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).GetCacheTime(@as(*const INetDiagHelper, @ptrCast(self)), p_cache_time_);
            }
            pub inline fn getAttributes(self: *const T, pcelt_: ?*u32, pprg_attributes_: [*]?*HELPER_ATTRIBUTE) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).GetAttributes(@as(*const INetDiagHelper, @ptrCast(self)), pcelt_, pprg_attributes_);
            }
            pub inline fn cancel(self: *const T) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).Cancel(@as(*const INetDiagHelper, @ptrCast(self)));
            }
            pub inline fn cleanup(self: *const T) HRESULT {
                return @as(*const INetDiagHelper.VTable, @ptrCast(self.vtable)).Cleanup(@as(*const INetDiagHelper, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

pub const HypothesisResult = extern struct {
    hypothesis: HYPOTHESIS,
    pathStatus: DIAGNOSIS_STATUS,
};

// TODO: this type is limited to platform 'windows6.1'
const IID_INetDiagHelperUtilFactory_Value = Guid.initString("104613fb-bc57-4178-95ba-88809698354a");
pub const IID_INetDiagHelperUtilFactory = &IID_INetDiagHelperUtilFactory_Value;
pub const INetDiagHelperUtilFactory = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        CreateUtilityInstance: *const fn (
            self: *const INetDiagHelperUtilFactory,
            riid: ?*const Guid,
            ppv_object: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn createUtilityInstance(self: *const T, riid_: ?*const Guid, ppv_object_: ?*?*anyopaque) HRESULT {
                return @as(*const INetDiagHelperUtilFactory.VTable, @ptrCast(self.vtable)).CreateUtilityInstance(@as(*const INetDiagHelperUtilFactory, @ptrCast(self)), riid_, ppv_object_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows6.1'
const IID_INetDiagHelperEx_Value = Guid.initString("972dab4d-e4e3-4fc6-ae54-5f65ccde4a15");
pub const IID_INetDiagHelperEx = &IID_INetDiagHelperEx_Value;
pub const INetDiagHelperEx = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        ReconfirmLowHealth: *const fn (
            self: *const INetDiagHelperEx,
            celt: u32,
            p_results: [*]HypothesisResult,
            ppwsz_updated_description: ?*?PWSTR,
            p_updated_status: ?*DIAGNOSIS_STATUS,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetUtilities: *const fn (
            self: *const INetDiagHelperEx,
            p_utilities: ?*INetDiagHelperUtilFactory,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ReproduceFailure: *const fn (
            self: *const INetDiagHelperEx,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn reconfirmLowHealth(self: *const T, celt_: u32, p_results_: [*]HypothesisResult, ppwsz_updated_description_: ?*?PWSTR, p_updated_status_: ?*DIAGNOSIS_STATUS) HRESULT {
                return @as(*const INetDiagHelperEx.VTable, @ptrCast(self.vtable)).ReconfirmLowHealth(@as(*const INetDiagHelperEx, @ptrCast(self)), celt_, p_results_, ppwsz_updated_description_, p_updated_status_);
            }
            pub inline fn setUtilities(self: *const T, p_utilities_: ?*INetDiagHelperUtilFactory) HRESULT {
                return @as(*const INetDiagHelperEx.VTable, @ptrCast(self.vtable)).SetUtilities(@as(*const INetDiagHelperEx, @ptrCast(self)), p_utilities_);
            }
            pub inline fn reproduceFailure(self: *const T) HRESULT {
                return @as(*const INetDiagHelperEx.VTable, @ptrCast(self.vtable)).ReproduceFailure(@as(*const INetDiagHelperEx, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows6.0.6000'
const IID_INetDiagHelperInfo_Value = Guid.initString("c0b35747-ebf5-11d8-bbe9-505054503030");
pub const IID_INetDiagHelperInfo = &IID_INetDiagHelperInfo_Value;
pub const INetDiagHelperInfo = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        GetAttributeInfo: *const fn (
            self: *const INetDiagHelperInfo,
            pcelt: ?*u32,
            pprg_attribute_infos: [*]?*HelperAttributeInfo,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getAttributeInfo(self: *const T, pcelt_: ?*u32, pprg_attribute_infos_: [*]?*HelperAttributeInfo) HRESULT {
                return @as(*const INetDiagHelperInfo.VTable, @ptrCast(self.vtable)).GetAttributeInfo(@as(*const INetDiagHelperInfo, @ptrCast(self)), pcelt_, pprg_attribute_infos_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_INetDiagExtensibleHelper_Value = Guid.initString("c0b35748-ebf5-11d8-bbe9-505054503030");
pub const IID_INetDiagExtensibleHelper = &IID_INetDiagExtensibleHelper_Value;
pub const INetDiagExtensibleHelper = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        ResolveAttributes: *const fn (
            self: *const INetDiagExtensibleHelper,
            celt: u32,
            rg_key_attributes: [*]HELPER_ATTRIBUTE,
            pcelt: ?*u32,
            prg_match_values: [*]?*HELPER_ATTRIBUTE,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn resolveAttributes(self: *const T, celt_: u32, rg_key_attributes_: [*]HELPER_ATTRIBUTE, pcelt_: ?*u32, prg_match_values_: [*]?*HELPER_ATTRIBUTE) HRESULT {
                return @as(*const INetDiagExtensibleHelper.VTable, @ptrCast(self.vtable)).ResolveAttributes(@as(*const INetDiagExtensibleHelper, @ptrCast(self)), celt_, rg_key_attributes_, pcelt_, prg_match_values_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

//--------------------------------------------------------------------------------
// Section: Functions (16)
//--------------------------------------------------------------------------------
// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "ndfapi" fn NdfCreateIncident(
    helper_class_name: ?[*:0]const u16,
    celt: u32,
    attributes: [*]HELPER_ATTRIBUTE,
    handle: ?*?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "ndfapi" fn NdfCreateWinSockIncident(
    sock: ?SOCKET,
    host: ?[*:0]const u16,
    port: u16,
    app_id: ?[*:0]const u16,
    user_id: ?*SID,
    handle: ?*?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "ndfapi" fn NdfCreateWebIncident(
    url: ?[*:0]const u16,
    handle: ?*?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "ndfapi" fn NdfCreateWebIncidentEx(
    url: ?[*:0]const u16,
    use_win_h_t_t_p: BOOL,
    module_name: ?PWSTR,
    handle: ?*?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "ndfapi" fn NdfCreateSharingIncident(
    u_n_c_path: ?[*:0]const u16,
    handle: ?*?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "ndfapi" fn NdfCreateDNSIncident(
    hostname: ?[*:0]const u16,
    query_type: u16,
    handle: ?*?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "ndfapi" fn NdfCreateConnectivityIncident(
    handle: ?*?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows8.0'
pub extern "ndfapi" fn NdfCreateNetConnectionIncident(
    handle: ?*?*anyopaque,
    id: Guid,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.1'
pub extern "ndfapi" fn NdfCreatePnrpIncident(
    cloudname: ?[*:0]const u16,
    peername: ?[*:0]const u16,
    diagnose_publish: BOOL,
    app_id: ?[*:0]const u16,
    handle: ?*?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.1'
pub extern "ndfapi" fn NdfCreateGroupingIncident(
    cloud_name: ?[*:0]const u16,
    group_name: ?[*:0]const u16,
    identity: ?[*:0]const u16,
    invitation: ?[*:0]const u16,
    addresses: ?*SOCKET_ADDRESS_LIST,
    app_id: ?[*:0]const u16,
    handle: ?*?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "ndfapi" fn NdfExecuteDiagnosis(
    handle: ?*anyopaque,
    hwnd: ?HWND,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "ndfapi" fn NdfCloseIncident(
    handle: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.1'
pub extern "ndfapi" fn NdfDiagnoseIncident(
    handle: ?*anyopaque,
    root_cause_count: ?*u32,
    root_causes: ?*?*RootCauseInfo,
    dw_wait: u32,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.1'
pub extern "ndfapi" fn NdfRepairIncident(
    handle: ?*anyopaque,
    repair_ex: ?*RepairInfoEx,
    dw_wait: u32,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.1'
pub extern "ndfapi" fn NdfCancelIncident(
    handle: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.1'
pub extern "ndfapi" fn NdfGetTraceFile(
    handle: ?*anyopaque,
    trace_file_location: ?*?PWSTR,
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
// Section: Imports (11)
//--------------------------------------------------------------------------------
const Guid = @import("../zig.zig").Guid;
const BOOL = @import("../foundation.zig").BOOL;
const CHAR = @import("../foundation.zig").CHAR;
const FILETIME = @import("../foundation.zig").FILETIME;
const HRESULT = @import("../foundation.zig").HRESULT;
const HWND = @import("../foundation.zig").HWND;
const IUnknown = @import("../system/com.zig").IUnknown;
const PWSTR = @import("../foundation.zig").PWSTR;
const SID = @import("../security.zig").SID;
const SOCKET = @import("../networking/win_sock.zig").SOCKET;
const SOCKET_ADDRESS_LIST = @import("../networking/win_sock.zig").SOCKET_ADDRESS_LIST;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
