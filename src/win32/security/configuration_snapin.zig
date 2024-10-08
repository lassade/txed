//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (33)
//--------------------------------------------------------------------------------
pub const cNodetypeSceTemplateServices = Guid.initString("24a7f717-1f0c-11d1-affb-00c04fb984f9");
pub const cNodetypeSceAnalysisServices = Guid.initString("678050c7-1ff8-11d1-affb-00c04fb984f9");
pub const cNodetypeSceEventLog = Guid.initString("2ce06698-4bf3-11d1-8c30-00c04fb984f9");
pub const SCESTATUS_SUCCESS = @as(i32, 0);
pub const SCESTATUS_INVALID_PARAMETER = @as(i32, 1);
pub const SCESTATUS_RECORD_NOT_FOUND = @as(i32, 2);
pub const SCESTATUS_INVALID_DATA = @as(i32, 3);
pub const SCESTATUS_OBJECT_EXIST = @as(i32, 4);
pub const SCESTATUS_BUFFER_TOO_SMALL = @as(i32, 5);
pub const SCESTATUS_PROFILE_NOT_FOUND = @as(i32, 6);
pub const SCESTATUS_BAD_FORMAT = @as(i32, 7);
pub const SCESTATUS_NOT_ENOUGH_RESOURCE = @as(i32, 8);
pub const SCESTATUS_ACCESS_DENIED = @as(i32, 9);
pub const SCESTATUS_CANT_DELETE = @as(i32, 10);
pub const SCESTATUS_PREFIX_OVERFLOW = @as(i32, 11);
pub const SCESTATUS_OTHER_ERROR = @as(i32, 12);
pub const SCESTATUS_ALREADY_RUNNING = @as(i32, 13);
pub const SCESTATUS_SERVICE_NOT_SUPPORT = @as(i32, 14);
pub const SCESTATUS_MOD_NOT_FOUND = @as(i32, 15);
pub const SCESTATUS_EXCEPTION_IN_SERVER = @as(i32, 16);
pub const SCESTATUS_NO_TEMPLATE_GIVEN = @as(i32, 17);
pub const SCESTATUS_NO_MAPPING = @as(i32, 18);
pub const SCESTATUS_TRUST_FAIL = @as(i32, 19);
pub const SCE_ROOT_PATH = "Software\\Microsoft\\Windows NT\\CurrentVersion\\SeCEdit";
pub const SCESVC_ENUMERATION_MAX = @as(i32, 100);
pub const struuidNodetypeSceTemplateServices = "{24a7f717-1f0c-11d1-affb-00c04fb984f9}";
pub const lstruuidNodetypeSceTemplateServices = "{24a7f717-1f0c-11d1-affb-00c04fb984f9}";
pub const struuidNodetypeSceAnalysisServices = "{678050c7-1ff8-11d1-affb-00c04fb984f9}";
pub const lstruuidNodetypeSceAnalysisServices = "{678050c7-1ff8-11d1-affb-00c04fb984f9}";
pub const struuidNodetypeSceEventLog = "{2ce06698-4bf3-11d1-8c30-00c04fb984f9}";
pub const lstruuidNodetypeSceEventLog = "{2ce06698-4bf3-11d1-8c30-00c04fb984f9}";
pub const CCF_SCESVC_ATTACHMENT = "CCF_SCESVC_ATTACHMENT";
pub const CCF_SCESVC_ATTACHMENT_DATA = "CCF_SCESVC_ATTACHMENT_DATA";

//--------------------------------------------------------------------------------
// Section: Types (15)
//--------------------------------------------------------------------------------
pub const SCE_LOG_ERR_LEVEL = enum(u32) {
    ALWAYS = 0,
    ERROR = 1,
    DETAIL = 2,
    DEBUG = 3,
};
pub const SCE_LOG_LEVEL_ALWAYS = SCE_LOG_ERR_LEVEL.ALWAYS;
pub const SCE_LOG_LEVEL_ERROR = SCE_LOG_ERR_LEVEL.ERROR;
pub const SCE_LOG_LEVEL_DETAIL = SCE_LOG_ERR_LEVEL.DETAIL;
pub const SCE_LOG_LEVEL_DEBUG = SCE_LOG_ERR_LEVEL.DEBUG;

pub const SCESVC_CONFIGURATION_LINE = extern struct {
    Key: ?*i8,
    Value: ?*i8,
    ValueLen: u32,
};

pub const SCESVC_CONFIGURATION_INFO = extern struct {
    Count: u32,
    Lines: ?*SCESVC_CONFIGURATION_LINE,
};

pub const SCESVC_INFO_TYPE = enum(i32) {
    ConfigurationInfo = 0,
    MergedPolicyInfo = 1,
    AnalysisInfo = 2,
    InternalUse = 3,
};
pub const SceSvcConfigurationInfo = SCESVC_INFO_TYPE.ConfigurationInfo;
pub const SceSvcMergedPolicyInfo = SCESVC_INFO_TYPE.MergedPolicyInfo;
pub const SceSvcAnalysisInfo = SCESVC_INFO_TYPE.AnalysisInfo;
pub const SceSvcInternalUse = SCESVC_INFO_TYPE.InternalUse;

pub const SCESVC_ANALYSIS_LINE = extern struct {
    Key: ?*i8,
    Value: ?*u8,
    ValueLen: u32,
};

pub const SCESVC_ANALYSIS_INFO = extern struct {
    Count: u32,
    Lines: ?*SCESVC_ANALYSIS_LINE,
};

pub const PFSCE_QUERY_INFO = *const fn (
    sce_handle: ?*anyopaque,
    sce_type: SCESVC_INFO_TYPE,
    lp_prefix: ?*i8,
    b_exact: BOOL,
    ppv_info: ?*?*anyopaque,
    psce_enum_handle: ?*u32,
) callconv(@import("std").os.windows.WINAPI) u32;

pub const PFSCE_SET_INFO = *const fn (
    sce_handle: ?*anyopaque,
    sce_type: SCESVC_INFO_TYPE,
    lp_prefix: ?*i8,
    b_exact: BOOL,
    pv_info: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) u32;

pub const PFSCE_FREE_INFO = *const fn (
    pv_service_info: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) u32;

pub const PFSCE_LOG_INFO = *const fn (
    err_level: SCE_LOG_ERR_LEVEL,
    win32rc: u32,
    p_err_fmt: ?*i8,
) callconv(@import("std").os.windows.WINAPI) u32;

pub const SCESVC_CALLBACK_INFO = extern struct {
    sceHandle: ?*anyopaque,
    pfQueryInfo: ?PFSCE_QUERY_INFO,
    pfSetInfo: ?PFSCE_SET_INFO,
    pfFreeInfo: ?PFSCE_FREE_INFO,
    pfLogInfo: ?PFSCE_LOG_INFO,
};

pub const PF_ConfigAnalyzeService = *const fn (
    p_sce_cb_info: ?*SCESVC_CALLBACK_INFO,
) callconv(@import("std").os.windows.WINAPI) u32;

pub const PF_UpdateService = *const fn (
    p_sce_cb_info: ?*SCESVC_CALLBACK_INFO,
    service_info: ?*SCESVC_CONFIGURATION_INFO,
) callconv(@import("std").os.windows.WINAPI) u32;

// TODO: this type is limited to platform 'windows5.1.2600'
const IID_ISceSvcAttachmentPersistInfo_Value = Guid.initString("6d90e0d0-200d-11d1-affb-00c04fb984f9");
pub const IID_ISceSvcAttachmentPersistInfo = &IID_ISceSvcAttachmentPersistInfo_Value;
pub const ISceSvcAttachmentPersistInfo = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        Save: *const fn (
            self: *const ISceSvcAttachmentPersistInfo,
            lp_template_name: ?*i8,
            scesvc_handle: ?*?*anyopaque,
            ppv_data: ?*?*anyopaque,
            pb_overwrite_all: ?*BOOL,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        IsDirty: *const fn (
            self: *const ISceSvcAttachmentPersistInfo,
            lp_template_name: ?*i8,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        FreeBuffer: *const fn (
            self: *const ISceSvcAttachmentPersistInfo,
            pv_data: ?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn save(self: *const T, lp_template_name_: ?*i8, scesvc_handle_: ?*?*anyopaque, ppv_data_: ?*?*anyopaque, pb_overwrite_all_: ?*BOOL) HRESULT {
                return @as(*const ISceSvcAttachmentPersistInfo.VTable, @ptrCast(self.vtable)).Save(@as(*const ISceSvcAttachmentPersistInfo, @ptrCast(self)), lp_template_name_, scesvc_handle_, ppv_data_, pb_overwrite_all_);
            }
            pub inline fn isDirty(self: *const T, lp_template_name_: ?*i8) HRESULT {
                return @as(*const ISceSvcAttachmentPersistInfo.VTable, @ptrCast(self.vtable)).IsDirty(@as(*const ISceSvcAttachmentPersistInfo, @ptrCast(self)), lp_template_name_);
            }
            pub inline fn freeBuffer(self: *const T, pv_data_: ?*anyopaque) HRESULT {
                return @as(*const ISceSvcAttachmentPersistInfo.VTable, @ptrCast(self.vtable)).FreeBuffer(@as(*const ISceSvcAttachmentPersistInfo, @ptrCast(self)), pv_data_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows5.1.2600'
const IID_ISceSvcAttachmentData_Value = Guid.initString("17c35fde-200d-11d1-affb-00c04fb984f9");
pub const IID_ISceSvcAttachmentData = &IID_ISceSvcAttachmentData_Value;
pub const ISceSvcAttachmentData = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        GetData: *const fn (
            self: *const ISceSvcAttachmentData,
            scesvc_handle: ?*anyopaque,
            sce_type: SCESVC_INFO_TYPE,
            ppv_data: ?*?*anyopaque,
            psce_enum_handle: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Initialize: *const fn (
            self: *const ISceSvcAttachmentData,
            lp_service_name: ?*i8,
            lp_template_name: ?*i8,
            lp_sce_svc_persist_info: ?*ISceSvcAttachmentPersistInfo,
            pscesvc_handle: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        FreeBuffer: *const fn (
            self: *const ISceSvcAttachmentData,
            pv_data: ?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        CloseHandle: *const fn (
            self: *const ISceSvcAttachmentData,
            scesvc_handle: ?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getData(self: *const T, scesvc_handle_: ?*anyopaque, sce_type_: SCESVC_INFO_TYPE, ppv_data_: ?*?*anyopaque, psce_enum_handle_: ?*u32) HRESULT {
                return @as(*const ISceSvcAttachmentData.VTable, @ptrCast(self.vtable)).GetData(@as(*const ISceSvcAttachmentData, @ptrCast(self)), scesvc_handle_, sce_type_, ppv_data_, psce_enum_handle_);
            }
            pub inline fn initialize(self: *const T, lp_service_name_: ?*i8, lp_template_name_: ?*i8, lp_sce_svc_persist_info_: ?*ISceSvcAttachmentPersistInfo, pscesvc_handle_: ?*?*anyopaque) HRESULT {
                return @as(*const ISceSvcAttachmentData.VTable, @ptrCast(self.vtable)).Initialize(@as(*const ISceSvcAttachmentData, @ptrCast(self)), lp_service_name_, lp_template_name_, lp_sce_svc_persist_info_, pscesvc_handle_);
            }
            pub inline fn freeBuffer(self: *const T, pv_data_: ?*anyopaque) HRESULT {
                return @as(*const ISceSvcAttachmentData.VTable, @ptrCast(self.vtable)).FreeBuffer(@as(*const ISceSvcAttachmentData, @ptrCast(self)), pv_data_);
            }
            pub inline fn closeHandle(self: *const T, scesvc_handle_: ?*anyopaque) HRESULT {
                return @as(*const ISceSvcAttachmentData.VTable, @ptrCast(self.vtable)).CloseHandle(@as(*const ISceSvcAttachmentData, @ptrCast(self)), scesvc_handle_);
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
pub usingnamespace switch (@import("../zig.zig").unicode_mode) {
    .ansi => struct {},
    .wide => struct {},
    .unspecified => if (@import("builtin").is_test) struct {} else struct {},
};
//--------------------------------------------------------------------------------
// Section: Imports (4)
//--------------------------------------------------------------------------------
const Guid = @import("../zig.zig").Guid;
const BOOL = @import("../foundation.zig").BOOL;
const HRESULT = @import("../foundation.zig").HRESULT;
const IUnknown = @import("../system/com.zig").IUnknown;

test {
    // The following '_ = <FuncPtrType>' lines are a workaround for https://github.com/ziglang/zig/issues/4476
    if (@hasDecl(@This(), "PFSCE_QUERY_INFO")) {
        _ = PFSCE_QUERY_INFO;
    }
    if (@hasDecl(@This(), "PFSCE_SET_INFO")) {
        _ = PFSCE_SET_INFO;
    }
    if (@hasDecl(@This(), "PFSCE_FREE_INFO")) {
        _ = PFSCE_FREE_INFO;
    }
    if (@hasDecl(@This(), "PFSCE_LOG_INFO")) {
        _ = PFSCE_LOG_INFO;
    }
    if (@hasDecl(@This(), "PF_ConfigAnalyzeService")) {
        _ = PF_ConfigAnalyzeService;
    }
    if (@hasDecl(@This(), "PF_UpdateService")) {
        _ = PF_UpdateService;
    }

    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
