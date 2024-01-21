//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (15)
//--------------------------------------------------------------------------------
// TODO: this type has a FreeFunc 'HcsCloseOperation', what can Zig do with this information?
// TODO: this type has an InvalidHandleValue of '0', what can Zig do with this information?
pub const HCS_OPERATION = isize;

// TODO: this type has a FreeFunc 'HcsCloseComputeSystem', what can Zig do with this information?
// TODO: this type has an InvalidHandleValue of '0', what can Zig do with this information?
pub const HCS_SYSTEM = isize;

// TODO: this type has a FreeFunc 'HcsCloseProcess', what can Zig do with this information?
// TODO: this type has an InvalidHandleValue of '0', what can Zig do with this information?
pub const HCS_PROCESS = isize;

pub const HCS_OPERATION_TYPE = enum(i32) {
    None = -1,
    Enumerate = 0,
    Create = 1,
    Start = 2,
    Shutdown = 3,
    Pause = 4,
    Resume = 5,
    Save = 6,
    Terminate = 7,
    Modify = 8,
    GetProperties = 9,
    CreateProcess = 10,
    SignalProcess = 11,
    GetProcessInfo = 12,
    GetProcessProperties = 13,
    ModifyProcess = 14,
    Crash = 15,
};
pub const HcsOperationTypeNone = HCS_OPERATION_TYPE.None;
pub const HcsOperationTypeEnumerate = HCS_OPERATION_TYPE.Enumerate;
pub const HcsOperationTypeCreate = HCS_OPERATION_TYPE.Create;
pub const HcsOperationTypeStart = HCS_OPERATION_TYPE.Start;
pub const HcsOperationTypeShutdown = HCS_OPERATION_TYPE.Shutdown;
pub const HcsOperationTypePause = HCS_OPERATION_TYPE.Pause;
pub const HcsOperationTypeResume = HCS_OPERATION_TYPE.Resume;
pub const HcsOperationTypeSave = HCS_OPERATION_TYPE.Save;
pub const HcsOperationTypeTerminate = HCS_OPERATION_TYPE.Terminate;
pub const HcsOperationTypeModify = HCS_OPERATION_TYPE.Modify;
pub const HcsOperationTypeGetProperties = HCS_OPERATION_TYPE.GetProperties;
pub const HcsOperationTypeCreateProcess = HCS_OPERATION_TYPE.CreateProcess;
pub const HcsOperationTypeSignalProcess = HCS_OPERATION_TYPE.SignalProcess;
pub const HcsOperationTypeGetProcessInfo = HCS_OPERATION_TYPE.GetProcessInfo;
pub const HcsOperationTypeGetProcessProperties = HCS_OPERATION_TYPE.GetProcessProperties;
pub const HcsOperationTypeModifyProcess = HCS_OPERATION_TYPE.ModifyProcess;
pub const HcsOperationTypeCrash = HCS_OPERATION_TYPE.Crash;

pub const HCS_OPERATION_COMPLETION = *const fn (
    operation: HCS_OPERATION,
    context: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) void;

pub const HCS_EVENT_TYPE = enum(i32) {
    Invalid = 0,
    SystemExited = 1,
    SystemCrashInitiated = 2,
    SystemCrashReport = 3,
    SystemRdpEnhancedModeStateChanged = 4,
    SystemSiloJobCreated = 5,
    SystemGuestConnectionClosed = 6,
    ProcessExited = 65536,
    OperationCallback = 16777216,
    ServiceDisconnect = 33554432,
};
pub const HcsEventInvalid = HCS_EVENT_TYPE.Invalid;
pub const HcsEventSystemExited = HCS_EVENT_TYPE.SystemExited;
pub const HcsEventSystemCrashInitiated = HCS_EVENT_TYPE.SystemCrashInitiated;
pub const HcsEventSystemCrashReport = HCS_EVENT_TYPE.SystemCrashReport;
pub const HcsEventSystemRdpEnhancedModeStateChanged = HCS_EVENT_TYPE.SystemRdpEnhancedModeStateChanged;
pub const HcsEventSystemSiloJobCreated = HCS_EVENT_TYPE.SystemSiloJobCreated;
pub const HcsEventSystemGuestConnectionClosed = HCS_EVENT_TYPE.SystemGuestConnectionClosed;
pub const HcsEventProcessExited = HCS_EVENT_TYPE.ProcessExited;
pub const HcsEventOperationCallback = HCS_EVENT_TYPE.OperationCallback;
pub const HcsEventServiceDisconnect = HCS_EVENT_TYPE.ServiceDisconnect;

pub const HCS_EVENT = extern struct {
    Type: HCS_EVENT_TYPE,
    EventData: ?[*:0]const u16,
    Operation: HCS_OPERATION,
};

pub const HCS_EVENT_OPTIONS = enum(u32) {
    None = 0,
    EnableOperationCallbacks = 1,
    _,
    pub fn initFlags(o: struct {
        None: u1 = 0,
        EnableOperationCallbacks: u1 = 0,
    }) HCS_EVENT_OPTIONS {
        return @as(HCS_EVENT_OPTIONS, @enumFromInt((if (o.None == 1) @intFromEnum(HCS_EVENT_OPTIONS.None) else 0) | (if (o.EnableOperationCallbacks == 1) @intFromEnum(HCS_EVENT_OPTIONS.EnableOperationCallbacks) else 0)));
    }
};
pub const HcsEventOptionNone = HCS_EVENT_OPTIONS.None;
pub const HcsEventOptionEnableOperationCallbacks = HCS_EVENT_OPTIONS.EnableOperationCallbacks;

pub const HCS_EVENT_CALLBACK = *const fn (
    event: ?*HCS_EVENT,
    context: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) void;

pub const HCS_NOTIFICATION_FLAGS = enum(i32) {
    Success = 0,
    Failure = -2147483648,
};
pub const HcsNotificationFlagSuccess = HCS_NOTIFICATION_FLAGS.Success;
pub const HcsNotificationFlagFailure = HCS_NOTIFICATION_FLAGS.Failure;

pub const HCS_NOTIFICATIONS = enum(i32) {
    Invalid = 0,
    SystemExited = 1,
    SystemCreateCompleted = 2,
    SystemStartCompleted = 3,
    SystemPauseCompleted = 4,
    SystemResumeCompleted = 5,
    SystemCrashReport = 6,
    SystemSiloJobCreated = 7,
    SystemSaveCompleted = 8,
    SystemRdpEnhancedModeStateChanged = 9,
    SystemShutdownFailed = 10,
    // SystemShutdownCompleted = 10, this enum value conflicts with SystemShutdownFailed
    SystemGetPropertiesCompleted = 11,
    SystemModifyCompleted = 12,
    SystemCrashInitiated = 13,
    SystemGuestConnectionClosed = 14,
    SystemOperationCompletion = 15,
    SystemPassThru = 16,
    ProcessExited = 65536,
    ServiceDisconnect = 16777216,
    FlagsReserved = -268435456,
};
pub const HcsNotificationInvalid = HCS_NOTIFICATIONS.Invalid;
pub const HcsNotificationSystemExited = HCS_NOTIFICATIONS.SystemExited;
pub const HcsNotificationSystemCreateCompleted = HCS_NOTIFICATIONS.SystemCreateCompleted;
pub const HcsNotificationSystemStartCompleted = HCS_NOTIFICATIONS.SystemStartCompleted;
pub const HcsNotificationSystemPauseCompleted = HCS_NOTIFICATIONS.SystemPauseCompleted;
pub const HcsNotificationSystemResumeCompleted = HCS_NOTIFICATIONS.SystemResumeCompleted;
pub const HcsNotificationSystemCrashReport = HCS_NOTIFICATIONS.SystemCrashReport;
pub const HcsNotificationSystemSiloJobCreated = HCS_NOTIFICATIONS.SystemSiloJobCreated;
pub const HcsNotificationSystemSaveCompleted = HCS_NOTIFICATIONS.SystemSaveCompleted;
pub const HcsNotificationSystemRdpEnhancedModeStateChanged = HCS_NOTIFICATIONS.SystemRdpEnhancedModeStateChanged;
pub const HcsNotificationSystemShutdownFailed = HCS_NOTIFICATIONS.SystemShutdownFailed;
pub const HcsNotificationSystemShutdownCompleted = HCS_NOTIFICATIONS.SystemShutdownFailed;
pub const HcsNotificationSystemGetPropertiesCompleted = HCS_NOTIFICATIONS.SystemGetPropertiesCompleted;
pub const HcsNotificationSystemModifyCompleted = HCS_NOTIFICATIONS.SystemModifyCompleted;
pub const HcsNotificationSystemCrashInitiated = HCS_NOTIFICATIONS.SystemCrashInitiated;
pub const HcsNotificationSystemGuestConnectionClosed = HCS_NOTIFICATIONS.SystemGuestConnectionClosed;
pub const HcsNotificationSystemOperationCompletion = HCS_NOTIFICATIONS.SystemOperationCompletion;
pub const HcsNotificationSystemPassThru = HCS_NOTIFICATIONS.SystemPassThru;
pub const HcsNotificationProcessExited = HCS_NOTIFICATIONS.ProcessExited;
pub const HcsNotificationServiceDisconnect = HCS_NOTIFICATIONS.ServiceDisconnect;
pub const HcsNotificationFlagsReserved = HCS_NOTIFICATIONS.FlagsReserved;

pub const HCS_NOTIFICATION_CALLBACK = *const fn (
    notification_type: u32,
    context: ?*anyopaque,
    notification_status: HRESULT,
    notification_data: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) void;

pub const HCS_PROCESS_INFORMATION = extern struct {
    ProcessId: u32,
    Reserved: u32,
    StdInput: ?HANDLE,
    StdOutput: ?HANDLE,
    StdError: ?HANDLE,
};

pub const HCS_CREATE_OPTIONS = enum(i32) {
    @"1" = 65536,
};
pub const HcsCreateOptions_1 = HCS_CREATE_OPTIONS.@"1";

pub const HCS_CREATE_OPTIONS_1 = extern struct {
    Version: HCS_CREATE_OPTIONS,
    UserToken: ?HANDLE,
    SecurityDescriptor: ?*SECURITY_DESCRIPTOR,
    CallbackOptions: HCS_EVENT_OPTIONS,
    CallbackContext: ?*anyopaque,
    Callback: ?HCS_EVENT_CALLBACK,
};

//--------------------------------------------------------------------------------
// Section: Functions (64)
//--------------------------------------------------------------------------------
pub extern "computecore" fn HcsEnumerateComputeSystems(
    query: ?[*:0]const u16,
    operation: HCS_OPERATION,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsEnumerateComputeSystemsInNamespace(
    id_namespace: ?[*:0]const u16,
    query: ?[*:0]const u16,
    operation: HCS_OPERATION,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsCreateOperation(
    context: ?*const anyopaque,
    callback: ?HCS_OPERATION_COMPLETION,
) callconv(@import("std").os.windows.WINAPI) HCS_OPERATION;

pub extern "computecore" fn HcsCloseOperation(
    operation: HCS_OPERATION,
) callconv(@import("std").os.windows.WINAPI) void;

pub extern "computecore" fn HcsGetOperationContext(
    operation: HCS_OPERATION,
) callconv(@import("std").os.windows.WINAPI) ?*anyopaque;

pub extern "computecore" fn HcsSetOperationContext(
    operation: HCS_OPERATION,
    context: ?*const anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsGetComputeSystemFromOperation(
    operation: HCS_OPERATION,
) callconv(@import("std").os.windows.WINAPI) HCS_SYSTEM;

pub extern "computecore" fn HcsGetProcessFromOperation(
    operation: HCS_OPERATION,
) callconv(@import("std").os.windows.WINAPI) HCS_PROCESS;

pub extern "computecore" fn HcsGetOperationType(
    operation: HCS_OPERATION,
) callconv(@import("std").os.windows.WINAPI) HCS_OPERATION_TYPE;

pub extern "computecore" fn HcsGetOperationId(
    operation: HCS_OPERATION,
) callconv(@import("std").os.windows.WINAPI) u64;

pub extern "computecore" fn HcsGetOperationResult(
    operation: HCS_OPERATION,
    result_document: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsGetOperationResultAndProcessInfo(
    operation: HCS_OPERATION,
    process_information: ?*HCS_PROCESS_INFORMATION,
    result_document: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsGetProcessorCompatibilityFromSavedState(
    runtime_file_name: ?[*:0]const u16,
    processor_features_string: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsWaitForOperationResult(
    operation: HCS_OPERATION,
    timeout_ms: u32,
    result_document: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsWaitForOperationResultAndProcessInfo(
    operation: HCS_OPERATION,
    timeout_ms: u32,
    process_information: ?*HCS_PROCESS_INFORMATION,
    result_document: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsSetOperationCallback(
    operation: HCS_OPERATION,
    context: ?*const anyopaque,
    callback: ?HCS_OPERATION_COMPLETION,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsCancelOperation(
    operation: HCS_OPERATION,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsCreateComputeSystem(
    id: ?[*:0]const u16,
    configuration: ?[*:0]const u16,
    operation: HCS_OPERATION,
    security_descriptor: ?*const SECURITY_DESCRIPTOR,
    compute_system: ?*HCS_SYSTEM,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsCreateComputeSystemInNamespace(
    id_namespace: ?[*:0]const u16,
    id: ?[*:0]const u16,
    configuration: ?[*:0]const u16,
    operation: HCS_OPERATION,
    options: ?*const HCS_CREATE_OPTIONS,
    compute_system: ?*HCS_SYSTEM,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsOpenComputeSystem(
    id: ?[*:0]const u16,
    requested_access: u32,
    compute_system: ?*HCS_SYSTEM,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsOpenComputeSystemInNamespace(
    id_namespace: ?[*:0]const u16,
    id: ?[*:0]const u16,
    requested_access: u32,
    compute_system: ?*HCS_SYSTEM,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsCloseComputeSystem(
    compute_system: HCS_SYSTEM,
) callconv(@import("std").os.windows.WINAPI) void;

pub extern "computecore" fn HcsStartComputeSystem(
    compute_system: HCS_SYSTEM,
    operation: HCS_OPERATION,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsShutDownComputeSystem(
    compute_system: HCS_SYSTEM,
    operation: HCS_OPERATION,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsTerminateComputeSystem(
    compute_system: HCS_SYSTEM,
    operation: HCS_OPERATION,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsCrashComputeSystem(
    compute_system: HCS_SYSTEM,
    operation: HCS_OPERATION,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsPauseComputeSystem(
    compute_system: HCS_SYSTEM,
    operation: HCS_OPERATION,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsResumeComputeSystem(
    compute_system: HCS_SYSTEM,
    operation: HCS_OPERATION,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsSaveComputeSystem(
    compute_system: HCS_SYSTEM,
    operation: HCS_OPERATION,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsGetComputeSystemProperties(
    compute_system: HCS_SYSTEM,
    operation: HCS_OPERATION,
    property_query: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsModifyComputeSystem(
    compute_system: HCS_SYSTEM,
    operation: HCS_OPERATION,
    configuration: ?[*:0]const u16,
    identity: ?HANDLE,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsWaitForComputeSystemExit(
    compute_system: HCS_SYSTEM,
    timeout_ms: u32,
    result: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsSetComputeSystemCallback(
    compute_system: HCS_SYSTEM,
    callback_options: HCS_EVENT_OPTIONS,
    context: ?*const anyopaque,
    callback: ?HCS_EVENT_CALLBACK,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsCreateProcess(
    compute_system: HCS_SYSTEM,
    process_parameters: ?[*:0]const u16,
    operation: HCS_OPERATION,
    security_descriptor: ?*const SECURITY_DESCRIPTOR,
    process: ?*HCS_PROCESS,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsOpenProcess(
    compute_system: HCS_SYSTEM,
    process_id: u32,
    requested_access: u32,
    process: ?*HCS_PROCESS,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsCloseProcess(
    process: HCS_PROCESS,
) callconv(@import("std").os.windows.WINAPI) void;

pub extern "computecore" fn HcsTerminateProcess(
    process: HCS_PROCESS,
    operation: HCS_OPERATION,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsSignalProcess(
    process: HCS_PROCESS,
    operation: HCS_OPERATION,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsGetProcessInfo(
    process: HCS_PROCESS,
    operation: HCS_OPERATION,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsGetProcessProperties(
    process: HCS_PROCESS,
    operation: HCS_OPERATION,
    property_query: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsModifyProcess(
    process: HCS_PROCESS,
    operation: HCS_OPERATION,
    settings: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsSetProcessCallback(
    process: HCS_PROCESS,
    callback_options: HCS_EVENT_OPTIONS,
    context: ?*anyopaque,
    callback: ?HCS_EVENT_CALLBACK,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsWaitForProcessExit(
    compute_system: HCS_PROCESS,
    timeout_ms: u32,
    result: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsGetServiceProperties(
    property_query: ?[*:0]const u16,
    result: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsModifyServiceSettings(
    settings: ?[*:0]const u16,
    result: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsSubmitWerReport(
    settings: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsCreateEmptyGuestStateFile(
    guest_state_file_path: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsCreateEmptyRuntimeStateFile(
    runtime_state_file_path: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsGrantVmAccess(
    vm_id: ?[*:0]const u16,
    file_path: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsRevokeVmAccess(
    vm_id: ?[*:0]const u16,
    file_path: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsGrantVmGroupAccess(
    file_path: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computecore" fn HcsRevokeVmGroupAccess(
    file_path: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsImportLayer(
    layer_path: ?[*:0]const u16,
    source_folder_path: ?[*:0]const u16,
    layer_data: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsExportLayer(
    layer_path: ?[*:0]const u16,
    export_folder_path: ?[*:0]const u16,
    layer_data: ?[*:0]const u16,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsExportLegacyWritableLayer(
    writable_layer_mount_path: ?[*:0]const u16,
    writable_layer_folder_path: ?[*:0]const u16,
    export_folder_path: ?[*:0]const u16,
    layer_data: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsDestroyLayer(
    layer_path: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsSetupBaseOSLayer(
    layer_path: ?[*:0]const u16,
    vhd_handle: ?HANDLE,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsInitializeWritableLayer(
    writable_layer_path: ?[*:0]const u16,
    layer_data: ?[*:0]const u16,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsInitializeLegacyWritableLayer(
    writable_layer_mount_path: ?[*:0]const u16,
    writable_layer_folder_path: ?[*:0]const u16,
    layer_data: ?[*:0]const u16,
    options: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsAttachLayerStorageFilter(
    layer_path: ?[*:0]const u16,
    layer_data: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsDetachLayerStorageFilter(
    layer_path: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsFormatWritableLayerVhd(
    vhd_handle: ?HANDLE,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsGetLayerVhdMountPath(
    vhd_handle: ?HANDLE,
    mount_path: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computestorage" fn HcsSetupBaseOSVolume(
    layer_path: ?[*:0]const u16,
    volume_path: ?[*:0]const u16,
    options: ?[*:0]const u16,
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
// Section: Imports (4)
//--------------------------------------------------------------------------------
const HANDLE = @import("../foundation.zig").HANDLE;
const HRESULT = @import("../foundation.zig").HRESULT;
const PWSTR = @import("../foundation.zig").PWSTR;
const SECURITY_DESCRIPTOR = @import("../security.zig").SECURITY_DESCRIPTOR;

test {
    // The following '_ = <FuncPtrType>' lines are a workaround for https://github.com/ziglang/zig/issues/4476
    if (@hasDecl(@This(), "HCS_OPERATION_COMPLETION")) {
        _ = HCS_OPERATION_COMPLETION;
    }
    if (@hasDecl(@This(), "HCS_EVENT_CALLBACK")) {
        _ = HCS_EVENT_CALLBACK;
    }
    if (@hasDecl(@This(), "HCS_NOTIFICATION_CALLBACK")) {
        _ = HCS_NOTIFICATION_CALLBACK;
    }

    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}