//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (1)
//--------------------------------------------------------------------------------
pub const REGISTER_APPLICATION_RESTART_FLAGS = enum(u32) {
    CRASH = 1,
    HANG = 2,
    PATCH = 4,
    REBOOT = 8,
    _,
    pub fn initFlags(o: struct {
        CRASH: u1 = 0,
        HANG: u1 = 0,
        PATCH: u1 = 0,
        REBOOT: u1 = 0,
    }) REGISTER_APPLICATION_RESTART_FLAGS {
        return @as(REGISTER_APPLICATION_RESTART_FLAGS, @enumFromInt((if (o.CRASH == 1) @intFromEnum(REGISTER_APPLICATION_RESTART_FLAGS.CRASH) else 0) | (if (o.HANG == 1) @intFromEnum(REGISTER_APPLICATION_RESTART_FLAGS.HANG) else 0) | (if (o.PATCH == 1) @intFromEnum(REGISTER_APPLICATION_RESTART_FLAGS.PATCH) else 0) | (if (o.REBOOT == 1) @intFromEnum(REGISTER_APPLICATION_RESTART_FLAGS.REBOOT) else 0)));
    }
};
pub const RESTART_NO_CRASH = REGISTER_APPLICATION_RESTART_FLAGS.CRASH;
pub const RESTART_NO_HANG = REGISTER_APPLICATION_RESTART_FLAGS.HANG;
pub const RESTART_NO_PATCH = REGISTER_APPLICATION_RESTART_FLAGS.PATCH;
pub const RESTART_NO_REBOOT = REGISTER_APPLICATION_RESTART_FLAGS.REBOOT;

//--------------------------------------------------------------------------------
// Section: Functions (8)
//--------------------------------------------------------------------------------
// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn RegisterApplicationRecoveryCallback(
    p_recovey_callback: ?APPLICATION_RECOVERY_CALLBACK,
    pv_parameter: ?*anyopaque,
    dw_ping_interval: u32,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn UnregisterApplicationRecoveryCallback() callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn RegisterApplicationRestart(
    pwz_commandline: ?[*:0]const u16,
    dw_flags: REGISTER_APPLICATION_RESTART_FLAGS,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn UnregisterApplicationRestart() callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn GetApplicationRecoveryCallback(
    h_process: ?HANDLE,
    p_recovery_callback: ?*?APPLICATION_RECOVERY_CALLBACK,
    ppv_parameter: ?*?*anyopaque,
    pdw_ping_interval: ?*u32,
    pdw_flags: ?*u32,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn GetApplicationRestartSettings(
    h_process: ?HANDLE,
    pwz_commandline: ?[*:0]u16,
    pcch_size: ?*u32,
    pdw_flags: ?*u32,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn ApplicationRecoveryInProgress(
    pb_cancelled: ?*BOOL,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn ApplicationRecoveryFinished(
    b_success: BOOL,
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
// Section: Imports (5)
//--------------------------------------------------------------------------------
const APPLICATION_RECOVERY_CALLBACK = @import("../system/windows_programming.zig").APPLICATION_RECOVERY_CALLBACK;
const BOOL = @import("../foundation.zig").BOOL;
const HANDLE = @import("../foundation.zig").HANDLE;
const HRESULT = @import("../foundation.zig").HRESULT;
const PWSTR = @import("../foundation.zig").PWSTR;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}