//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (1)
//--------------------------------------------------------------------------------
pub const WSL_DISTRIBUTION_FLAGS = enum(u32) {
    NONE = 0,
    ENABLE_INTEROP = 1,
    APPEND_NT_PATH = 2,
    ENABLE_DRIVE_MOUNTING = 4,
    _,
    pub fn initFlags(o: struct {
        NONE: u1 = 0,
        ENABLE_INTEROP: u1 = 0,
        APPEND_NT_PATH: u1 = 0,
        ENABLE_DRIVE_MOUNTING: u1 = 0,
    }) WSL_DISTRIBUTION_FLAGS {
        return @as(WSL_DISTRIBUTION_FLAGS, @enumFromInt((if (o.NONE == 1) @intFromEnum(WSL_DISTRIBUTION_FLAGS.NONE) else 0) | (if (o.ENABLE_INTEROP == 1) @intFromEnum(WSL_DISTRIBUTION_FLAGS.ENABLE_INTEROP) else 0) | (if (o.APPEND_NT_PATH == 1) @intFromEnum(WSL_DISTRIBUTION_FLAGS.APPEND_NT_PATH) else 0) | (if (o.ENABLE_DRIVE_MOUNTING == 1) @intFromEnum(WSL_DISTRIBUTION_FLAGS.ENABLE_DRIVE_MOUNTING) else 0)));
    }
};
pub const WSL_DISTRIBUTION_FLAGS_NONE = WSL_DISTRIBUTION_FLAGS.NONE;
pub const WSL_DISTRIBUTION_FLAGS_ENABLE_INTEROP = WSL_DISTRIBUTION_FLAGS.ENABLE_INTEROP;
pub const WSL_DISTRIBUTION_FLAGS_APPEND_NT_PATH = WSL_DISTRIBUTION_FLAGS.APPEND_NT_PATH;
pub const WSL_DISTRIBUTION_FLAGS_ENABLE_DRIVE_MOUNTING = WSL_DISTRIBUTION_FLAGS.ENABLE_DRIVE_MOUNTING;

//--------------------------------------------------------------------------------
// Section: Functions (7)
//--------------------------------------------------------------------------------
pub extern "api-ms-win-wsl-api-l1-1-0" fn WslIsDistributionRegistered(
    distribution_name: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) BOOL;

pub extern "api-ms-win-wsl-api-l1-1-0" fn WslRegisterDistribution(
    distribution_name: ?[*:0]const u16,
    tar_gz_filename: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-wsl-api-l1-1-0" fn WslUnregisterDistribution(
    distribution_name: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-wsl-api-l1-1-0" fn WslConfigureDistribution(
    distribution_name: ?[*:0]const u16,
    default_u_i_d: u32,
    wsl_distribution_flags: WSL_DISTRIBUTION_FLAGS,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-wsl-api-l1-1-0" fn WslGetDistributionConfiguration(
    distribution_name: ?[*:0]const u16,
    distribution_version: ?*u32,
    default_u_i_d: ?*u32,
    wsl_distribution_flags: ?*WSL_DISTRIBUTION_FLAGS,
    default_environment_variables: ?*?*?PSTR,
    default_environment_variable_count: ?*u32,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-wsl-api-l1-1-0" fn WslLaunchInteractive(
    distribution_name: ?[*:0]const u16,
    command: ?[*:0]const u16,
    use_current_working_directory: BOOL,
    exit_code: ?*u32,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-wsl-api-l1-1-0" fn WslLaunch(
    distribution_name: ?[*:0]const u16,
    command: ?[*:0]const u16,
    use_current_working_directory: BOOL,
    std_in: ?HANDLE,
    std_out: ?HANDLE,
    std_err: ?HANDLE,
    process: ?*?HANDLE,
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
// Section: Imports (5)
//--------------------------------------------------------------------------------
const BOOL = @import("../foundation.zig").BOOL;
const HANDLE = @import("../foundation.zig").HANDLE;
const HRESULT = @import("../foundation.zig").HRESULT;
const PSTR = @import("../foundation.zig").PSTR;
const PWSTR = @import("../foundation.zig").PWSTR;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
