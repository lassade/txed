//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (4)
//--------------------------------------------------------------------------------
pub const COMDB_MIN_PORTS_ARBITRATED = @as(u32, 256);
pub const COMDB_MAX_PORTS_ARBITRATED = @as(u32, 4096);
pub const CDB_REPORT_BITS = @as(u32, 0);
pub const CDB_REPORT_BYTES = @as(u32, 1);

//--------------------------------------------------------------------------------
// Section: Types (1)
//--------------------------------------------------------------------------------
// TODO: this type has an InvalidHandleValue of '0', what can Zig do with this information?
pub const HCOMDB = *opaque {};

//--------------------------------------------------------------------------------
// Section: Functions (7)
//--------------------------------------------------------------------------------
pub extern "msports" fn ComDBOpen(
    p_h_com_d_b: ?*isize,
) callconv(@import("std").os.windows.WINAPI) i32;

pub extern "msports" fn ComDBClose(
    h_com_d_b: ?HCOMDB,
) callconv(@import("std").os.windows.WINAPI) i32;

pub extern "msports" fn ComDBGetCurrentPortUsage(
    h_com_d_b: ?HCOMDB,
    // TODO: what to do with BytesParamIndex 2?
    buffer: ?*u8,
    buffer_size: u32,
    report_type: u32,
    max_ports_reported: ?*u32,
) callconv(@import("std").os.windows.WINAPI) i32;

pub extern "msports" fn ComDBClaimNextFreePort(
    h_com_d_b: ?HCOMDB,
    com_number: ?*u32,
) callconv(@import("std").os.windows.WINAPI) i32;

pub extern "msports" fn ComDBClaimPort(
    h_com_d_b: ?HCOMDB,
    com_number: u32,
    force_claim: BOOL,
    forced: ?*BOOL,
) callconv(@import("std").os.windows.WINAPI) i32;

pub extern "msports" fn ComDBReleasePort(
    h_com_d_b: ?HCOMDB,
    com_number: u32,
) callconv(@import("std").os.windows.WINAPI) i32;

pub extern "msports" fn ComDBResizeDatabase(
    h_com_d_b: ?HCOMDB,
    new_size: u32,
) callconv(@import("std").os.windows.WINAPI) i32;

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
// Section: Imports (1)
//--------------------------------------------------------------------------------
const BOOL = @import("../foundation.zig").BOOL;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}