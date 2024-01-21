//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (4)
//--------------------------------------------------------------------------------
pub const PIPE_UNLIMITED_INSTANCES = @as(u32, 255);
pub const NMPWAIT_WAIT_FOREVER = @as(u32, 4294967295);
pub const NMPWAIT_NOWAIT = @as(u32, 1);
pub const NMPWAIT_USE_DEFAULT_WAIT = @as(u32, 0);

//--------------------------------------------------------------------------------
// Section: Types (1)
//--------------------------------------------------------------------------------
pub const NAMED_PIPE_MODE = enum(u32) {
    WAIT = 0,
    NOWAIT = 1,
    // READMODE_BYTE = 0, this enum value conflicts with WAIT
    READMODE_MESSAGE = 2,
    // CLIENT_END = 0, this enum value conflicts with WAIT
    // SERVER_END = 1, this enum value conflicts with NOWAIT
    // TYPE_BYTE = 0, this enum value conflicts with WAIT
    TYPE_MESSAGE = 4,
    // ACCEPT_REMOTE_CLIENTS = 0, this enum value conflicts with WAIT
    REJECT_REMOTE_CLIENTS = 8,
    _,
    pub fn initFlags(o: struct {
        WAIT: u1 = 0,
        NOWAIT: u1 = 0,
        READMODE_MESSAGE: u1 = 0,
        TYPE_MESSAGE: u1 = 0,
        REJECT_REMOTE_CLIENTS: u1 = 0,
    }) NAMED_PIPE_MODE {
        return @as(NAMED_PIPE_MODE, @enumFromInt((if (o.WAIT == 1) @intFromEnum(NAMED_PIPE_MODE.WAIT) else 0) | (if (o.NOWAIT == 1) @intFromEnum(NAMED_PIPE_MODE.NOWAIT) else 0) | (if (o.READMODE_MESSAGE == 1) @intFromEnum(NAMED_PIPE_MODE.READMODE_MESSAGE) else 0) | (if (o.TYPE_MESSAGE == 1) @intFromEnum(NAMED_PIPE_MODE.TYPE_MESSAGE) else 0) | (if (o.REJECT_REMOTE_CLIENTS == 1) @intFromEnum(NAMED_PIPE_MODE.REJECT_REMOTE_CLIENTS) else 0)));
    }
};
pub const PIPE_WAIT = NAMED_PIPE_MODE.WAIT;
pub const PIPE_NOWAIT = NAMED_PIPE_MODE.NOWAIT;
pub const PIPE_READMODE_BYTE = NAMED_PIPE_MODE.WAIT;
pub const PIPE_READMODE_MESSAGE = NAMED_PIPE_MODE.READMODE_MESSAGE;
pub const PIPE_CLIENT_END = NAMED_PIPE_MODE.WAIT;
pub const PIPE_SERVER_END = NAMED_PIPE_MODE.NOWAIT;
pub const PIPE_TYPE_BYTE = NAMED_PIPE_MODE.WAIT;
pub const PIPE_TYPE_MESSAGE = NAMED_PIPE_MODE.TYPE_MESSAGE;
pub const PIPE_ACCEPT_REMOTE_CLIENTS = NAMED_PIPE_MODE.WAIT;
pub const PIPE_REJECT_REMOTE_CLIENTS = NAMED_PIPE_MODE.REJECT_REMOTE_CLIENTS;

//--------------------------------------------------------------------------------
// Section: Functions (22)
//--------------------------------------------------------------------------------
// TODO: this type is limited to platform 'windows5.0'
pub extern "kernel32" fn CreatePipe(
    h_read_pipe: ?*?HANDLE,
    h_write_pipe: ?*?HANDLE,
    lp_pipe_attributes: ?*SECURITY_ATTRIBUTES,
    n_size: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.0'
pub extern "kernel32" fn ConnectNamedPipe(
    h_named_pipe: ?HANDLE,
    lp_overlapped: ?*OVERLAPPED,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.0'
pub extern "kernel32" fn DisconnectNamedPipe(
    h_named_pipe: ?HANDLE,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.0'
pub extern "kernel32" fn SetNamedPipeHandleState(
    h_named_pipe: ?HANDLE,
    lp_mode: ?*NAMED_PIPE_MODE,
    lp_max_collection_count: ?*u32,
    lp_collect_data_timeout: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.0'
pub extern "kernel32" fn PeekNamedPipe(
    h_named_pipe: ?HANDLE,
    // TODO: what to do with BytesParamIndex 2?
    lp_buffer: ?*anyopaque,
    n_buffer_size: u32,
    lp_bytes_read: ?*u32,
    lp_total_bytes_avail: ?*u32,
    lp_bytes_left_this_message: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.0'
pub extern "kernel32" fn TransactNamedPipe(
    h_named_pipe: ?HANDLE,
    // TODO: what to do with BytesParamIndex 2?
    lp_in_buffer: ?*anyopaque,
    n_in_buffer_size: u32,
    // TODO: what to do with BytesParamIndex 4?
    lp_out_buffer: ?*anyopaque,
    n_out_buffer_size: u32,
    lp_bytes_read: ?*u32,
    lp_overlapped: ?*OVERLAPPED,
) callconv(@import("std").os.windows.WINAPI) BOOL;

pub extern "kernel32" fn CreateNamedPipeW(
    lp_name: ?[*:0]const u16,
    dw_open_mode: FILE_FLAGS_AND_ATTRIBUTES,
    dw_pipe_mode: NAMED_PIPE_MODE,
    n_max_instances: u32,
    n_out_buffer_size: u32,
    n_in_buffer_size: u32,
    n_default_time_out: u32,
    lp_security_attributes: ?*SECURITY_ATTRIBUTES,
) callconv(@import("std").os.windows.WINAPI) ?HANDLE;

pub extern "kernel32" fn WaitNamedPipeW(
    lp_named_pipe_name: ?[*:0]const u16,
    n_time_out: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

pub extern "kernel32" fn GetNamedPipeClientComputerNameW(
    pipe: ?HANDLE,
    // TODO: what to do with BytesParamIndex 2?
    client_computer_name: ?PWSTR,
    client_computer_name_length: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "advapi32" fn ImpersonateNamedPipeClient(
    h_named_pipe: ?HANDLE,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.0'
pub extern "kernel32" fn GetNamedPipeInfo(
    h_named_pipe: ?HANDLE,
    lp_flags: ?*NAMED_PIPE_MODE,
    lp_out_buffer_size: ?*u32,
    lp_in_buffer_size: ?*u32,
    lp_max_instances: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

pub extern "kernel32" fn GetNamedPipeHandleStateW(
    h_named_pipe: ?HANDLE,
    lp_state: ?*NAMED_PIPE_MODE,
    lp_cur_instances: ?*u32,
    lp_max_collection_count: ?*u32,
    lp_collect_data_timeout: ?*u32,
    lp_user_name: ?[*:0]u16,
    n_max_user_name_size: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

pub extern "kernel32" fn CallNamedPipeW(
    lp_named_pipe_name: ?[*:0]const u16,
    // TODO: what to do with BytesParamIndex 2?
    lp_in_buffer: ?*anyopaque,
    n_in_buffer_size: u32,
    // TODO: what to do with BytesParamIndex 4?
    lp_out_buffer: ?*anyopaque,
    n_out_buffer_size: u32,
    lp_bytes_read: ?*u32,
    n_time_out: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.0'
pub extern "kernel32" fn CreateNamedPipeA(
    lp_name: ?[*:0]const u8,
    dw_open_mode: FILE_FLAGS_AND_ATTRIBUTES,
    dw_pipe_mode: NAMED_PIPE_MODE,
    n_max_instances: u32,
    n_out_buffer_size: u32,
    n_in_buffer_size: u32,
    n_default_time_out: u32,
    lp_security_attributes: ?*SECURITY_ATTRIBUTES,
) callconv(@import("std").os.windows.WINAPI) ?HANDLE;

// TODO: this type is limited to platform 'windows5.0'
pub extern "kernel32" fn GetNamedPipeHandleStateA(
    h_named_pipe: ?HANDLE,
    lp_state: ?*NAMED_PIPE_MODE,
    lp_cur_instances: ?*u32,
    lp_max_collection_count: ?*u32,
    lp_collect_data_timeout: ?*u32,
    lp_user_name: ?[*:0]u8,
    n_max_user_name_size: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.0'
pub extern "kernel32" fn CallNamedPipeA(
    lp_named_pipe_name: ?[*:0]const u8,
    // TODO: what to do with BytesParamIndex 2?
    lp_in_buffer: ?*anyopaque,
    n_in_buffer_size: u32,
    // TODO: what to do with BytesParamIndex 4?
    lp_out_buffer: ?*anyopaque,
    n_out_buffer_size: u32,
    lp_bytes_read: ?*u32,
    n_time_out: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.0'
pub extern "kernel32" fn WaitNamedPipeA(
    lp_named_pipe_name: ?[*:0]const u8,
    n_time_out: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn GetNamedPipeClientComputerNameA(
    pipe: ?HANDLE,
    // TODO: what to do with BytesParamIndex 2?
    client_computer_name: ?PSTR,
    client_computer_name_length: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn GetNamedPipeClientProcessId(
    pipe: ?HANDLE,
    client_process_id: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn GetNamedPipeClientSessionId(
    pipe: ?HANDLE,
    client_session_id: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn GetNamedPipeServerProcessId(
    pipe: ?HANDLE,
    server_process_id: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "kernel32" fn GetNamedPipeServerSessionId(
    pipe: ?HANDLE,
    server_session_id: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

//--------------------------------------------------------------------------------
// Section: Unicode Aliases (5)
//--------------------------------------------------------------------------------
const thismodule = @This();
pub usingnamespace switch (@import("../zig.zig").unicode_mode) {
    .ansi => struct {
        pub const CreateNamedPipe = thismodule.CreateNamedPipeA;
        pub const WaitNamedPipe = thismodule.WaitNamedPipeA;
        pub const GetNamedPipeClientComputerName = thismodule.GetNamedPipeClientComputerNameA;
        pub const GetNamedPipeHandleState = thismodule.GetNamedPipeHandleStateA;
        pub const CallNamedPipe = thismodule.CallNamedPipeA;
    },
    .wide => struct {
        pub const CreateNamedPipe = thismodule.CreateNamedPipeW;
        pub const WaitNamedPipe = thismodule.WaitNamedPipeW;
        pub const GetNamedPipeClientComputerName = thismodule.GetNamedPipeClientComputerNameW;
        pub const GetNamedPipeHandleState = thismodule.GetNamedPipeHandleStateW;
        pub const CallNamedPipe = thismodule.CallNamedPipeW;
    },
    .unspecified => if (@import("builtin").is_test) struct {
        pub const CreateNamedPipe = *opaque {};
        pub const WaitNamedPipe = *opaque {};
        pub const GetNamedPipeClientComputerName = *opaque {};
        pub const GetNamedPipeHandleState = *opaque {};
        pub const CallNamedPipe = *opaque {};
    } else struct {
        pub const CreateNamedPipe = @compileError("'CreateNamedPipe' requires that UNICODE be set to true or false in the root module");
        pub const WaitNamedPipe = @compileError("'WaitNamedPipe' requires that UNICODE be set to true or false in the root module");
        pub const GetNamedPipeClientComputerName = @compileError("'GetNamedPipeClientComputerName' requires that UNICODE be set to true or false in the root module");
        pub const GetNamedPipeHandleState = @compileError("'GetNamedPipeHandleState' requires that UNICODE be set to true or false in the root module");
        pub const CallNamedPipe = @compileError("'CallNamedPipe' requires that UNICODE be set to true or false in the root module");
    },
};
//--------------------------------------------------------------------------------
// Section: Imports (7)
//--------------------------------------------------------------------------------
const BOOL = @import("../foundation.zig").BOOL;
const FILE_FLAGS_AND_ATTRIBUTES = @import("../storage/file_system.zig").FILE_FLAGS_AND_ATTRIBUTES;
const HANDLE = @import("../foundation.zig").HANDLE;
const OVERLAPPED = @import("../system/io.zig").OVERLAPPED;
const PSTR = @import("../foundation.zig").PSTR;
const PWSTR = @import("../foundation.zig").PWSTR;
const SECURITY_ATTRIBUTES = @import("../security.zig").SECURITY_ATTRIBUTES;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
