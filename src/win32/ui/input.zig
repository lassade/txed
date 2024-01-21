//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (19)
//--------------------------------------------------------------------------------
pub const RAW_INPUT_DATA_COMMAND_FLAGS = enum(u32) {
    HEADER = 268435461,
    INPUT = 268435459,
};
pub const RID_HEADER = RAW_INPUT_DATA_COMMAND_FLAGS.HEADER;
pub const RID_INPUT = RAW_INPUT_DATA_COMMAND_FLAGS.INPUT;

pub const RAW_INPUT_DEVICE_INFO_COMMAND = enum(u32) {
    PREPARSEDDATA = 536870917,
    DEVICENAME = 536870919,
    DEVICEINFO = 536870923,
};
pub const RIDI_PREPARSEDDATA = RAW_INPUT_DEVICE_INFO_COMMAND.PREPARSEDDATA;
pub const RIDI_DEVICENAME = RAW_INPUT_DEVICE_INFO_COMMAND.DEVICENAME;
pub const RIDI_DEVICEINFO = RAW_INPUT_DEVICE_INFO_COMMAND.DEVICEINFO;

pub const RID_DEVICE_INFO_TYPE = enum(u32) {
    MOUSE = 0,
    KEYBOARD = 1,
    HID = 2,
};
pub const RIM_TYPEMOUSE = RID_DEVICE_INFO_TYPE.MOUSE;
pub const RIM_TYPEKEYBOARD = RID_DEVICE_INFO_TYPE.KEYBOARD;
pub const RIM_TYPEHID = RID_DEVICE_INFO_TYPE.HID;

pub const RAWINPUTDEVICE_FLAGS = enum(u32) {
    REMOVE = 1,
    EXCLUDE = 16,
    PAGEONLY = 32,
    NOLEGACY = 48,
    INPUTSINK = 256,
    CAPTUREMOUSE = 512,
    // NOHOTKEYS = 512, this enum value conflicts with CAPTUREMOUSE
    APPKEYS = 1024,
    EXINPUTSINK = 4096,
    DEVNOTIFY = 8192,
    _,
    pub fn initFlags(o: struct {
        REMOVE: u1 = 0,
        EXCLUDE: u1 = 0,
        PAGEONLY: u1 = 0,
        NOLEGACY: u1 = 0,
        INPUTSINK: u1 = 0,
        CAPTUREMOUSE: u1 = 0,
        APPKEYS: u1 = 0,
        EXINPUTSINK: u1 = 0,
        DEVNOTIFY: u1 = 0,
    }) RAWINPUTDEVICE_FLAGS {
        return @as(RAWINPUTDEVICE_FLAGS, @enumFromInt((if (o.REMOVE == 1) @intFromEnum(RAWINPUTDEVICE_FLAGS.REMOVE) else 0) | (if (o.EXCLUDE == 1) @intFromEnum(RAWINPUTDEVICE_FLAGS.EXCLUDE) else 0) | (if (o.PAGEONLY == 1) @intFromEnum(RAWINPUTDEVICE_FLAGS.PAGEONLY) else 0) | (if (o.NOLEGACY == 1) @intFromEnum(RAWINPUTDEVICE_FLAGS.NOLEGACY) else 0) | (if (o.INPUTSINK == 1) @intFromEnum(RAWINPUTDEVICE_FLAGS.INPUTSINK) else 0) | (if (o.CAPTUREMOUSE == 1) @intFromEnum(RAWINPUTDEVICE_FLAGS.CAPTUREMOUSE) else 0) | (if (o.APPKEYS == 1) @intFromEnum(RAWINPUTDEVICE_FLAGS.APPKEYS) else 0) | (if (o.EXINPUTSINK == 1) @intFromEnum(RAWINPUTDEVICE_FLAGS.EXINPUTSINK) else 0) | (if (o.DEVNOTIFY == 1) @intFromEnum(RAWINPUTDEVICE_FLAGS.DEVNOTIFY) else 0)));
    }
};
pub const RIDEV_REMOVE = RAWINPUTDEVICE_FLAGS.REMOVE;
pub const RIDEV_EXCLUDE = RAWINPUTDEVICE_FLAGS.EXCLUDE;
pub const RIDEV_PAGEONLY = RAWINPUTDEVICE_FLAGS.PAGEONLY;
pub const RIDEV_NOLEGACY = RAWINPUTDEVICE_FLAGS.NOLEGACY;
pub const RIDEV_INPUTSINK = RAWINPUTDEVICE_FLAGS.INPUTSINK;
pub const RIDEV_CAPTUREMOUSE = RAWINPUTDEVICE_FLAGS.CAPTUREMOUSE;
pub const RIDEV_NOHOTKEYS = RAWINPUTDEVICE_FLAGS.CAPTUREMOUSE;
pub const RIDEV_APPKEYS = RAWINPUTDEVICE_FLAGS.APPKEYS;
pub const RIDEV_EXINPUTSINK = RAWINPUTDEVICE_FLAGS.EXINPUTSINK;
pub const RIDEV_DEVNOTIFY = RAWINPUTDEVICE_FLAGS.DEVNOTIFY;

// TODO: this type has an InvalidHandleValue of '0', what can Zig do with this information?
pub const HRAWINPUT = *opaque {};

pub const RAWINPUTHEADER = extern struct {
    dwType: u32,
    dwSize: u32,
    hDevice: ?HANDLE,
    wParam: WPARAM,
};

pub const RAWMOUSE = extern struct {
    usFlags: u16,
    Anonymous: extern union {
        ulButtons: u32,
        Anonymous: extern struct {
            usButtonFlags: u16,
            usButtonData: u16,
        },
    },
    ulRawButtons: u32,
    lLastX: i32,
    lLastY: i32,
    ulExtraInformation: u32,
};

pub const RAWKEYBOARD = extern struct {
    MakeCode: u16,
    Flags: u16,
    Reserved: u16,
    VKey: u16,
    Message: u32,
    ExtraInformation: u32,
};

pub const RAWHID = extern struct {
    dwSizeHid: u32,
    dwCount: u32,
    bRawData: [1]u8,
};

pub const RAWINPUT = extern struct {
    header: RAWINPUTHEADER,
    data: extern union {
        mouse: RAWMOUSE,
        keyboard: RAWKEYBOARD,
        hid: RAWHID,
    },
};

pub const RID_DEVICE_INFO_MOUSE = extern struct {
    dwId: u32,
    dwNumberOfButtons: u32,
    dwSampleRate: u32,
    fHasHorizontalWheel: BOOL,
};

pub const RID_DEVICE_INFO_KEYBOARD = extern struct {
    dwType: u32,
    dwSubType: u32,
    dwKeyboardMode: u32,
    dwNumberOfFunctionKeys: u32,
    dwNumberOfIndicators: u32,
    dwNumberOfKeysTotal: u32,
};

pub const RID_DEVICE_INFO_HID = extern struct {
    dwVendorId: u32,
    dwProductId: u32,
    dwVersionNumber: u32,
    usUsagePage: u16,
    usUsage: u16,
};

pub const RID_DEVICE_INFO = extern struct {
    cbSize: u32,
    dwType: RID_DEVICE_INFO_TYPE,
    Anonymous: extern union {
        mouse: RID_DEVICE_INFO_MOUSE,
        keyboard: RID_DEVICE_INFO_KEYBOARD,
        hid: RID_DEVICE_INFO_HID,
    },
};

pub const RAWINPUTDEVICE = extern struct {
    usUsagePage: u16,
    usUsage: u16,
    dwFlags: RAWINPUTDEVICE_FLAGS,
    hwndTarget: ?HWND,
};

pub const RAWINPUTDEVICELIST = extern struct {
    hDevice: ?HANDLE,
    dwType: RID_DEVICE_INFO_TYPE,
};

pub const INPUT_MESSAGE_DEVICE_TYPE = enum(i32) {
    UNAVAILABLE = 0,
    KEYBOARD = 1,
    MOUSE = 2,
    TOUCH = 4,
    PEN = 8,
    TOUCHPAD = 16,
};
pub const IMDT_UNAVAILABLE = INPUT_MESSAGE_DEVICE_TYPE.UNAVAILABLE;
pub const IMDT_KEYBOARD = INPUT_MESSAGE_DEVICE_TYPE.KEYBOARD;
pub const IMDT_MOUSE = INPUT_MESSAGE_DEVICE_TYPE.MOUSE;
pub const IMDT_TOUCH = INPUT_MESSAGE_DEVICE_TYPE.TOUCH;
pub const IMDT_PEN = INPUT_MESSAGE_DEVICE_TYPE.PEN;
pub const IMDT_TOUCHPAD = INPUT_MESSAGE_DEVICE_TYPE.TOUCHPAD;

pub const INPUT_MESSAGE_ORIGIN_ID = enum(i32) {
    UNAVAILABLE = 0,
    HARDWARE = 1,
    INJECTED = 2,
    SYSTEM = 4,
};
pub const IMO_UNAVAILABLE = INPUT_MESSAGE_ORIGIN_ID.UNAVAILABLE;
pub const IMO_HARDWARE = INPUT_MESSAGE_ORIGIN_ID.HARDWARE;
pub const IMO_INJECTED = INPUT_MESSAGE_ORIGIN_ID.INJECTED;
pub const IMO_SYSTEM = INPUT_MESSAGE_ORIGIN_ID.SYSTEM;

pub const INPUT_MESSAGE_SOURCE = extern struct {
    deviceType: INPUT_MESSAGE_DEVICE_TYPE,
    originId: INPUT_MESSAGE_ORIGIN_ID,
};

//--------------------------------------------------------------------------------
// Section: Functions (10)
//--------------------------------------------------------------------------------
// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "user32" fn GetRawInputData(
    h_raw_input: ?HRAWINPUT,
    ui_command: RAW_INPUT_DATA_COMMAND_FLAGS,
    // TODO: what to do with BytesParamIndex 3?
    p_data: ?*anyopaque,
    pcb_size: ?*u32,
    cb_size_header: u32,
) callconv(@import("std").os.windows.WINAPI) u32;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "user32" fn GetRawInputDeviceInfoA(
    h_device: ?HANDLE,
    ui_command: RAW_INPUT_DEVICE_INFO_COMMAND,
    // TODO: what to do with BytesParamIndex 3?
    p_data: ?*anyopaque,
    pcb_size: ?*u32,
) callconv(@import("std").os.windows.WINAPI) u32;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "user32" fn GetRawInputDeviceInfoW(
    h_device: ?HANDLE,
    ui_command: RAW_INPUT_DEVICE_INFO_COMMAND,
    // TODO: what to do with BytesParamIndex 3?
    p_data: ?*anyopaque,
    pcb_size: ?*u32,
) callconv(@import("std").os.windows.WINAPI) u32;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "user32" fn GetRawInputBuffer(
    // TODO: what to do with BytesParamIndex 1?
    p_data: ?*RAWINPUT,
    pcb_size: ?*u32,
    cb_size_header: u32,
) callconv(@import("std").os.windows.WINAPI) u32;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "user32" fn RegisterRawInputDevices(
    p_raw_input_devices: [*]RAWINPUTDEVICE,
    ui_num_devices: u32,
    cb_size: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "user32" fn GetRegisteredRawInputDevices(
    p_raw_input_devices: ?[*]RAWINPUTDEVICE,
    pui_num_devices: ?*u32,
    cb_size: u32,
) callconv(@import("std").os.windows.WINAPI) u32;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "user32" fn GetRawInputDeviceList(
    p_raw_input_device_list: ?[*]RAWINPUTDEVICELIST,
    pui_num_devices: ?*u32,
    cb_size: u32,
) callconv(@import("std").os.windows.WINAPI) u32;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "user32" fn DefRawInputProc(
    pa_raw_input: [*]?*RAWINPUT,
    n_input: i32,
    cb_size_header: u32,
) callconv(@import("std").os.windows.WINAPI) LRESULT;

// TODO: this type is limited to platform 'windows8.0'
pub extern "user32" fn GetCurrentInputMessageSource(
    input_message_source: ?*INPUT_MESSAGE_SOURCE,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows8.0'
pub extern "user32" fn GetCIMSSM(
    input_message_source: ?*INPUT_MESSAGE_SOURCE,
) callconv(@import("std").os.windows.WINAPI) BOOL;

//--------------------------------------------------------------------------------
// Section: Unicode Aliases (1)
//--------------------------------------------------------------------------------
const thismodule = @This();
pub usingnamespace switch (@import("../zig.zig").unicode_mode) {
    .ansi => struct {
        pub const GetRawInputDeviceInfo = thismodule.GetRawInputDeviceInfoA;
    },
    .wide => struct {
        pub const GetRawInputDeviceInfo = thismodule.GetRawInputDeviceInfoW;
    },
    .unspecified => if (@import("builtin").is_test) struct {
        pub const GetRawInputDeviceInfo = *opaque {};
    } else struct {
        pub const GetRawInputDeviceInfo = @compileError("'GetRawInputDeviceInfo' requires that UNICODE be set to true or false in the root module");
    },
};
//--------------------------------------------------------------------------------
// Section: Imports (5)
//--------------------------------------------------------------------------------
const BOOL = @import("../foundation.zig").BOOL;
const HANDLE = @import("../foundation.zig").HANDLE;
const HWND = @import("../foundation.zig").HWND;
const LRESULT = @import("../foundation.zig").LRESULT;
const WPARAM = @import("../foundation.zig").WPARAM;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
//--------------------------------------------------------------------------------
// Section: SubModules (7)
//--------------------------------------------------------------------------------
pub const ime = @import("input/ime.zig");
pub const ink = @import("input/ink.zig");
pub const keyboard_and_mouse = @import("input/keyboard_and_mouse.zig");
pub const pointer = @import("input/pointer.zig");
pub const radial = @import("input/radial.zig");
pub const touch = @import("input/touch.zig");
pub const xbox_controller = @import("input/xbox_controller.zig");
