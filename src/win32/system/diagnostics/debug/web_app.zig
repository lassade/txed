//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (9)
//--------------------------------------------------------------------------------
// TODO: this type is limited to platform 'windows8.0'
const IID_IWebApplicationScriptEvents_Value = Guid.initString("7c3f6998-1567-4bba-b52b-48d32141d613");
pub const IID_IWebApplicationScriptEvents = &IID_IWebApplicationScriptEvents_Value;
pub const IWebApplicationScriptEvents = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        BeforeScriptExecute: *const fn (
            self: *const IWebApplicationScriptEvents,
            html_window: ?*IHTMLWindow2,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ScriptError: *const fn (
            self: *const IWebApplicationScriptEvents,
            html_window: ?*IHTMLWindow2,
            script_error: ?*IActiveScriptError,
            url: ?[*:0]const u16,
            error_handled: BOOL,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn beforeScriptExecute(self: *const T, html_window_: ?*IHTMLWindow2) HRESULT {
                return @as(*const IWebApplicationScriptEvents.VTable, @ptrCast(self.vtable)).BeforeScriptExecute(@as(*const IWebApplicationScriptEvents, @ptrCast(self)), html_window_);
            }
            pub inline fn scriptError(self: *const T, html_window_: ?*IHTMLWindow2, script_error_: ?*IActiveScriptError, url_: ?[*:0]const u16, error_handled_: BOOL) HRESULT {
                return @as(*const IWebApplicationScriptEvents.VTable, @ptrCast(self.vtable)).ScriptError(@as(*const IWebApplicationScriptEvents, @ptrCast(self)), html_window_, script_error_, url_, error_handled_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows8.0'
const IID_IWebApplicationNavigationEvents_Value = Guid.initString("c22615d2-d318-4da2-8422-1fcaf77b10e4");
pub const IID_IWebApplicationNavigationEvents = &IID_IWebApplicationNavigationEvents_Value;
pub const IWebApplicationNavigationEvents = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        BeforeNavigate: *const fn (
            self: *const IWebApplicationNavigationEvents,
            html_window: ?*IHTMLWindow2,
            url: ?[*:0]const u16,
            navigation_flags: u32,
            target_frame_name: ?[*:0]const u16,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        NavigateComplete: *const fn (
            self: *const IWebApplicationNavigationEvents,
            html_window: ?*IHTMLWindow2,
            url: ?[*:0]const u16,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        NavigateError: *const fn (
            self: *const IWebApplicationNavigationEvents,
            html_window: ?*IHTMLWindow2,
            url: ?[*:0]const u16,
            target_frame_name: ?[*:0]const u16,
            status_code: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        DocumentComplete: *const fn (
            self: *const IWebApplicationNavigationEvents,
            html_window: ?*IHTMLWindow2,
            url: ?[*:0]const u16,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        DownloadBegin: *const fn (
            self: *const IWebApplicationNavigationEvents,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        DownloadComplete: *const fn (
            self: *const IWebApplicationNavigationEvents,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn beforeNavigate(self: *const T, html_window_: ?*IHTMLWindow2, url_: ?[*:0]const u16, navigation_flags_: u32, target_frame_name_: ?[*:0]const u16) HRESULT {
                return @as(*const IWebApplicationNavigationEvents.VTable, @ptrCast(self.vtable)).BeforeNavigate(@as(*const IWebApplicationNavigationEvents, @ptrCast(self)), html_window_, url_, navigation_flags_, target_frame_name_);
            }
            pub inline fn navigateComplete(self: *const T, html_window_: ?*IHTMLWindow2, url_: ?[*:0]const u16) HRESULT {
                return @as(*const IWebApplicationNavigationEvents.VTable, @ptrCast(self.vtable)).NavigateComplete(@as(*const IWebApplicationNavigationEvents, @ptrCast(self)), html_window_, url_);
            }
            pub inline fn navigateError(self: *const T, html_window_: ?*IHTMLWindow2, url_: ?[*:0]const u16, target_frame_name_: ?[*:0]const u16, status_code_: u32) HRESULT {
                return @as(*const IWebApplicationNavigationEvents.VTable, @ptrCast(self.vtable)).NavigateError(@as(*const IWebApplicationNavigationEvents, @ptrCast(self)), html_window_, url_, target_frame_name_, status_code_);
            }
            pub inline fn documentComplete(self: *const T, html_window_: ?*IHTMLWindow2, url_: ?[*:0]const u16) HRESULT {
                return @as(*const IWebApplicationNavigationEvents.VTable, @ptrCast(self.vtable)).DocumentComplete(@as(*const IWebApplicationNavigationEvents, @ptrCast(self)), html_window_, url_);
            }
            pub inline fn downloadBegin(self: *const T) HRESULT {
                return @as(*const IWebApplicationNavigationEvents.VTable, @ptrCast(self.vtable)).DownloadBegin(@as(*const IWebApplicationNavigationEvents, @ptrCast(self)));
            }
            pub inline fn downloadComplete(self: *const T) HRESULT {
                return @as(*const IWebApplicationNavigationEvents.VTable, @ptrCast(self.vtable)).DownloadComplete(@as(*const IWebApplicationNavigationEvents, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows8.0'
const IID_IWebApplicationUIEvents_Value = Guid.initString("5b2b3f99-328c-41d5-a6f7-7483ed8e71dd");
pub const IID_IWebApplicationUIEvents = &IID_IWebApplicationUIEvents_Value;
pub const IWebApplicationUIEvents = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        SecurityProblem: *const fn (
            self: *const IWebApplicationUIEvents,
            security_problem: u32,
            result: ?*HRESULT,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn securityProblem(self: *const T, security_problem_: u32, result_: ?*HRESULT) HRESULT {
                return @as(*const IWebApplicationUIEvents.VTable, @ptrCast(self.vtable)).SecurityProblem(@as(*const IWebApplicationUIEvents, @ptrCast(self)), security_problem_, result_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows8.0'
const IID_IWebApplicationUpdateEvents_Value = Guid.initString("3e59e6b7-c652-4daf-ad5e-16feb350cde3");
pub const IID_IWebApplicationUpdateEvents = &IID_IWebApplicationUpdateEvents_Value;
pub const IWebApplicationUpdateEvents = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        OnPaint: *const fn (
            self: *const IWebApplicationUpdateEvents,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        OnCssChanged: *const fn (
            self: *const IWebApplicationUpdateEvents,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn onPaint(self: *const T) HRESULT {
                return @as(*const IWebApplicationUpdateEvents.VTable, @ptrCast(self.vtable)).OnPaint(@as(*const IWebApplicationUpdateEvents, @ptrCast(self)));
            }
            pub inline fn onCssChanged(self: *const T) HRESULT {
                return @as(*const IWebApplicationUpdateEvents.VTable, @ptrCast(self.vtable)).OnCssChanged(@as(*const IWebApplicationUpdateEvents, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows8.0'
const IID_IWebApplicationHost_Value = Guid.initString("cecbd2c3-a3a5-4749-9681-20e9161c6794");
pub const IID_IWebApplicationHost = &IID_IWebApplicationHost_Value;
pub const IWebApplicationHost = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        get_HWND: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const IWebApplicationHost,
            hwnd: ?*?HWND,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        get_Document: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const IWebApplicationHost,
            html_document: ?*?*IHTMLDocument2,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Refresh: *const fn (
            self: *const IWebApplicationHost,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Advise: *const fn (
            self: *const IWebApplicationHost,
            interface_id: ?*const Guid,
            callback: ?*IUnknown,
            cookie: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Unadvise: *const fn (
            self: *const IWebApplicationHost,
            cookie: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getHWND(self: *const T, hwnd_: ?*?HWND) HRESULT {
                return @as(*const IWebApplicationHost.VTable, @ptrCast(self.vtable)).get_HWND(@as(*const IWebApplicationHost, @ptrCast(self)), hwnd_);
            }
            pub inline fn getDocument(self: *const T, html_document_: ?*?*IHTMLDocument2) HRESULT {
                return @as(*const IWebApplicationHost.VTable, @ptrCast(self.vtable)).get_Document(@as(*const IWebApplicationHost, @ptrCast(self)), html_document_);
            }
            pub inline fn refresh(self: *const T) HRESULT {
                return @as(*const IWebApplicationHost.VTable, @ptrCast(self.vtable)).Refresh(@as(*const IWebApplicationHost, @ptrCast(self)));
            }
            pub inline fn advise(self: *const T, interface_id_: ?*const Guid, callback_: ?*IUnknown, cookie_: ?*u32) HRESULT {
                return @as(*const IWebApplicationHost.VTable, @ptrCast(self.vtable)).Advise(@as(*const IWebApplicationHost, @ptrCast(self)), interface_id_, callback_, cookie_);
            }
            pub inline fn unadvise(self: *const T, cookie_: u32) HRESULT {
                return @as(*const IWebApplicationHost.VTable, @ptrCast(self.vtable)).Unadvise(@as(*const IWebApplicationHost, @ptrCast(self)), cookie_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows8.0'
const IID_IWebApplicationActivation_Value = Guid.initString("bcdcd0de-330e-481b-b843-4898a6a8ebac");
pub const IID_IWebApplicationActivation = &IID_IWebApplicationActivation_Value;
pub const IWebApplicationActivation = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        CancelPendingActivation: *const fn (
            self: *const IWebApplicationActivation,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn cancelPendingActivation(self: *const T) HRESULT {
                return @as(*const IWebApplicationActivation.VTable, @ptrCast(self.vtable)).CancelPendingActivation(@as(*const IWebApplicationActivation, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows8.0'
const IID_IWebApplicationAuthoringMode_Value = Guid.initString("720aea93-1964-4db0-b005-29eb9e2b18a9");
pub const IID_IWebApplicationAuthoringMode = &IID_IWebApplicationAuthoringMode_Value;
pub const IWebApplicationAuthoringMode = extern struct {
    pub const VTable = extern struct {
        base: IServiceProvider.VTable,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        get_AuthoringClientBinary: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const IWebApplicationAuthoringMode,
            design_mode_dll_path: ?*?BSTR,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IServiceProvider.MethodMixin(T);
            pub inline fn getAuthoringClientBinary(self: *const T, design_mode_dll_path_: ?*?BSTR) HRESULT {
                return @as(*const IWebApplicationAuthoringMode.VTable, @ptrCast(self.vtable)).get_AuthoringClientBinary(@as(*const IWebApplicationAuthoringMode, @ptrCast(self)), design_mode_dll_path_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

pub const RegisterAuthoringClientFunctionType = *const fn (
    authoring_mode_object: ?*IWebApplicationAuthoringMode,
    host: ?*IWebApplicationHost,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub const UnregisterAuthoringClientFunctionType = *const fn (
    host: ?*IWebApplicationHost,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

//--------------------------------------------------------------------------------
// Section: Functions (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Unicode Aliases (0)
//--------------------------------------------------------------------------------
const thismodule = @This();
pub usingnamespace switch (@import("../../../zig.zig").unicode_mode) {
    .ansi => struct {},
    .wide => struct {},
    .unspecified => if (@import("builtin").is_test) struct {} else struct {},
};
//--------------------------------------------------------------------------------
// Section: Imports (11)
//--------------------------------------------------------------------------------
const Guid = @import("../../../zig.zig").Guid;
const BOOL = @import("../../../foundation.zig").BOOL;
const BSTR = @import("../../../foundation.zig").BSTR;
const HRESULT = @import("../../../foundation.zig").HRESULT;
const HWND = @import("../../../foundation.zig").HWND;
const IActiveScriptError = @import("../../../system/diagnostics/debug.zig").IActiveScriptError;
const IHTMLDocument2 = @import("../../../web/ms_html.zig").IHTMLDocument2;
const IHTMLWindow2 = @import("../../../web/ms_html.zig").IHTMLWindow2;
const IServiceProvider = @import("../../../system/com.zig").IServiceProvider;
const IUnknown = @import("../../../system/com.zig").IUnknown;
const PWSTR = @import("../../../foundation.zig").PWSTR;

test {
    // The following '_ = <FuncPtrType>' lines are a workaround for https://github.com/ziglang/zig/issues/4476
    if (@hasDecl(@This(), "RegisterAuthoringClientFunctionType")) {
        _ = RegisterAuthoringClientFunctionType;
    }
    if (@hasDecl(@This(), "UnregisterAuthoringClientFunctionType")) {
        _ = UnregisterAuthoringClientFunctionType;
    }

    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
