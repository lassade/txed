//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (1)
//--------------------------------------------------------------------------------
pub const E_SURFACE_CONTENTS_LOST = @as(u32, 2150301728);

//--------------------------------------------------------------------------------
// Section: Types (19)
//--------------------------------------------------------------------------------
const IID_ISurfaceImageSourceNative_Value = Guid.initString("f2e9edc1-d307-4525-9886-0fafaa44163c");
pub const IID_ISurfaceImageSourceNative = &IID_ISurfaceImageSourceNative_Value;
pub const ISurfaceImageSourceNative = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        SetDevice: *const fn (
            self: *const ISurfaceImageSourceNative,
            device: ?*IDXGIDevice,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        BeginDraw: *const fn (
            self: *const ISurfaceImageSourceNative,
            update_rect: RECT,
            surface: ?*?*IDXGISurface,
            offset: ?*POINT,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        EndDraw: *const fn (
            self: *const ISurfaceImageSourceNative,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn setDevice(self: *const T, device_: ?*IDXGIDevice) HRESULT {
                return @as(*const ISurfaceImageSourceNative.VTable, @ptrCast(self.vtable)).SetDevice(@as(*const ISurfaceImageSourceNative, @ptrCast(self)), device_);
            }
            pub inline fn beginDraw(self: *const T, update_rect_: RECT, surface_: ?*?*IDXGISurface, offset_: ?*POINT) HRESULT {
                return @as(*const ISurfaceImageSourceNative.VTable, @ptrCast(self.vtable)).BeginDraw(@as(*const ISurfaceImageSourceNative, @ptrCast(self)), update_rect_, surface_, offset_);
            }
            pub inline fn endDraw(self: *const T) HRESULT {
                return @as(*const ISurfaceImageSourceNative.VTable, @ptrCast(self.vtable)).EndDraw(@as(*const ISurfaceImageSourceNative, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IVirtualSurfaceUpdatesCallbackNative_Value = Guid.initString("dbf2e947-8e6c-4254-9eee-7738f71386c9");
pub const IID_IVirtualSurfaceUpdatesCallbackNative = &IID_IVirtualSurfaceUpdatesCallbackNative_Value;
pub const IVirtualSurfaceUpdatesCallbackNative = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        UpdatesNeeded: *const fn (
            self: *const IVirtualSurfaceUpdatesCallbackNative,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn updatesNeeded(self: *const T) HRESULT {
                return @as(*const IVirtualSurfaceUpdatesCallbackNative.VTable, @ptrCast(self.vtable)).UpdatesNeeded(@as(*const IVirtualSurfaceUpdatesCallbackNative, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IVirtualSurfaceImageSourceNative_Value = Guid.initString("e9550983-360b-4f53-b391-afd695078691");
pub const IID_IVirtualSurfaceImageSourceNative = &IID_IVirtualSurfaceImageSourceNative_Value;
pub const IVirtualSurfaceImageSourceNative = extern struct {
    pub const VTable = extern struct {
        base: ISurfaceImageSourceNative.VTable,
        Invalidate: *const fn (
            self: *const IVirtualSurfaceImageSourceNative,
            update_rect: RECT,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetUpdateRectCount: *const fn (
            self: *const IVirtualSurfaceImageSourceNative,
            count: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetUpdateRects: *const fn (
            self: *const IVirtualSurfaceImageSourceNative,
            updates: [*]RECT,
            count: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetVisibleBounds: *const fn (
            self: *const IVirtualSurfaceImageSourceNative,
            bounds: ?*RECT,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RegisterForUpdatesNeeded: *const fn (
            self: *const IVirtualSurfaceImageSourceNative,
            callback: ?*IVirtualSurfaceUpdatesCallbackNative,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Resize: *const fn (
            self: *const IVirtualSurfaceImageSourceNative,
            new_width: i32,
            new_height: i32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace ISurfaceImageSourceNative.MethodMixin(T);
            pub inline fn invalidate(self: *const T, update_rect_: RECT) HRESULT {
                return @as(*const IVirtualSurfaceImageSourceNative.VTable, @ptrCast(self.vtable)).Invalidate(@as(*const IVirtualSurfaceImageSourceNative, @ptrCast(self)), update_rect_);
            }
            pub inline fn getUpdateRectCount(self: *const T, count_: ?*u32) HRESULT {
                return @as(*const IVirtualSurfaceImageSourceNative.VTable, @ptrCast(self.vtable)).GetUpdateRectCount(@as(*const IVirtualSurfaceImageSourceNative, @ptrCast(self)), count_);
            }
            pub inline fn getUpdateRects(self: *const T, updates_: [*]RECT, count_: u32) HRESULT {
                return @as(*const IVirtualSurfaceImageSourceNative.VTable, @ptrCast(self.vtable)).GetUpdateRects(@as(*const IVirtualSurfaceImageSourceNative, @ptrCast(self)), updates_, count_);
            }
            pub inline fn getVisibleBounds(self: *const T, bounds_: ?*RECT) HRESULT {
                return @as(*const IVirtualSurfaceImageSourceNative.VTable, @ptrCast(self.vtable)).GetVisibleBounds(@as(*const IVirtualSurfaceImageSourceNative, @ptrCast(self)), bounds_);
            }
            pub inline fn registerForUpdatesNeeded(self: *const T, callback_: ?*IVirtualSurfaceUpdatesCallbackNative) HRESULT {
                return @as(*const IVirtualSurfaceImageSourceNative.VTable, @ptrCast(self.vtable)).RegisterForUpdatesNeeded(@as(*const IVirtualSurfaceImageSourceNative, @ptrCast(self)), callback_);
            }
            pub inline fn resize(self: *const T, new_width_: i32, new_height_: i32) HRESULT {
                return @as(*const IVirtualSurfaceImageSourceNative.VTable, @ptrCast(self.vtable)).Resize(@as(*const IVirtualSurfaceImageSourceNative, @ptrCast(self)), new_width_, new_height_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISwapChainBackgroundPanelNative_Value = Guid.initString("43bebd4e-add5-4035-8f85-5608d08e9dc9");
pub const IID_ISwapChainBackgroundPanelNative = &IID_ISwapChainBackgroundPanelNative_Value;
pub const ISwapChainBackgroundPanelNative = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        SetSwapChain: *const fn (
            self: *const ISwapChainBackgroundPanelNative,
            swap_chain: ?*IDXGISwapChain,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn setSwapChain(self: *const T, swap_chain_: ?*IDXGISwapChain) HRESULT {
                return @as(*const ISwapChainBackgroundPanelNative.VTable, @ptrCast(self.vtable)).SetSwapChain(@as(*const ISwapChainBackgroundPanelNative, @ptrCast(self)), swap_chain_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISurfaceImageSourceManagerNative_Value = Guid.initString("4c8798b7-1d88-4a0f-b59b-b93f600de8c8");
pub const IID_ISurfaceImageSourceManagerNative = &IID_ISurfaceImageSourceManagerNative_Value;
pub const ISurfaceImageSourceManagerNative = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        FlushAllSurfacesWithDevice: *const fn (
            self: *const ISurfaceImageSourceManagerNative,
            device: ?*IUnknown,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn flushAllSurfacesWithDevice(self: *const T, device_: ?*IUnknown) HRESULT {
                return @as(*const ISurfaceImageSourceManagerNative.VTable, @ptrCast(self.vtable)).FlushAllSurfacesWithDevice(@as(*const ISurfaceImageSourceManagerNative, @ptrCast(self)), device_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISurfaceImageSourceNativeWithD2D_Value = Guid.initString("54298223-41e1-4a41-9c08-02e8256864a1");
pub const IID_ISurfaceImageSourceNativeWithD2D = &IID_ISurfaceImageSourceNativeWithD2D_Value;
pub const ISurfaceImageSourceNativeWithD2D = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        SetDevice: *const fn (
            self: *const ISurfaceImageSourceNativeWithD2D,
            device: ?*IUnknown,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        BeginDraw: *const fn (
            self: *const ISurfaceImageSourceNativeWithD2D,
            update_rect: ?*const RECT,
            iid: ?*const Guid,
            update_object: ?*?*anyopaque,
            offset: ?*POINT,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        EndDraw: *const fn (
            self: *const ISurfaceImageSourceNativeWithD2D,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SuspendDraw: *const fn (
            self: *const ISurfaceImageSourceNativeWithD2D,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ResumeDraw: *const fn (
            self: *const ISurfaceImageSourceNativeWithD2D,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn setDevice(self: *const T, device_: ?*IUnknown) HRESULT {
                return @as(*const ISurfaceImageSourceNativeWithD2D.VTable, @ptrCast(self.vtable)).SetDevice(@as(*const ISurfaceImageSourceNativeWithD2D, @ptrCast(self)), device_);
            }
            pub inline fn beginDraw(self: *const T, update_rect_: ?*const RECT, iid_: ?*const Guid, update_object_: ?*?*anyopaque, offset_: ?*POINT) HRESULT {
                return @as(*const ISurfaceImageSourceNativeWithD2D.VTable, @ptrCast(self.vtable)).BeginDraw(@as(*const ISurfaceImageSourceNativeWithD2D, @ptrCast(self)), update_rect_, iid_, update_object_, offset_);
            }
            pub inline fn endDraw(self: *const T) HRESULT {
                return @as(*const ISurfaceImageSourceNativeWithD2D.VTable, @ptrCast(self.vtable)).EndDraw(@as(*const ISurfaceImageSourceNativeWithD2D, @ptrCast(self)));
            }
            pub inline fn suspendDraw(self: *const T) HRESULT {
                return @as(*const ISurfaceImageSourceNativeWithD2D.VTable, @ptrCast(self.vtable)).SuspendDraw(@as(*const ISurfaceImageSourceNativeWithD2D, @ptrCast(self)));
            }
            pub inline fn resumeDraw(self: *const T) HRESULT {
                return @as(*const ISurfaceImageSourceNativeWithD2D.VTable, @ptrCast(self.vtable)).ResumeDraw(@as(*const ISurfaceImageSourceNativeWithD2D, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISwapChainPanelNative_Value = Guid.initString("f92f19d2-3ade-45a6-a20c-f6f1ea90554b");
pub const IID_ISwapChainPanelNative = &IID_ISwapChainPanelNative_Value;
pub const ISwapChainPanelNative = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        SetSwapChain: *const fn (
            self: *const ISwapChainPanelNative,
            swap_chain: ?*IDXGISwapChain,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn setSwapChain(self: *const T, swap_chain_: ?*IDXGISwapChain) HRESULT {
                return @as(*const ISwapChainPanelNative.VTable, @ptrCast(self.vtable)).SetSwapChain(@as(*const ISwapChainPanelNative, @ptrCast(self)), swap_chain_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISwapChainPanelNative2_Value = Guid.initString("d5a2f60c-37b2-44a2-937b-8d8eb9726821");
pub const IID_ISwapChainPanelNative2 = &IID_ISwapChainPanelNative2_Value;
pub const ISwapChainPanelNative2 = extern struct {
    pub const VTable = extern struct {
        base: ISwapChainPanelNative.VTable,
        SetSwapChainHandle: *const fn (
            self: *const ISwapChainPanelNative2,
            swap_chain_handle: ?HANDLE,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace ISwapChainPanelNative.MethodMixin(T);
            pub inline fn setSwapChainHandle(self: *const T, swap_chain_handle_: ?HANDLE) HRESULT {
                return @as(*const ISwapChainPanelNative2.VTable, @ptrCast(self.vtable)).SetSwapChainHandle(@as(*const ISwapChainPanelNative2, @ptrCast(self)), swap_chain_handle_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IDesktopWindowXamlSourceNative_Value = Guid.initString("3cbcf1bf-2f76-4e9c-96ab-e84b37972554");
pub const IID_IDesktopWindowXamlSourceNative = &IID_IDesktopWindowXamlSourceNative_Value;
pub const IDesktopWindowXamlSourceNative = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        AttachToWindow: *const fn (
            self: *const IDesktopWindowXamlSourceNative,
            parent_wnd: ?HWND,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        get_WindowHandle: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const IDesktopWindowXamlSourceNative,
            h_wnd: ?*?HWND,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn attachToWindow(self: *const T, parent_wnd_: ?HWND) HRESULT {
                return @as(*const IDesktopWindowXamlSourceNative.VTable, @ptrCast(self.vtable)).AttachToWindow(@as(*const IDesktopWindowXamlSourceNative, @ptrCast(self)), parent_wnd_);
            }
            pub inline fn getWindowHandle(self: *const T, h_wnd_: ?*?HWND) HRESULT {
                return @as(*const IDesktopWindowXamlSourceNative.VTable, @ptrCast(self.vtable)).get_WindowHandle(@as(*const IDesktopWindowXamlSourceNative, @ptrCast(self)), h_wnd_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IDesktopWindowXamlSourceNative2_Value = Guid.initString("e3dcd8c7-3057-4692-99c3-7b7720afda31");
pub const IID_IDesktopWindowXamlSourceNative2 = &IID_IDesktopWindowXamlSourceNative2_Value;
pub const IDesktopWindowXamlSourceNative2 = extern struct {
    pub const VTable = extern struct {
        base: IDesktopWindowXamlSourceNative.VTable,
        PreTranslateMessage: *const fn (
            self: *const IDesktopWindowXamlSourceNative2,
            message: ?*const MSG,
            result: ?*BOOL,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IDesktopWindowXamlSourceNative.MethodMixin(T);
            pub inline fn preTranslateMessage(self: *const T, message_: ?*const MSG, result_: ?*BOOL) HRESULT {
                return @as(*const IDesktopWindowXamlSourceNative2.VTable, @ptrCast(self.vtable)).PreTranslateMessage(@as(*const IDesktopWindowXamlSourceNative2, @ptrCast(self)), message_, result_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IReferenceTrackerTarget_Value = Guid.initString("64bd43f8-bfee-4ec4-b7eb-2935158dae21");
pub const IID_IReferenceTrackerTarget = &IID_IReferenceTrackerTarget_Value;
pub const IReferenceTrackerTarget = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        AddRefFromReferenceTracker: *const fn (
            self: *const IReferenceTrackerTarget,
        ) callconv(@import("std").os.windows.WINAPI) u32,
        ReleaseFromReferenceTracker: *const fn (
            self: *const IReferenceTrackerTarget,
        ) callconv(@import("std").os.windows.WINAPI) u32,
        Peg: *const fn (
            self: *const IReferenceTrackerTarget,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Unpeg: *const fn (
            self: *const IReferenceTrackerTarget,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn addRefFromReferenceTracker(self: *const T) u32 {
                return @as(*const IReferenceTrackerTarget.VTable, @ptrCast(self.vtable)).AddRefFromReferenceTracker(@as(*const IReferenceTrackerTarget, @ptrCast(self)));
            }
            pub inline fn releaseFromReferenceTracker(self: *const T) u32 {
                return @as(*const IReferenceTrackerTarget.VTable, @ptrCast(self.vtable)).ReleaseFromReferenceTracker(@as(*const IReferenceTrackerTarget, @ptrCast(self)));
            }
            pub inline fn peg(self: *const T) HRESULT {
                return @as(*const IReferenceTrackerTarget.VTable, @ptrCast(self.vtable)).Peg(@as(*const IReferenceTrackerTarget, @ptrCast(self)));
            }
            pub inline fn unpeg(self: *const T) HRESULT {
                return @as(*const IReferenceTrackerTarget.VTable, @ptrCast(self.vtable)).Unpeg(@as(*const IReferenceTrackerTarget, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IReferenceTracker_Value = Guid.initString("11d3b13a-180e-4789-a8be-7712882893e6");
pub const IID_IReferenceTracker = &IID_IReferenceTracker_Value;
pub const IReferenceTracker = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        ConnectFromTrackerSource: *const fn (
            self: *const IReferenceTracker,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        DisconnectFromTrackerSource: *const fn (
            self: *const IReferenceTracker,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        FindTrackerTargets: *const fn (
            self: *const IReferenceTracker,
            callback: ?*IFindReferenceTargetsCallback,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetReferenceTrackerManager: *const fn (
            self: *const IReferenceTracker,
            value: ?*?*IReferenceTrackerManager,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        AddRefFromTrackerSource: *const fn (
            self: *const IReferenceTracker,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ReleaseFromTrackerSource: *const fn (
            self: *const IReferenceTracker,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        PegFromTrackerSource: *const fn (
            self: *const IReferenceTracker,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn connectFromTrackerSource(self: *const T) HRESULT {
                return @as(*const IReferenceTracker.VTable, @ptrCast(self.vtable)).ConnectFromTrackerSource(@as(*const IReferenceTracker, @ptrCast(self)));
            }
            pub inline fn disconnectFromTrackerSource(self: *const T) HRESULT {
                return @as(*const IReferenceTracker.VTable, @ptrCast(self.vtable)).DisconnectFromTrackerSource(@as(*const IReferenceTracker, @ptrCast(self)));
            }
            pub inline fn findTrackerTargets(self: *const T, callback_: ?*IFindReferenceTargetsCallback) HRESULT {
                return @as(*const IReferenceTracker.VTable, @ptrCast(self.vtable)).FindTrackerTargets(@as(*const IReferenceTracker, @ptrCast(self)), callback_);
            }
            pub inline fn getReferenceTrackerManager(self: *const T, value_: ?*?*IReferenceTrackerManager) HRESULT {
                return @as(*const IReferenceTracker.VTable, @ptrCast(self.vtable)).GetReferenceTrackerManager(@as(*const IReferenceTracker, @ptrCast(self)), value_);
            }
            pub inline fn addRefFromTrackerSource(self: *const T) HRESULT {
                return @as(*const IReferenceTracker.VTable, @ptrCast(self.vtable)).AddRefFromTrackerSource(@as(*const IReferenceTracker, @ptrCast(self)));
            }
            pub inline fn releaseFromTrackerSource(self: *const T) HRESULT {
                return @as(*const IReferenceTracker.VTable, @ptrCast(self.vtable)).ReleaseFromTrackerSource(@as(*const IReferenceTracker, @ptrCast(self)));
            }
            pub inline fn pegFromTrackerSource(self: *const T) HRESULT {
                return @as(*const IReferenceTracker.VTable, @ptrCast(self.vtable)).PegFromTrackerSource(@as(*const IReferenceTracker, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IReferenceTrackerManager_Value = Guid.initString("3cf184b4-7ccb-4dda-8455-7e6ce99a3298");
pub const IID_IReferenceTrackerManager = &IID_IReferenceTrackerManager_Value;
pub const IReferenceTrackerManager = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        ReferenceTrackingStarted: *const fn (
            self: *const IReferenceTrackerManager,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        FindTrackerTargetsCompleted: *const fn (
            self: *const IReferenceTrackerManager,
            find_failed: u8,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ReferenceTrackingCompleted: *const fn (
            self: *const IReferenceTrackerManager,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetReferenceTrackerHost: *const fn (
            self: *const IReferenceTrackerManager,
            value: ?*IReferenceTrackerHost,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn referenceTrackingStarted(self: *const T) HRESULT {
                return @as(*const IReferenceTrackerManager.VTable, @ptrCast(self.vtable)).ReferenceTrackingStarted(@as(*const IReferenceTrackerManager, @ptrCast(self)));
            }
            pub inline fn findTrackerTargetsCompleted(self: *const T, find_failed_: u8) HRESULT {
                return @as(*const IReferenceTrackerManager.VTable, @ptrCast(self.vtable)).FindTrackerTargetsCompleted(@as(*const IReferenceTrackerManager, @ptrCast(self)), find_failed_);
            }
            pub inline fn referenceTrackingCompleted(self: *const T) HRESULT {
                return @as(*const IReferenceTrackerManager.VTable, @ptrCast(self.vtable)).ReferenceTrackingCompleted(@as(*const IReferenceTrackerManager, @ptrCast(self)));
            }
            pub inline fn setReferenceTrackerHost(self: *const T, value_: ?*IReferenceTrackerHost) HRESULT {
                return @as(*const IReferenceTrackerManager.VTable, @ptrCast(self.vtable)).SetReferenceTrackerHost(@as(*const IReferenceTrackerManager, @ptrCast(self)), value_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IFindReferenceTargetsCallback_Value = Guid.initString("04b3486c-4687-4229-8d14-505ab584dd88");
pub const IID_IFindReferenceTargetsCallback = &IID_IFindReferenceTargetsCallback_Value;
pub const IFindReferenceTargetsCallback = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        FoundTrackerTarget: *const fn (
            self: *const IFindReferenceTargetsCallback,
            target: ?*IReferenceTrackerTarget,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn foundTrackerTarget(self: *const T, target_: ?*IReferenceTrackerTarget) HRESULT {
                return @as(*const IFindReferenceTargetsCallback.VTable, @ptrCast(self.vtable)).FoundTrackerTarget(@as(*const IFindReferenceTargetsCallback, @ptrCast(self)), target_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

pub const XAML_REFERENCETRACKER_DISCONNECT = enum(i32) {
    DEFAULT = 0,
    SUSPEND = 1,
};
pub const XAML_REFERENCETRACKER_DISCONNECT_DEFAULT = XAML_REFERENCETRACKER_DISCONNECT.DEFAULT;
pub const XAML_REFERENCETRACKER_DISCONNECT_SUSPEND = XAML_REFERENCETRACKER_DISCONNECT.SUSPEND;

const IID_IReferenceTrackerHost_Value = Guid.initString("29a71c6a-3c42-4416-a39d-e2825a07a773");
pub const IID_IReferenceTrackerHost = &IID_IReferenceTrackerHost_Value;
pub const IReferenceTrackerHost = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        DisconnectUnusedReferenceSources: *const fn (
            self: *const IReferenceTrackerHost,
            options: XAML_REFERENCETRACKER_DISCONNECT,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ReleaseDisconnectedReferenceSources: *const fn (
            self: *const IReferenceTrackerHost,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        NotifyEndOfReferenceTrackingOnThread: *const fn (
            self: *const IReferenceTrackerHost,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetTrackerTarget: *const fn (
            self: *const IReferenceTrackerHost,
            unknown: ?*IUnknown,
            new_reference: ?*?*IReferenceTrackerTarget,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        AddMemoryPressure: *const fn (
            self: *const IReferenceTrackerHost,
            bytes_allocated: u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RemoveMemoryPressure: *const fn (
            self: *const IReferenceTrackerHost,
            bytes_allocated: u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn disconnectUnusedReferenceSources(self: *const T, options_: XAML_REFERENCETRACKER_DISCONNECT) HRESULT {
                return @as(*const IReferenceTrackerHost.VTable, @ptrCast(self.vtable)).DisconnectUnusedReferenceSources(@as(*const IReferenceTrackerHost, @ptrCast(self)), options_);
            }
            pub inline fn releaseDisconnectedReferenceSources(self: *const T) HRESULT {
                return @as(*const IReferenceTrackerHost.VTable, @ptrCast(self.vtable)).ReleaseDisconnectedReferenceSources(@as(*const IReferenceTrackerHost, @ptrCast(self)));
            }
            pub inline fn notifyEndOfReferenceTrackingOnThread(self: *const T) HRESULT {
                return @as(*const IReferenceTrackerHost.VTable, @ptrCast(self.vtable)).NotifyEndOfReferenceTrackingOnThread(@as(*const IReferenceTrackerHost, @ptrCast(self)));
            }
            pub inline fn getTrackerTarget(self: *const T, unknown_: ?*IUnknown, new_reference_: ?*?*IReferenceTrackerTarget) HRESULT {
                return @as(*const IReferenceTrackerHost.VTable, @ptrCast(self.vtable)).GetTrackerTarget(@as(*const IReferenceTrackerHost, @ptrCast(self)), unknown_, new_reference_);
            }
            pub inline fn addMemoryPressure(self: *const T, bytes_allocated_: u64) HRESULT {
                return @as(*const IReferenceTrackerHost.VTable, @ptrCast(self.vtable)).AddMemoryPressure(@as(*const IReferenceTrackerHost, @ptrCast(self)), bytes_allocated_);
            }
            pub inline fn removeMemoryPressure(self: *const T, bytes_allocated_: u64) HRESULT {
                return @as(*const IReferenceTrackerHost.VTable, @ptrCast(self.vtable)).RemoveMemoryPressure(@as(*const IReferenceTrackerHost, @ptrCast(self)), bytes_allocated_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IReferenceTrackerExtension_Value = Guid.initString("4e897caa-59d5-4613-8f8c-f7ebd1f399b0");
pub const IID_IReferenceTrackerExtension = &IID_IReferenceTrackerExtension_Value;
pub const IReferenceTrackerExtension = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
        };
    }
    pub usingnamespace MethodMixin(@This());
};

pub const TrackerHandle__ = extern struct {
    unused: i32,
};

const IID_ITrackerOwner_Value = Guid.initString("eb24c20b-9816-4ac7-8cff-36f67a118f4e");
pub const IID_ITrackerOwner = &IID_ITrackerOwner_Value;
pub const ITrackerOwner = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        CreateTrackerHandle: *const fn (
            self: *const ITrackerOwner,
            return_value: ?*?*TrackerHandle__,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        DeleteTrackerHandle: *const fn (
            self: *const ITrackerOwner,
            handle: ?*TrackerHandle__,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetTrackerValue: *const fn (
            self: *const ITrackerOwner,
            handle: ?*TrackerHandle__,
            value: ?*IUnknown,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        TryGetSafeTrackerValue: *const fn (
            self: *const ITrackerOwner,
            handle: ?*TrackerHandle__,
            return_value: ?*?*IUnknown,
        ) callconv(@import("std").os.windows.WINAPI) u8,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn createTrackerHandle(self: *const T, return_value_: ?*?*TrackerHandle__) HRESULT {
                return @as(*const ITrackerOwner.VTable, @ptrCast(self.vtable)).CreateTrackerHandle(@as(*const ITrackerOwner, @ptrCast(self)), return_value_);
            }
            pub inline fn deleteTrackerHandle(self: *const T, handle_: ?*TrackerHandle__) HRESULT {
                return @as(*const ITrackerOwner.VTable, @ptrCast(self.vtable)).DeleteTrackerHandle(@as(*const ITrackerOwner, @ptrCast(self)), handle_);
            }
            pub inline fn setTrackerValue(self: *const T, handle_: ?*TrackerHandle__, value_: ?*IUnknown) HRESULT {
                return @as(*const ITrackerOwner.VTable, @ptrCast(self.vtable)).SetTrackerValue(@as(*const ITrackerOwner, @ptrCast(self)), handle_, value_);
            }
            pub inline fn tryGetSafeTrackerValue(self: *const T, handle_: ?*TrackerHandle__, return_value_: ?*?*IUnknown) u8 {
                return @as(*const ITrackerOwner.VTable, @ptrCast(self.vtable)).TryGetSafeTrackerValue(@as(*const ITrackerOwner, @ptrCast(self)), handle_, return_value_);
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
// Section: Imports (12)
//--------------------------------------------------------------------------------
const Guid = @import("../../zig.zig").Guid;
const BOOL = @import("../../foundation.zig").BOOL;
const HANDLE = @import("../../foundation.zig").HANDLE;
const HRESULT = @import("../../foundation.zig").HRESULT;
const HWND = @import("../../foundation.zig").HWND;
const IDXGIDevice = @import("../../graphics/dxgi.zig").IDXGIDevice;
const IDXGISurface = @import("../../graphics/dxgi.zig").IDXGISurface;
const IDXGISwapChain = @import("../../graphics/dxgi.zig").IDXGISwapChain;
const IUnknown = @import("../../system/com.zig").IUnknown;
const MSG = @import("../../ui/windows_and_messaging.zig").MSG;
const POINT = @import("../../foundation.zig").POINT;
const RECT = @import("../../foundation.zig").RECT;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
