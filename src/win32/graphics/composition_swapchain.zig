//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (15)
//--------------------------------------------------------------------------------
pub const SystemInterruptTime = extern struct {
    value: u64,
};

pub const PresentationTransform = extern struct {
    M11: f32,
    M12: f32,
    M21: f32,
    M22: f32,
    M31: f32,
    M32: f32,
};

pub const PresentStatisticsKind = enum(i32) {
    PresentStatus = 1,
    CompositionFrame = 2,
    IndependentFlipFrame = 3,
};
pub const PresentStatisticsKind_PresentStatus = PresentStatisticsKind.PresentStatus;
pub const PresentStatisticsKind_CompositionFrame = PresentStatisticsKind.CompositionFrame;
pub const PresentStatisticsKind_IndependentFlipFrame = PresentStatisticsKind.IndependentFlipFrame;

const IID_IPresentationBuffer_Value = Guid.initString("2e217d3a-5abb-4138-9a13-a775593c89ca");
pub const IID_IPresentationBuffer = &IID_IPresentationBuffer_Value;
pub const IPresentationBuffer = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        GetAvailableEvent: *const fn (
            self: *const IPresentationBuffer,
            available_event_handle: ?*?HANDLE,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        IsAvailable: *const fn (
            self: *const IPresentationBuffer,
            is_available: ?*u8,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getAvailableEvent(self: *const T, available_event_handle_: ?*?HANDLE) HRESULT {
                return @as(*const IPresentationBuffer.VTable, @ptrCast(self.vtable)).GetAvailableEvent(@as(*const IPresentationBuffer, @ptrCast(self)), available_event_handle_);
            }
            pub inline fn isAvailable(self: *const T, is_available_: ?*u8) HRESULT {
                return @as(*const IPresentationBuffer.VTable, @ptrCast(self.vtable)).IsAvailable(@as(*const IPresentationBuffer, @ptrCast(self)), is_available_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IPresentationContent_Value = Guid.initString("5668bb79-3d8e-415c-b215-f38020f2d252");
pub const IID_IPresentationContent = &IID_IPresentationContent_Value;
pub const IPresentationContent = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        SetTag: *const fn (
            self: *const IPresentationContent,
            tag: usize,
        ) callconv(@import("std").os.windows.WINAPI) void,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn setTag(self: *const T, tag_: usize) void {
                return @as(*const IPresentationContent.VTable, @ptrCast(self.vtable)).SetTag(@as(*const IPresentationContent, @ptrCast(self)), tag_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IPresentationSurface_Value = Guid.initString("956710fb-ea40-4eba-a3eb-4375a0eb4edc");
pub const IID_IPresentationSurface = &IID_IPresentationSurface_Value;
pub const IPresentationSurface = extern struct {
    pub const VTable = extern struct {
        base: IPresentationContent.VTable,
        SetBuffer: *const fn (
            self: *const IPresentationSurface,
            presentation_buffer: ?*IPresentationBuffer,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetColorSpace: *const fn (
            self: *const IPresentationSurface,
            color_space: DXGI_COLOR_SPACE_TYPE,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetAlphaMode: *const fn (
            self: *const IPresentationSurface,
            alpha_mode: DXGI_ALPHA_MODE,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetSourceRect: *const fn (
            self: *const IPresentationSurface,
            source_rect: ?*const RECT,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetTransform: *const fn (
            self: *const IPresentationSurface,
            transform: ?*PresentationTransform,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RestrictToOutput: *const fn (
            self: *const IPresentationSurface,
            output: ?*IUnknown,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetDisableReadback: *const fn (
            self: *const IPresentationSurface,
            value: u8,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetLetterboxingMargins: *const fn (
            self: *const IPresentationSurface,
            left_letterbox_size: f32,
            top_letterbox_size: f32,
            right_letterbox_size: f32,
            bottom_letterbox_size: f32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IPresentationContent.MethodMixin(T);
            pub inline fn setBuffer(self: *const T, presentation_buffer_: ?*IPresentationBuffer) HRESULT {
                return @as(*const IPresentationSurface.VTable, @ptrCast(self.vtable)).SetBuffer(@as(*const IPresentationSurface, @ptrCast(self)), presentation_buffer_);
            }
            pub inline fn setColorSpace(self: *const T, color_space_: DXGI_COLOR_SPACE_TYPE) HRESULT {
                return @as(*const IPresentationSurface.VTable, @ptrCast(self.vtable)).SetColorSpace(@as(*const IPresentationSurface, @ptrCast(self)), color_space_);
            }
            pub inline fn setAlphaMode(self: *const T, alpha_mode_: DXGI_ALPHA_MODE) HRESULT {
                return @as(*const IPresentationSurface.VTable, @ptrCast(self.vtable)).SetAlphaMode(@as(*const IPresentationSurface, @ptrCast(self)), alpha_mode_);
            }
            pub inline fn setSourceRect(self: *const T, source_rect_: ?*const RECT) HRESULT {
                return @as(*const IPresentationSurface.VTable, @ptrCast(self.vtable)).SetSourceRect(@as(*const IPresentationSurface, @ptrCast(self)), source_rect_);
            }
            pub inline fn setTransform(self: *const T, transform_: ?*PresentationTransform) HRESULT {
                return @as(*const IPresentationSurface.VTable, @ptrCast(self.vtable)).SetTransform(@as(*const IPresentationSurface, @ptrCast(self)), transform_);
            }
            pub inline fn restrictToOutput(self: *const T, output_: ?*IUnknown) HRESULT {
                return @as(*const IPresentationSurface.VTable, @ptrCast(self.vtable)).RestrictToOutput(@as(*const IPresentationSurface, @ptrCast(self)), output_);
            }
            pub inline fn setDisableReadback(self: *const T, value_: u8) HRESULT {
                return @as(*const IPresentationSurface.VTable, @ptrCast(self.vtable)).SetDisableReadback(@as(*const IPresentationSurface, @ptrCast(self)), value_);
            }
            pub inline fn setLetterboxingMargins(self: *const T, left_letterbox_size_: f32, top_letterbox_size_: f32, right_letterbox_size_: f32, bottom_letterbox_size_: f32) HRESULT {
                return @as(*const IPresentationSurface.VTable, @ptrCast(self.vtable)).SetLetterboxingMargins(@as(*const IPresentationSurface, @ptrCast(self)), left_letterbox_size_, top_letterbox_size_, right_letterbox_size_, bottom_letterbox_size_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IPresentStatistics_Value = Guid.initString("b44b8bda-7282-495d-9dd7-ceadd8b4bb86");
pub const IID_IPresentStatistics = &IID_IPresentStatistics_Value;
pub const IPresentStatistics = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        GetPresentId: *const fn (
            self: *const IPresentStatistics,
        ) callconv(@import("std").os.windows.WINAPI) u64,
        GetKind: *const fn (
            self: *const IPresentStatistics,
        ) callconv(@import("std").os.windows.WINAPI) PresentStatisticsKind,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getPresentId(self: *const T) u64 {
                return @as(*const IPresentStatistics.VTable, @ptrCast(self.vtable)).GetPresentId(@as(*const IPresentStatistics, @ptrCast(self)));
            }
            pub inline fn getKind(self: *const T) PresentStatisticsKind {
                return @as(*const IPresentStatistics.VTable, @ptrCast(self.vtable)).GetKind(@as(*const IPresentStatistics, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IPresentationManager_Value = Guid.initString("fb562f82-6292-470a-88b1-843661e7f20c");
pub const IID_IPresentationManager = &IID_IPresentationManager_Value;
pub const IPresentationManager = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        AddBufferFromResource: *const fn (
            self: *const IPresentationManager,
            resource: ?*IUnknown,
            presentation_buffer: ?*?*IPresentationBuffer,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        CreatePresentationSurface: *const fn (
            self: *const IPresentationManager,
            composition_surface_handle: ?HANDLE,
            presentation_surface: ?*?*IPresentationSurface,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetNextPresentId: *const fn (
            self: *const IPresentationManager,
        ) callconv(@import("std").os.windows.WINAPI) u64,
        SetTargetTime: *const fn (
            self: *const IPresentationManager,
            target_time: SystemInterruptTime,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetPreferredPresentDuration: *const fn (
            self: *const IPresentationManager,
            preferred_duration: SystemInterruptTime,
            deviation_tolerance: SystemInterruptTime,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ForceVSyncInterrupt: *const fn (
            self: *const IPresentationManager,
            force_vsync_interrupt: u8,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Present: *const fn (
            self: *const IPresentationManager,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetPresentRetiringFence: *const fn (
            self: *const IPresentationManager,
            riid: ?*const Guid,
            fence: ?*?*anyopaque,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        CancelPresentsFrom: *const fn (
            self: *const IPresentationManager,
            present_id_to_cancel_from: u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetLostEvent: *const fn (
            self: *const IPresentationManager,
            lost_event_handle: ?*?HANDLE,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetPresentStatisticsAvailableEvent: *const fn (
            self: *const IPresentationManager,
            present_statistics_available_event_handle: ?*?HANDLE,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        EnablePresentStatisticsKind: *const fn (
            self: *const IPresentationManager,
            present_statistics_kind: PresentStatisticsKind,
            enabled: u8,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetNextPresentStatistics: *const fn (
            self: *const IPresentationManager,
            next_present_statistics: ?*?*IPresentStatistics,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn addBufferFromResource(self: *const T, resource_: ?*IUnknown, presentation_buffer_: ?*?*IPresentationBuffer) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).AddBufferFromResource(@as(*const IPresentationManager, @ptrCast(self)), resource_, presentation_buffer_);
            }
            pub inline fn createPresentationSurface(self: *const T, composition_surface_handle_: ?HANDLE, presentation_surface_: ?*?*IPresentationSurface) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).CreatePresentationSurface(@as(*const IPresentationManager, @ptrCast(self)), composition_surface_handle_, presentation_surface_);
            }
            pub inline fn getNextPresentId(self: *const T) u64 {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).GetNextPresentId(@as(*const IPresentationManager, @ptrCast(self)));
            }
            pub inline fn setTargetTime(self: *const T, target_time_: SystemInterruptTime) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).SetTargetTime(@as(*const IPresentationManager, @ptrCast(self)), target_time_);
            }
            pub inline fn setPreferredPresentDuration(self: *const T, preferred_duration_: SystemInterruptTime, deviation_tolerance_: SystemInterruptTime) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).SetPreferredPresentDuration(@as(*const IPresentationManager, @ptrCast(self)), preferred_duration_, deviation_tolerance_);
            }
            pub inline fn forceVSyncInterrupt(self: *const T, force_vsync_interrupt_: u8) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).ForceVSyncInterrupt(@as(*const IPresentationManager, @ptrCast(self)), force_vsync_interrupt_);
            }
            pub inline fn present(self: *const T) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).Present(@as(*const IPresentationManager, @ptrCast(self)));
            }
            pub inline fn getPresentRetiringFence(self: *const T, riid_: ?*const Guid, fence_: ?*?*anyopaque) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).GetPresentRetiringFence(@as(*const IPresentationManager, @ptrCast(self)), riid_, fence_);
            }
            pub inline fn cancelPresentsFrom(self: *const T, present_id_to_cancel_from_: u64) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).CancelPresentsFrom(@as(*const IPresentationManager, @ptrCast(self)), present_id_to_cancel_from_);
            }
            pub inline fn getLostEvent(self: *const T, lost_event_handle_: ?*?HANDLE) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).GetLostEvent(@as(*const IPresentationManager, @ptrCast(self)), lost_event_handle_);
            }
            pub inline fn getPresentStatisticsAvailableEvent(self: *const T, present_statistics_available_event_handle_: ?*?HANDLE) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).GetPresentStatisticsAvailableEvent(@as(*const IPresentationManager, @ptrCast(self)), present_statistics_available_event_handle_);
            }
            pub inline fn enablePresentStatisticsKind(self: *const T, present_statistics_kind_: PresentStatisticsKind, enabled_: u8) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).EnablePresentStatisticsKind(@as(*const IPresentationManager, @ptrCast(self)), present_statistics_kind_, enabled_);
            }
            pub inline fn getNextPresentStatistics(self: *const T, next_present_statistics_: ?*?*IPresentStatistics) HRESULT {
                return @as(*const IPresentationManager.VTable, @ptrCast(self.vtable)).GetNextPresentStatistics(@as(*const IPresentationManager, @ptrCast(self)), next_present_statistics_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IPresentationFactory_Value = Guid.initString("8fb37b58-1d74-4f64-a49c-1f97a80a2ec0");
pub const IID_IPresentationFactory = &IID_IPresentationFactory_Value;
pub const IPresentationFactory = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        IsPresentationSupported: *const fn (
            self: *const IPresentationFactory,
        ) callconv(@import("std").os.windows.WINAPI) u8,
        IsPresentationSupportedWithIndependentFlip: *const fn (
            self: *const IPresentationFactory,
        ) callconv(@import("std").os.windows.WINAPI) u8,
        CreatePresentationManager: *const fn (
            self: *const IPresentationFactory,
            pp_presentation_manager: ?*?*IPresentationManager,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn isPresentationSupported(self: *const T) u8 {
                return @as(*const IPresentationFactory.VTable, @ptrCast(self.vtable)).IsPresentationSupported(@as(*const IPresentationFactory, @ptrCast(self)));
            }
            pub inline fn isPresentationSupportedWithIndependentFlip(self: *const T) u8 {
                return @as(*const IPresentationFactory.VTable, @ptrCast(self.vtable)).IsPresentationSupportedWithIndependentFlip(@as(*const IPresentationFactory, @ptrCast(self)));
            }
            pub inline fn createPresentationManager(self: *const T, pp_presentation_manager_: ?*?*IPresentationManager) HRESULT {
                return @as(*const IPresentationFactory.VTable, @ptrCast(self.vtable)).CreatePresentationManager(@as(*const IPresentationFactory, @ptrCast(self)), pp_presentation_manager_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

pub const PresentStatus = enum(i32) {
    Queued = 0,
    Skipped = 1,
    Canceled = 2,
};
pub const PresentStatus_Queued = PresentStatus.Queued;
pub const PresentStatus_Skipped = PresentStatus.Skipped;
pub const PresentStatus_Canceled = PresentStatus.Canceled;

const IID_IPresentStatusPresentStatistics_Value = Guid.initString("c9ed2a41-79cb-435e-964e-c8553055420c");
pub const IID_IPresentStatusPresentStatistics = &IID_IPresentStatusPresentStatistics_Value;
pub const IPresentStatusPresentStatistics = extern struct {
    pub const VTable = extern struct {
        base: IPresentStatistics.VTable,
        GetCompositionFrameId: *const fn (
            self: *const IPresentStatusPresentStatistics,
        ) callconv(@import("std").os.windows.WINAPI) u64,
        GetPresentStatus: *const fn (
            self: *const IPresentStatusPresentStatistics,
        ) callconv(@import("std").os.windows.WINAPI) PresentStatus,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IPresentStatistics.MethodMixin(T);
            pub inline fn getCompositionFrameId(self: *const T) u64 {
                return @as(*const IPresentStatusPresentStatistics.VTable, @ptrCast(self.vtable)).GetCompositionFrameId(@as(*const IPresentStatusPresentStatistics, @ptrCast(self)));
            }
            pub inline fn getPresentStatus(self: *const T) PresentStatus {
                return @as(*const IPresentStatusPresentStatistics.VTable, @ptrCast(self.vtable)).GetPresentStatus(@as(*const IPresentStatusPresentStatistics, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

pub const CompositionFrameInstanceKind = enum(i32) {
    ComposedOnScreen = 0,
    ScanoutOnScreen = 1,
    ComposedToIntermediate = 2,
};
pub const CompositionFrameInstanceKind_ComposedOnScreen = CompositionFrameInstanceKind.ComposedOnScreen;
pub const CompositionFrameInstanceKind_ScanoutOnScreen = CompositionFrameInstanceKind.ScanoutOnScreen;
pub const CompositionFrameInstanceKind_ComposedToIntermediate = CompositionFrameInstanceKind.ComposedToIntermediate;

pub const CompositionFrameDisplayInstance = extern struct {
    displayAdapterLUID: LUID,
    displayVidPnSourceId: u32,
    displayUniqueId: u32,
    renderAdapterLUID: LUID,
    instanceKind: CompositionFrameInstanceKind,
    finalTransform: PresentationTransform,
    requiredCrossAdapterCopy: u8,
    colorSpace: DXGI_COLOR_SPACE_TYPE,
};

const IID_ICompositionFramePresentStatistics_Value = Guid.initString("ab41d127-c101-4c0a-911d-f9f2e9d08e64");
pub const IID_ICompositionFramePresentStatistics = &IID_ICompositionFramePresentStatistics_Value;
pub const ICompositionFramePresentStatistics = extern struct {
    pub const VTable = extern struct {
        base: IPresentStatistics.VTable,
        GetContentTag: *const fn (
            self: *const ICompositionFramePresentStatistics,
        ) callconv(@import("std").os.windows.WINAPI) usize,
        GetCompositionFrameId: *const fn (
            self: *const ICompositionFramePresentStatistics,
        ) callconv(@import("std").os.windows.WINAPI) u64,
        GetDisplayInstanceArray: *const fn (
            self: *const ICompositionFramePresentStatistics,
            display_instance_array_count: ?*u32,
            display_instance_array: ?*const ?*CompositionFrameDisplayInstance,
        ) callconv(@import("std").os.windows.WINAPI) void,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IPresentStatistics.MethodMixin(T);
            pub inline fn getContentTag(self: *const T) usize {
                return @as(*const ICompositionFramePresentStatistics.VTable, @ptrCast(self.vtable)).GetContentTag(@as(*const ICompositionFramePresentStatistics, @ptrCast(self)));
            }
            pub inline fn getCompositionFrameId(self: *const T) u64 {
                return @as(*const ICompositionFramePresentStatistics.VTable, @ptrCast(self.vtable)).GetCompositionFrameId(@as(*const ICompositionFramePresentStatistics, @ptrCast(self)));
            }
            pub inline fn getDisplayInstanceArray(self: *const T, display_instance_array_count_: ?*u32, display_instance_array_: ?*const ?*CompositionFrameDisplayInstance) void {
                return @as(*const ICompositionFramePresentStatistics.VTable, @ptrCast(self.vtable)).GetDisplayInstanceArray(@as(*const ICompositionFramePresentStatistics, @ptrCast(self)), display_instance_array_count_, display_instance_array_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IIndependentFlipFramePresentStatistics_Value = Guid.initString("8c93be27-ad94-4da0-8fd4-2413132d124e");
pub const IID_IIndependentFlipFramePresentStatistics = &IID_IIndependentFlipFramePresentStatistics_Value;
pub const IIndependentFlipFramePresentStatistics = extern struct {
    pub const VTable = extern struct {
        base: IPresentStatistics.VTable,
        GetOutputAdapterLUID: *const fn (
            self: *const IIndependentFlipFramePresentStatistics,
            retval: *LUID,
        ) callconv(@import("std").os.windows.WINAPI) void,
        GetOutputVidPnSourceId: *const fn (
            self: *const IIndependentFlipFramePresentStatistics,
        ) callconv(@import("std").os.windows.WINAPI) u32,
        GetContentTag: *const fn (
            self: *const IIndependentFlipFramePresentStatistics,
        ) callconv(@import("std").os.windows.WINAPI) usize,
        GetDisplayedTime: *const fn (
            self: *const IIndependentFlipFramePresentStatistics,
            retval: *SystemInterruptTime,
        ) callconv(@import("std").os.windows.WINAPI) void,
        GetPresentDuration: *const fn (
            self: *const IIndependentFlipFramePresentStatistics,
            retval: *SystemInterruptTime,
        ) callconv(@import("std").os.windows.WINAPI) void,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IPresentStatistics.MethodMixin(T);
            pub inline fn getOutputAdapterLUID(self: *const T) LUID {
                var retval: LUID = undefined;
                @as(*const IIndependentFlipFramePresentStatistics.VTable, @ptrCast(self.vtable)).GetOutputAdapterLUID(@as(*const IIndependentFlipFramePresentStatistics, @ptrCast(self)), &retval);
                return retval;
            }
            pub inline fn getOutputVidPnSourceId(self: *const T) u32 {
                return @as(*const IIndependentFlipFramePresentStatistics.VTable, @ptrCast(self.vtable)).GetOutputVidPnSourceId(@as(*const IIndependentFlipFramePresentStatistics, @ptrCast(self)));
            }
            pub inline fn getContentTag(self: *const T) usize {
                return @as(*const IIndependentFlipFramePresentStatistics.VTable, @ptrCast(self.vtable)).GetContentTag(@as(*const IIndependentFlipFramePresentStatistics, @ptrCast(self)));
            }
            pub inline fn getDisplayedTime(self: *const T) SystemInterruptTime {
                var retval: SystemInterruptTime = undefined;
                @as(*const IIndependentFlipFramePresentStatistics.VTable, @ptrCast(self.vtable)).GetDisplayedTime(@as(*const IIndependentFlipFramePresentStatistics, @ptrCast(self)), &retval);
                return retval;
            }
            pub inline fn getPresentDuration(self: *const T) SystemInterruptTime {
                var retval: SystemInterruptTime = undefined;
                @as(*const IIndependentFlipFramePresentStatistics.VTable, @ptrCast(self.vtable)).GetPresentDuration(@as(*const IIndependentFlipFramePresentStatistics, @ptrCast(self)), &retval);
                return retval;
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

//--------------------------------------------------------------------------------
// Section: Functions (1)
//--------------------------------------------------------------------------------
pub extern "dcomp" fn CreatePresentationFactory(
    d3d_device: ?*IUnknown,
    riid: ?*const Guid,
    presentation_factory: ?*?*anyopaque,
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
// Section: Imports (8)
//--------------------------------------------------------------------------------
const Guid = @import("../zig.zig").Guid;
const DXGI_ALPHA_MODE = @import("../graphics/dxgi/common.zig").DXGI_ALPHA_MODE;
const DXGI_COLOR_SPACE_TYPE = @import("../graphics/dxgi/common.zig").DXGI_COLOR_SPACE_TYPE;
const HANDLE = @import("../foundation.zig").HANDLE;
const HRESULT = @import("../foundation.zig").HRESULT;
const IUnknown = @import("../system/com.zig").IUnknown;
const LUID = @import("../foundation.zig").LUID;
const RECT = @import("../foundation.zig").RECT;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
