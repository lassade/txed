//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (26)
//--------------------------------------------------------------------------------
pub const SIDESHOW_ENDPOINT_SIMPLE_CONTENT_FORMAT = Guid.initString("a9a5353f-2d4b-47ce-93ee-759f3a7dda4f");
pub const SIDESHOW_ENDPOINT_ICAL = Guid.initString("4dff36b5-9dde-4f76-9a2a-96435047063d");
pub const SIDESHOW_CAPABILITY_DEVICE_PROPERTIES = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99");
pub const SIDESHOW_CAPABILITY_DEVICE_ID = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 1 };
pub const SIDESHOW_CAPABILITY_SCREEN_TYPE = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 2 };
pub const SIDESHOW_CAPABILITY_SCREEN_WIDTH = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 3 };
pub const SIDESHOW_CAPABILITY_SCREEN_HEIGHT = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 4 };
pub const SIDESHOW_CAPABILITY_COLOR_DEPTH = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 5 };
pub const SIDESHOW_CAPABILITY_COLOR_TYPE = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 6 };
pub const SIDESHOW_CAPABILITY_DATA_CACHE = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 7 };
pub const SIDESHOW_CAPABILITY_SUPPORTED_LANGUAGES = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 8 };
pub const SIDESHOW_CAPABILITY_CURRENT_LANGUAGE = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 9 };
pub const SIDESHOW_CAPABILITY_SUPPORTED_THEMES = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 10 };
pub const SIDESHOW_CAPABILITY_SUPPORTED_IMAGE_FORMATS = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 14 };
pub const SIDESHOW_CAPABILITY_CLIENT_AREA_WIDTH = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 15 };
pub const SIDESHOW_CAPABILITY_CLIENT_AREA_HEIGHT = PROPERTYKEY{ .fmtid = Guid.initString("8abc88a8-857b-4ad7-a35a-b5942f492b99"), .pid = 16 };
pub const GUID_DEVINTERFACE_SIDESHOW = Guid.initString("152e5811-feb9-4b00-90f4-d32947ae1681");
pub const SIDESHOW_CONTENT_MISSING_EVENT = Guid.initString("5007fba8-d313-439f-bea2-a50201d3e9a8");
pub const SIDESHOW_APPLICATION_EVENT = Guid.initString("4cb572fa-1d3b-49b3-a17a-2e6bff052854");
pub const SIDESHOW_USER_CHANGE_REQUEST_EVENT = Guid.initString("5009673c-3f7d-4c7e-9971-eaa2e91f1575");
pub const SIDESHOW_NEW_EVENT_DATA_AVAILABLE = Guid.initString("57813854-2fc1-411c-a59f-f24927608804");
pub const CONTENT_ID_GLANCE = @as(u32, 0);
pub const SIDESHOW_EVENTID_APPLICATION_ENTER = @as(u32, 4294901760);
pub const SIDESHOW_EVENTID_APPLICATION_EXIT = @as(u32, 4294901761);
pub const CONTENT_ID_HOME = @as(u32, 1);
pub const VERSION_1_WINDOWS_7 = @as(u32, 0);

//--------------------------------------------------------------------------------
// Section: Types (28)
//--------------------------------------------------------------------------------
const CLSID_SideShowSession_Value = Guid.initString("e20543b9-f785-4ea2-981e-c4ffa76bbc7c");
pub const CLSID_SideShowSession = &CLSID_SideShowSession_Value;

const CLSID_SideShowNotification_Value = Guid.initString("0ce3e86f-d5cd-4525-a766-1abab1a752f5");
pub const CLSID_SideShowNotification = &CLSID_SideShowNotification_Value;

const CLSID_SideShowKeyCollection_Value = Guid.initString("dfbbdbf8-18de-49b8-83dc-ebc727c62d94");
pub const CLSID_SideShowKeyCollection = &CLSID_SideShowKeyCollection_Value;

const CLSID_SideShowPropVariantCollection_Value = Guid.initString("e640f415-539e-4923-96cd-5f093bc250cd");
pub const CLSID_SideShowPropVariantCollection = &CLSID_SideShowPropVariantCollection_Value;

const IID_ISideShowSession_Value = Guid.initString("e22331ee-9e7d-4922-9fc2-ab7aa41ce491");
pub const IID_ISideShowSession = &IID_ISideShowSession_Value;
pub const ISideShowSession = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        RegisterContent: *const fn (
            self: *const ISideShowSession,
            in_application_id: ?*Guid,
            in_endpoint_id: ?*Guid,
            out_pp_i_content: ?*?*ISideShowContentManager,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RegisterNotifications: *const fn (
            self: *const ISideShowSession,
            in_application_id: ?*Guid,
            out_pp_i_notification: ?*?*ISideShowNotificationManager,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn registerContent(self: *const T, in_application_id_: ?*Guid, in_endpoint_id_: ?*Guid, out_pp_i_content_: ?*?*ISideShowContentManager) HRESULT {
                return @as(*const ISideShowSession.VTable, @ptrCast(self.vtable)).RegisterContent(@as(*const ISideShowSession, @ptrCast(self)), in_application_id_, in_endpoint_id_, out_pp_i_content_);
            }
            pub inline fn registerNotifications(self: *const T, in_application_id_: ?*Guid, out_pp_i_notification_: ?*?*ISideShowNotificationManager) HRESULT {
                return @as(*const ISideShowSession.VTable, @ptrCast(self.vtable)).RegisterNotifications(@as(*const ISideShowSession, @ptrCast(self)), in_application_id_, out_pp_i_notification_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISideShowNotificationManager_Value = Guid.initString("63cea909-f2b9-4302-b5e1-c68e6d9ab833");
pub const IID_ISideShowNotificationManager = &IID_ISideShowNotificationManager_Value;
pub const ISideShowNotificationManager = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        Show: *const fn (
            self: *const ISideShowNotificationManager,
            in_p_i_notification: ?*ISideShowNotification,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Revoke: *const fn (
            self: *const ISideShowNotificationManager,
            in_notification_id: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RevokeAll: *const fn (
            self: *const ISideShowNotificationManager,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn show(self: *const T, in_p_i_notification_: ?*ISideShowNotification) HRESULT {
                return @as(*const ISideShowNotificationManager.VTable, @ptrCast(self.vtable)).Show(@as(*const ISideShowNotificationManager, @ptrCast(self)), in_p_i_notification_);
            }
            pub inline fn revoke(self: *const T, in_notification_id_: u32) HRESULT {
                return @as(*const ISideShowNotificationManager.VTable, @ptrCast(self.vtable)).Revoke(@as(*const ISideShowNotificationManager, @ptrCast(self)), in_notification_id_);
            }
            pub inline fn revokeAll(self: *const T) HRESULT {
                return @as(*const ISideShowNotificationManager.VTable, @ptrCast(self.vtable)).RevokeAll(@as(*const ISideShowNotificationManager, @ptrCast(self)));
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISideShowNotification_Value = Guid.initString("03c93300-8ab2-41c5-9b79-46127a30e148");
pub const IID_ISideShowNotification = &IID_ISideShowNotification_Value;
pub const ISideShowNotification = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        get_NotificationId: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowNotification,
            out_p_notification_id: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        put_NotificationId: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowNotification,
            in_notification_id: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        get_Title: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowNotification,
            out_ppwsz_title: ?*?PWSTR,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        put_Title: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowNotification,
            in_pwsz_title: ?PWSTR,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        get_Message: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowNotification,
            out_ppwsz_message: ?*?PWSTR,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        put_Message: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowNotification,
            in_pwsz_message: ?PWSTR,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        get_Image: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowNotification,
            out_ph_icon: ?*?HICON,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        put_Image: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowNotification,
            in_h_icon: ?HICON,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        get_ExpirationTime: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowNotification,
            out_p_time: ?*SYSTEMTIME,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        put_ExpirationTime: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowNotification,
            in_p_time: ?*SYSTEMTIME,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getNotificationId(self: *const T, out_p_notification_id_: ?*u32) HRESULT {
                return @as(*const ISideShowNotification.VTable, @ptrCast(self.vtable)).get_NotificationId(@as(*const ISideShowNotification, @ptrCast(self)), out_p_notification_id_);
            }
            pub inline fn putNotificationId(self: *const T, in_notification_id_: u32) HRESULT {
                return @as(*const ISideShowNotification.VTable, @ptrCast(self.vtable)).put_NotificationId(@as(*const ISideShowNotification, @ptrCast(self)), in_notification_id_);
            }
            pub inline fn getTitle(self: *const T, out_ppwsz_title_: ?*?PWSTR) HRESULT {
                return @as(*const ISideShowNotification.VTable, @ptrCast(self.vtable)).get_Title(@as(*const ISideShowNotification, @ptrCast(self)), out_ppwsz_title_);
            }
            pub inline fn putTitle(self: *const T, in_pwsz_title_: ?PWSTR) HRESULT {
                return @as(*const ISideShowNotification.VTable, @ptrCast(self.vtable)).put_Title(@as(*const ISideShowNotification, @ptrCast(self)), in_pwsz_title_);
            }
            pub inline fn getMessage(self: *const T, out_ppwsz_message_: ?*?PWSTR) HRESULT {
                return @as(*const ISideShowNotification.VTable, @ptrCast(self.vtable)).get_Message(@as(*const ISideShowNotification, @ptrCast(self)), out_ppwsz_message_);
            }
            pub inline fn putMessage(self: *const T, in_pwsz_message_: ?PWSTR) HRESULT {
                return @as(*const ISideShowNotification.VTable, @ptrCast(self.vtable)).put_Message(@as(*const ISideShowNotification, @ptrCast(self)), in_pwsz_message_);
            }
            pub inline fn getImage(self: *const T, out_ph_icon_: ?*?HICON) HRESULT {
                return @as(*const ISideShowNotification.VTable, @ptrCast(self.vtable)).get_Image(@as(*const ISideShowNotification, @ptrCast(self)), out_ph_icon_);
            }
            pub inline fn putImage(self: *const T, in_h_icon_: ?HICON) HRESULT {
                return @as(*const ISideShowNotification.VTable, @ptrCast(self.vtable)).put_Image(@as(*const ISideShowNotification, @ptrCast(self)), in_h_icon_);
            }
            pub inline fn getExpirationTime(self: *const T, out_p_time_: ?*SYSTEMTIME) HRESULT {
                return @as(*const ISideShowNotification.VTable, @ptrCast(self.vtable)).get_ExpirationTime(@as(*const ISideShowNotification, @ptrCast(self)), out_p_time_);
            }
            pub inline fn putExpirationTime(self: *const T, in_p_time_: ?*SYSTEMTIME) HRESULT {
                return @as(*const ISideShowNotification.VTable, @ptrCast(self.vtable)).put_ExpirationTime(@as(*const ISideShowNotification, @ptrCast(self)), in_p_time_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISideShowContentManager_Value = Guid.initString("a5d5b66b-eef9-41db-8d7e-e17c33ab10b0");
pub const IID_ISideShowContentManager = &IID_ISideShowContentManager_Value;
pub const ISideShowContentManager = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        Add: *const fn (
            self: *const ISideShowContentManager,
            in_p_i_content: ?*ISideShowContent,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Remove: *const fn (
            self: *const ISideShowContentManager,
            in_content_id: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RemoveAll: *const fn (
            self: *const ISideShowContentManager,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetEventSink: *const fn (
            self: *const ISideShowContentManager,
            in_p_i_events: ?*ISideShowEvents,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetDeviceCapabilities: *const fn (
            self: *const ISideShowContentManager,
            out_pp_collection: ?*?*ISideShowCapabilitiesCollection,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn add(self: *const T, in_p_i_content_: ?*ISideShowContent) HRESULT {
                return @as(*const ISideShowContentManager.VTable, @ptrCast(self.vtable)).Add(@as(*const ISideShowContentManager, @ptrCast(self)), in_p_i_content_);
            }
            pub inline fn remove(self: *const T, in_content_id_: u32) HRESULT {
                return @as(*const ISideShowContentManager.VTable, @ptrCast(self.vtable)).Remove(@as(*const ISideShowContentManager, @ptrCast(self)), in_content_id_);
            }
            pub inline fn removeAll(self: *const T) HRESULT {
                return @as(*const ISideShowContentManager.VTable, @ptrCast(self.vtable)).RemoveAll(@as(*const ISideShowContentManager, @ptrCast(self)));
            }
            pub inline fn setEventSink(self: *const T, in_p_i_events_: ?*ISideShowEvents) HRESULT {
                return @as(*const ISideShowContentManager.VTable, @ptrCast(self.vtable)).SetEventSink(@as(*const ISideShowContentManager, @ptrCast(self)), in_p_i_events_);
            }
            pub inline fn getDeviceCapabilities(self: *const T, out_pp_collection_: ?*?*ISideShowCapabilitiesCollection) HRESULT {
                return @as(*const ISideShowContentManager.VTable, @ptrCast(self.vtable)).GetDeviceCapabilities(@as(*const ISideShowContentManager, @ptrCast(self)), out_pp_collection_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISideShowContent_Value = Guid.initString("c18552ed-74ff-4fec-be07-4cfed29d4887");
pub const IID_ISideShowContent = &IID_ISideShowContent_Value;
pub const ISideShowContent = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        GetContent: *const fn (
            self: *const ISideShowContent,
            in_p_i_capabilities: ?*ISideShowCapabilities,
            out_pdw_size: ?*u32,
            out_ppb_data: [*]?*u8,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        get_ContentId: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowContent,
            out_pcontent_id: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        // TODO: this function has a "SpecialName", should Zig do anything with this?
        get_DifferentiateContent: *const fn (
            // TODO: this function has a "SpecialName", should Zig do anything with this?
            self: *const ISideShowContent,
            out_pf_differentiate_content: ?*BOOL,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getContent(self: *const T, in_p_i_capabilities_: ?*ISideShowCapabilities, out_pdw_size_: ?*u32, out_ppb_data_: [*]?*u8) HRESULT {
                return @as(*const ISideShowContent.VTable, @ptrCast(self.vtable)).GetContent(@as(*const ISideShowContent, @ptrCast(self)), in_p_i_capabilities_, out_pdw_size_, out_ppb_data_);
            }
            pub inline fn getContentId(self: *const T, out_pcontent_id_: ?*u32) HRESULT {
                return @as(*const ISideShowContent.VTable, @ptrCast(self.vtable)).get_ContentId(@as(*const ISideShowContent, @ptrCast(self)), out_pcontent_id_);
            }
            pub inline fn getDifferentiateContent(self: *const T, out_pf_differentiate_content_: ?*BOOL) HRESULT {
                return @as(*const ISideShowContent.VTable, @ptrCast(self.vtable)).get_DifferentiateContent(@as(*const ISideShowContent, @ptrCast(self)), out_pf_differentiate_content_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISideShowEvents_Value = Guid.initString("61feca4c-deb4-4a7e-8d75-51f1132d615b");
pub const IID_ISideShowEvents = &IID_ISideShowEvents_Value;
pub const ISideShowEvents = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        ContentMissing: *const fn (
            self: *const ISideShowEvents,
            in_content_id: u32,
            out_pp_i_content: ?*?*ISideShowContent,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ApplicationEvent: *const fn (
            self: *const ISideShowEvents,
            in_p_i_capabilities: ?*ISideShowCapabilities,
            in_dw_event_id: u32,
            in_dw_event_size: u32,
            in_pb_event_data: ?[*:0]const u8,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        DeviceAdded: *const fn (
            self: *const ISideShowEvents,
            in_p_i_device: ?*ISideShowCapabilities,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        DeviceRemoved: *const fn (
            self: *const ISideShowEvents,
            in_p_i_device: ?*ISideShowCapabilities,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn contentMissing(self: *const T, in_content_id_: u32, out_pp_i_content_: ?*?*ISideShowContent) HRESULT {
                return @as(*const ISideShowEvents.VTable, @ptrCast(self.vtable)).ContentMissing(@as(*const ISideShowEvents, @ptrCast(self)), in_content_id_, out_pp_i_content_);
            }
            pub inline fn applicationEvent(self: *const T, in_p_i_capabilities_: ?*ISideShowCapabilities, in_dw_event_id_: u32, in_dw_event_size_: u32, in_pb_event_data_: ?[*:0]const u8) HRESULT {
                return @as(*const ISideShowEvents.VTable, @ptrCast(self.vtable)).ApplicationEvent(@as(*const ISideShowEvents, @ptrCast(self)), in_p_i_capabilities_, in_dw_event_id_, in_dw_event_size_, in_pb_event_data_);
            }
            pub inline fn deviceAdded(self: *const T, in_p_i_device_: ?*ISideShowCapabilities) HRESULT {
                return @as(*const ISideShowEvents.VTable, @ptrCast(self.vtable)).DeviceAdded(@as(*const ISideShowEvents, @ptrCast(self)), in_p_i_device_);
            }
            pub inline fn deviceRemoved(self: *const T, in_p_i_device_: ?*ISideShowCapabilities) HRESULT {
                return @as(*const ISideShowEvents.VTable, @ptrCast(self.vtable)).DeviceRemoved(@as(*const ISideShowEvents, @ptrCast(self)), in_p_i_device_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISideShowCapabilities_Value = Guid.initString("535e1379-c09e-4a54-a511-597bab3a72b8");
pub const IID_ISideShowCapabilities = &IID_ISideShowCapabilities_Value;
pub const ISideShowCapabilities = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        GetCapability: *const fn (
            self: *const ISideShowCapabilities,
            in_key_capability: ?*const PROPERTYKEY,
            inout_p_value: ?*PROPVARIANT,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getCapability(self: *const T, in_key_capability_: ?*const PROPERTYKEY, inout_p_value_: ?*PROPVARIANT) HRESULT {
                return @as(*const ISideShowCapabilities.VTable, @ptrCast(self.vtable)).GetCapability(@as(*const ISideShowCapabilities, @ptrCast(self)), in_key_capability_, inout_p_value_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISideShowCapabilitiesCollection_Value = Guid.initString("50305597-5e0d-4ff7-b3af-33d0d9bd52dd");
pub const IID_ISideShowCapabilitiesCollection = &IID_ISideShowCapabilitiesCollection_Value;
pub const ISideShowCapabilitiesCollection = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        GetCount: *const fn (
            self: *const ISideShowCapabilitiesCollection,
            out_pdw_count: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetAt: *const fn (
            self: *const ISideShowCapabilitiesCollection,
            in_dw_index: u32,
            out_pp_capabilities: ?*?*ISideShowCapabilities,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getCount(self: *const T, out_pdw_count_: ?*u32) HRESULT {
                return @as(*const ISideShowCapabilitiesCollection.VTable, @ptrCast(self.vtable)).GetCount(@as(*const ISideShowCapabilitiesCollection, @ptrCast(self)), out_pdw_count_);
            }
            pub inline fn getAt(self: *const T, in_dw_index_: u32, out_pp_capabilities_: ?*?*ISideShowCapabilities) HRESULT {
                return @as(*const ISideShowCapabilitiesCollection.VTable, @ptrCast(self.vtable)).GetAt(@as(*const ISideShowCapabilitiesCollection, @ptrCast(self)), in_dw_index_, out_pp_capabilities_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISideShowBulkCapabilities_Value = Guid.initString("3a2b7fbc-3ad5-48bd-bbf1-0e6cfbd10807");
pub const IID_ISideShowBulkCapabilities = &IID_ISideShowBulkCapabilities_Value;
pub const ISideShowBulkCapabilities = extern struct {
    pub const VTable = extern struct {
        base: ISideShowCapabilities.VTable,
        GetCapabilities: *const fn (
            self: *const ISideShowBulkCapabilities,
            in_key_collection: ?*ISideShowKeyCollection,
            inout_p_values: ?*?*ISideShowPropVariantCollection,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace ISideShowCapabilities.MethodMixin(T);
            pub inline fn getCapabilities(self: *const T, in_key_collection_: ?*ISideShowKeyCollection, inout_p_values_: ?*?*ISideShowPropVariantCollection) HRESULT {
                return @as(*const ISideShowBulkCapabilities.VTable, @ptrCast(self.vtable)).GetCapabilities(@as(*const ISideShowBulkCapabilities, @ptrCast(self)), in_key_collection_, inout_p_values_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISideShowKeyCollection_Value = Guid.initString("045473bc-a37b-4957-b144-68105411ed8e");
pub const IID_ISideShowKeyCollection = &IID_ISideShowKeyCollection_Value;
pub const ISideShowKeyCollection = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        Add: *const fn (
            self: *const ISideShowKeyCollection,
            key: ?*const PROPERTYKEY,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Clear: *const fn (
            self: *const ISideShowKeyCollection,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetAt: *const fn (
            self: *const ISideShowKeyCollection,
            dw_index: u32,
            p_key: ?*PROPERTYKEY,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetCount: *const fn (
            self: *const ISideShowKeyCollection,
            pc_elems: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RemoveAt: *const fn (
            self: *const ISideShowKeyCollection,
            dw_index: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn add(self: *const T, key_: ?*const PROPERTYKEY) HRESULT {
                return @as(*const ISideShowKeyCollection.VTable, @ptrCast(self.vtable)).Add(@as(*const ISideShowKeyCollection, @ptrCast(self)), key_);
            }
            pub inline fn clear(self: *const T) HRESULT {
                return @as(*const ISideShowKeyCollection.VTable, @ptrCast(self.vtable)).Clear(@as(*const ISideShowKeyCollection, @ptrCast(self)));
            }
            pub inline fn getAt(self: *const T, dw_index_: u32, p_key_: ?*PROPERTYKEY) HRESULT {
                return @as(*const ISideShowKeyCollection.VTable, @ptrCast(self.vtable)).GetAt(@as(*const ISideShowKeyCollection, @ptrCast(self)), dw_index_, p_key_);
            }
            pub inline fn getCount(self: *const T, pc_elems_: ?*u32) HRESULT {
                return @as(*const ISideShowKeyCollection.VTable, @ptrCast(self.vtable)).GetCount(@as(*const ISideShowKeyCollection, @ptrCast(self)), pc_elems_);
            }
            pub inline fn removeAt(self: *const T, dw_index_: u32) HRESULT {
                return @as(*const ISideShowKeyCollection.VTable, @ptrCast(self.vtable)).RemoveAt(@as(*const ISideShowKeyCollection, @ptrCast(self)), dw_index_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_ISideShowPropVariantCollection_Value = Guid.initString("2ea7a549-7bff-4aae-bab0-22d43111de49");
pub const IID_ISideShowPropVariantCollection = &IID_ISideShowPropVariantCollection_Value;
pub const ISideShowPropVariantCollection = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        Add: *const fn (
            self: *const ISideShowPropVariantCollection,
            p_value: ?*const PROPVARIANT,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        Clear: *const fn (
            self: *const ISideShowPropVariantCollection,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetAt: *const fn (
            self: *const ISideShowPropVariantCollection,
            dw_index: u32,
            p_value: ?*PROPVARIANT,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetCount: *const fn (
            self: *const ISideShowPropVariantCollection,
            pc_elems: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RemoveAt: *const fn (
            self: *const ISideShowPropVariantCollection,
            dw_index: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn add(self: *const T, p_value_: ?*const PROPVARIANT) HRESULT {
                return @as(*const ISideShowPropVariantCollection.VTable, @ptrCast(self.vtable)).Add(@as(*const ISideShowPropVariantCollection, @ptrCast(self)), p_value_);
            }
            pub inline fn clear(self: *const T) HRESULT {
                return @as(*const ISideShowPropVariantCollection.VTable, @ptrCast(self.vtable)).Clear(@as(*const ISideShowPropVariantCollection, @ptrCast(self)));
            }
            pub inline fn getAt(self: *const T, dw_index_: u32, p_value_: ?*PROPVARIANT) HRESULT {
                return @as(*const ISideShowPropVariantCollection.VTable, @ptrCast(self.vtable)).GetAt(@as(*const ISideShowPropVariantCollection, @ptrCast(self)), dw_index_, p_value_);
            }
            pub inline fn getCount(self: *const T, pc_elems_: ?*u32) HRESULT {
                return @as(*const ISideShowPropVariantCollection.VTable, @ptrCast(self.vtable)).GetCount(@as(*const ISideShowPropVariantCollection, @ptrCast(self)), pc_elems_);
            }
            pub inline fn removeAt(self: *const T, dw_index_: u32) HRESULT {
                return @as(*const ISideShowPropVariantCollection.VTable, @ptrCast(self.vtable)).RemoveAt(@as(*const ISideShowPropVariantCollection, @ptrCast(self)), dw_index_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

pub const SIDESHOW_SCREEN_TYPE = enum(i32) {
    BITMAP = 0,
    TEXT = 1,
};
pub const SIDESHOW_SCREEN_TYPE_BITMAP = SIDESHOW_SCREEN_TYPE.BITMAP;
pub const SIDESHOW_SCREEN_TYPE_TEXT = SIDESHOW_SCREEN_TYPE.TEXT;

pub const SIDESHOW_COLOR_TYPE = enum(i32) {
    COLOR = 0,
    GREYSCALE = 1,
    BLACK_AND_WHITE = 2,
};
pub const SIDESHOW_COLOR_TYPE_COLOR = SIDESHOW_COLOR_TYPE.COLOR;
pub const SIDESHOW_COLOR_TYPE_GREYSCALE = SIDESHOW_COLOR_TYPE.GREYSCALE;
pub const SIDESHOW_COLOR_TYPE_BLACK_AND_WHITE = SIDESHOW_COLOR_TYPE.BLACK_AND_WHITE;

pub const SCF_EVENT_IDS = enum(i32) {
    NAVIGATION = 1,
    MENUACTION = 2,
    CONTEXTMENU = 3,
};
pub const SCF_EVENT_NAVIGATION = SCF_EVENT_IDS.NAVIGATION;
pub const SCF_EVENT_MENUACTION = SCF_EVENT_IDS.MENUACTION;
pub const SCF_EVENT_CONTEXTMENU = SCF_EVENT_IDS.CONTEXTMENU;

pub const SCF_BUTTON_IDS = enum(i32) {
    MENU = 1,
    SELECT = 2,
    UP = 3,
    DOWN = 4,
    LEFT = 5,
    RIGHT = 6,
    PLAY = 7,
    PAUSE = 8,
    FASTFORWARD = 9,
    REWIND = 10,
    STOP = 11,
    BACK = 65280,
};
pub const SCF_BUTTON_MENU = SCF_BUTTON_IDS.MENU;
pub const SCF_BUTTON_SELECT = SCF_BUTTON_IDS.SELECT;
pub const SCF_BUTTON_UP = SCF_BUTTON_IDS.UP;
pub const SCF_BUTTON_DOWN = SCF_BUTTON_IDS.DOWN;
pub const SCF_BUTTON_LEFT = SCF_BUTTON_IDS.LEFT;
pub const SCF_BUTTON_RIGHT = SCF_BUTTON_IDS.RIGHT;
pub const SCF_BUTTON_PLAY = SCF_BUTTON_IDS.PLAY;
pub const SCF_BUTTON_PAUSE = SCF_BUTTON_IDS.PAUSE;
pub const SCF_BUTTON_FASTFORWARD = SCF_BUTTON_IDS.FASTFORWARD;
pub const SCF_BUTTON_REWIND = SCF_BUTTON_IDS.REWIND;
pub const SCF_BUTTON_STOP = SCF_BUTTON_IDS.STOP;
pub const SCF_BUTTON_BACK = SCF_BUTTON_IDS.BACK;

pub const SCF_EVENT_HEADER = extern struct {
    PreviousPage: u32,
    TargetPage: u32,
};

pub const SCF_NAVIGATION_EVENT = extern struct {
    PreviousPage: u32,
    TargetPage: u32,
    Button: u32,
};

pub const SCF_MENUACTION_EVENT = extern struct {
    PreviousPage: u32,
    TargetPage: u32,
    Button: u32,
    ItemId: u32,
};

pub const SCF_CONTEXTMENU_EVENT = extern struct {
    PreviousPage: u32,
    TargetPage: u32,
    PreviousItemId: u32,
    MenuPage: u32,
    MenuItemId: u32,
};

pub const CONTENT_MISSING_EVENT_DATA = extern struct {
    cbContentMissingEventData: u32 align(1),
    ApplicationId: Guid align(1),
    EndpointId: Guid align(1),
    ContentId: u32 align(1),
};

pub const APPLICATION_EVENT_DATA = extern struct {
    cbApplicationEventData: u32 align(1),
    ApplicationId: Guid align(1),
    EndpointId: Guid align(1),
    dwEventId: u32 align(1),
    cbEventData: u32 align(1),
    bEventData: [1]u8 align(1),
};

pub const DEVICE_USER_CHANGE_EVENT_DATA = extern struct {
    cbDeviceUserChangeEventData: u32 align(1),
    wszUser: u16 align(1),
};

pub const NEW_EVENT_DATA_AVAILABLE = extern struct {
    cbNewEventDataAvailable: u32 align(1),
    dwVersion: u32 align(1),
};

pub const EVENT_DATA_HEADER = extern struct {
    cbEventDataHeader: u32 align(1),
    guidEventType: Guid align(1),
    dwVersion: u32 align(1),
    cbEventDataSid: u32 align(1),
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
// Section: Imports (9)
//--------------------------------------------------------------------------------
const Guid = @import("../zig.zig").Guid;
const BOOL = @import("../foundation.zig").BOOL;
const HICON = @import("../ui/windows_and_messaging.zig").HICON;
const HRESULT = @import("../foundation.zig").HRESULT;
const IUnknown = @import("../system/com.zig").IUnknown;
const PROPERTYKEY = @import("../ui/shell/properties_system.zig").PROPERTYKEY;
const PROPVARIANT = @import("../system/com/structured_storage.zig").PROPVARIANT;
const PWSTR = @import("../foundation.zig").PWSTR;
const SYSTEMTIME = @import("../foundation.zig").SYSTEMTIME;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
