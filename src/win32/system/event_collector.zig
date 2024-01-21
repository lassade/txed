//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (7)
//--------------------------------------------------------------------------------
pub const EC_VARIANT_TYPE_MASK = @as(u32, 127);
pub const EC_VARIANT_TYPE_ARRAY = @as(u32, 128);
pub const EC_READ_ACCESS = @as(u32, 1);
pub const EC_WRITE_ACCESS = @as(u32, 2);
pub const EC_OPEN_ALWAYS = @as(u32, 0);
pub const EC_CREATE_NEW = @as(u32, 1);
pub const EC_OPEN_EXISTING = @as(u32, 2);

//--------------------------------------------------------------------------------
// Section: Types (10)
//--------------------------------------------------------------------------------
pub const EC_SUBSCRIPTION_PROPERTY_ID = enum(i32) {
    Enabled = 0,
    EventSources = 1,
    EventSourceAddress = 2,
    EventSourceEnabled = 3,
    EventSourceUserName = 4,
    EventSourcePassword = 5,
    Description = 6,
    URI = 7,
    ConfigurationMode = 8,
    Expires = 9,
    Query = 10,
    TransportName = 11,
    TransportPort = 12,
    DeliveryMode = 13,
    DeliveryMaxItems = 14,
    DeliveryMaxLatencyTime = 15,
    HeartbeatInterval = 16,
    Locale = 17,
    ContentFormat = 18,
    LogFile = 19,
    PublisherName = 20,
    CredentialsType = 21,
    CommonUserName = 22,
    CommonPassword = 23,
    HostName = 24,
    ReadExistingEvents = 25,
    Dialect = 26,
    Type = 27,
    AllowedIssuerCAs = 28,
    AllowedSubjects = 29,
    DeniedSubjects = 30,
    AllowedSourceDomainComputers = 31,
    PropertyIdEND = 32,
};
pub const EcSubscriptionEnabled = EC_SUBSCRIPTION_PROPERTY_ID.Enabled;
pub const EcSubscriptionEventSources = EC_SUBSCRIPTION_PROPERTY_ID.EventSources;
pub const EcSubscriptionEventSourceAddress = EC_SUBSCRIPTION_PROPERTY_ID.EventSourceAddress;
pub const EcSubscriptionEventSourceEnabled = EC_SUBSCRIPTION_PROPERTY_ID.EventSourceEnabled;
pub const EcSubscriptionEventSourceUserName = EC_SUBSCRIPTION_PROPERTY_ID.EventSourceUserName;
pub const EcSubscriptionEventSourcePassword = EC_SUBSCRIPTION_PROPERTY_ID.EventSourcePassword;
pub const EcSubscriptionDescription = EC_SUBSCRIPTION_PROPERTY_ID.Description;
pub const EcSubscriptionURI = EC_SUBSCRIPTION_PROPERTY_ID.URI;
pub const EcSubscriptionConfigurationMode = EC_SUBSCRIPTION_PROPERTY_ID.ConfigurationMode;
pub const EcSubscriptionExpires = EC_SUBSCRIPTION_PROPERTY_ID.Expires;
pub const EcSubscriptionQuery = EC_SUBSCRIPTION_PROPERTY_ID.Query;
pub const EcSubscriptionTransportName = EC_SUBSCRIPTION_PROPERTY_ID.TransportName;
pub const EcSubscriptionTransportPort = EC_SUBSCRIPTION_PROPERTY_ID.TransportPort;
pub const EcSubscriptionDeliveryMode = EC_SUBSCRIPTION_PROPERTY_ID.DeliveryMode;
pub const EcSubscriptionDeliveryMaxItems = EC_SUBSCRIPTION_PROPERTY_ID.DeliveryMaxItems;
pub const EcSubscriptionDeliveryMaxLatencyTime = EC_SUBSCRIPTION_PROPERTY_ID.DeliveryMaxLatencyTime;
pub const EcSubscriptionHeartbeatInterval = EC_SUBSCRIPTION_PROPERTY_ID.HeartbeatInterval;
pub const EcSubscriptionLocale = EC_SUBSCRIPTION_PROPERTY_ID.Locale;
pub const EcSubscriptionContentFormat = EC_SUBSCRIPTION_PROPERTY_ID.ContentFormat;
pub const EcSubscriptionLogFile = EC_SUBSCRIPTION_PROPERTY_ID.LogFile;
pub const EcSubscriptionPublisherName = EC_SUBSCRIPTION_PROPERTY_ID.PublisherName;
pub const EcSubscriptionCredentialsType = EC_SUBSCRIPTION_PROPERTY_ID.CredentialsType;
pub const EcSubscriptionCommonUserName = EC_SUBSCRIPTION_PROPERTY_ID.CommonUserName;
pub const EcSubscriptionCommonPassword = EC_SUBSCRIPTION_PROPERTY_ID.CommonPassword;
pub const EcSubscriptionHostName = EC_SUBSCRIPTION_PROPERTY_ID.HostName;
pub const EcSubscriptionReadExistingEvents = EC_SUBSCRIPTION_PROPERTY_ID.ReadExistingEvents;
pub const EcSubscriptionDialect = EC_SUBSCRIPTION_PROPERTY_ID.Dialect;
pub const EcSubscriptionType = EC_SUBSCRIPTION_PROPERTY_ID.Type;
pub const EcSubscriptionAllowedIssuerCAs = EC_SUBSCRIPTION_PROPERTY_ID.AllowedIssuerCAs;
pub const EcSubscriptionAllowedSubjects = EC_SUBSCRIPTION_PROPERTY_ID.AllowedSubjects;
pub const EcSubscriptionDeniedSubjects = EC_SUBSCRIPTION_PROPERTY_ID.DeniedSubjects;
pub const EcSubscriptionAllowedSourceDomainComputers = EC_SUBSCRIPTION_PROPERTY_ID.AllowedSourceDomainComputers;
pub const EcSubscriptionPropertyIdEND = EC_SUBSCRIPTION_PROPERTY_ID.PropertyIdEND;

pub const EC_SUBSCRIPTION_CREDENTIALS_TYPE = enum(i32) {
    Default = 0,
    Negotiate = 1,
    Digest = 2,
    Basic = 3,
    LocalMachine = 4,
};
pub const EcSubscriptionCredDefault = EC_SUBSCRIPTION_CREDENTIALS_TYPE.Default;
pub const EcSubscriptionCredNegotiate = EC_SUBSCRIPTION_CREDENTIALS_TYPE.Negotiate;
pub const EcSubscriptionCredDigest = EC_SUBSCRIPTION_CREDENTIALS_TYPE.Digest;
pub const EcSubscriptionCredBasic = EC_SUBSCRIPTION_CREDENTIALS_TYPE.Basic;
pub const EcSubscriptionCredLocalMachine = EC_SUBSCRIPTION_CREDENTIALS_TYPE.LocalMachine;

pub const EC_SUBSCRIPTION_TYPE = enum(i32) {
    SourceInitiated = 0,
    CollectorInitiated = 1,
};
pub const EcSubscriptionTypeSourceInitiated = EC_SUBSCRIPTION_TYPE.SourceInitiated;
pub const EcSubscriptionTypeCollectorInitiated = EC_SUBSCRIPTION_TYPE.CollectorInitiated;

pub const EC_SUBSCRIPTION_RUNTIME_STATUS_INFO_ID = enum(i32) {
    Active = 0,
    LastError = 1,
    LastErrorMessage = 2,
    LastErrorTime = 3,
    NextRetryTime = 4,
    EventSources = 5,
    LastHeartbeatTime = 6,
    InfoIdEND = 7,
};
pub const EcSubscriptionRunTimeStatusActive = EC_SUBSCRIPTION_RUNTIME_STATUS_INFO_ID.Active;
pub const EcSubscriptionRunTimeStatusLastError = EC_SUBSCRIPTION_RUNTIME_STATUS_INFO_ID.LastError;
pub const EcSubscriptionRunTimeStatusLastErrorMessage = EC_SUBSCRIPTION_RUNTIME_STATUS_INFO_ID.LastErrorMessage;
pub const EcSubscriptionRunTimeStatusLastErrorTime = EC_SUBSCRIPTION_RUNTIME_STATUS_INFO_ID.LastErrorTime;
pub const EcSubscriptionRunTimeStatusNextRetryTime = EC_SUBSCRIPTION_RUNTIME_STATUS_INFO_ID.NextRetryTime;
pub const EcSubscriptionRunTimeStatusEventSources = EC_SUBSCRIPTION_RUNTIME_STATUS_INFO_ID.EventSources;
pub const EcSubscriptionRunTimeStatusLastHeartbeatTime = EC_SUBSCRIPTION_RUNTIME_STATUS_INFO_ID.LastHeartbeatTime;
pub const EcSubscriptionRunTimeStatusInfoIdEND = EC_SUBSCRIPTION_RUNTIME_STATUS_INFO_ID.InfoIdEND;

pub const EC_VARIANT_TYPE = enum(i32) {
    TypeNull = 0,
    TypeBoolean = 1,
    TypeUInt32 = 2,
    TypeDateTime = 3,
    TypeString = 4,
    ObjectArrayPropertyHandle = 5,
};
pub const EcVarTypeNull = EC_VARIANT_TYPE.TypeNull;
pub const EcVarTypeBoolean = EC_VARIANT_TYPE.TypeBoolean;
pub const EcVarTypeUInt32 = EC_VARIANT_TYPE.TypeUInt32;
pub const EcVarTypeDateTime = EC_VARIANT_TYPE.TypeDateTime;
pub const EcVarTypeString = EC_VARIANT_TYPE.TypeString;
pub const EcVarObjectArrayPropertyHandle = EC_VARIANT_TYPE.ObjectArrayPropertyHandle;

pub const EC_VARIANT = extern struct {
    Anonymous: extern union {
        BooleanVal: BOOL,
        UInt32Val: u32,
        DateTimeVal: u64,
        StringVal: ?[*:0]const u16,
        BinaryVal: ?*u8,
        BooleanArr: ?*BOOL,
        Int32Arr: ?*i32,
        StringArr: ?*?PWSTR,
        PropertyHandleVal: isize,
    },
    Count: u32,
    Type: u32,
};

pub const EC_SUBSCRIPTION_CONFIGURATION_MODE = enum(i32) {
    Normal = 0,
    Custom = 1,
    MinLatency = 2,
    MinBandwidth = 3,
};
pub const EcConfigurationModeNormal = EC_SUBSCRIPTION_CONFIGURATION_MODE.Normal;
pub const EcConfigurationModeCustom = EC_SUBSCRIPTION_CONFIGURATION_MODE.Custom;
pub const EcConfigurationModeMinLatency = EC_SUBSCRIPTION_CONFIGURATION_MODE.MinLatency;
pub const EcConfigurationModeMinBandwidth = EC_SUBSCRIPTION_CONFIGURATION_MODE.MinBandwidth;

pub const EC_SUBSCRIPTION_DELIVERY_MODE = enum(i32) {
    ll = 1,
    sh = 2,
};
pub const EcDeliveryModePull = EC_SUBSCRIPTION_DELIVERY_MODE.ll;
pub const EcDeliveryModePush = EC_SUBSCRIPTION_DELIVERY_MODE.sh;

pub const EC_SUBSCRIPTION_CONTENT_FORMAT = enum(i32) {
    Events = 1,
    RenderedText = 2,
};
pub const EcContentFormatEvents = EC_SUBSCRIPTION_CONTENT_FORMAT.Events;
pub const EcContentFormatRenderedText = EC_SUBSCRIPTION_CONTENT_FORMAT.RenderedText;

pub const EC_SUBSCRIPTION_RUNTIME_STATUS_ACTIVE_STATUS = enum(i32) {
    Disabled = 1,
    Active = 2,
    Inactive = 3,
    Trying = 4,
};
pub const EcRuntimeStatusActiveStatusDisabled = EC_SUBSCRIPTION_RUNTIME_STATUS_ACTIVE_STATUS.Disabled;
pub const EcRuntimeStatusActiveStatusActive = EC_SUBSCRIPTION_RUNTIME_STATUS_ACTIVE_STATUS.Active;
pub const EcRuntimeStatusActiveStatusInactive = EC_SUBSCRIPTION_RUNTIME_STATUS_ACTIVE_STATUS.Inactive;
pub const EcRuntimeStatusActiveStatusTrying = EC_SUBSCRIPTION_RUNTIME_STATUS_ACTIVE_STATUS.Trying;

//--------------------------------------------------------------------------------
// Section: Functions (15)
//--------------------------------------------------------------------------------
// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcOpenSubscriptionEnum(
    flags: u32,
) callconv(@import("std").os.windows.WINAPI) isize;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcEnumNextSubscription(
    subscription_enum: isize,
    subscription_name_buffer_size: u32,
    subscription_name_buffer: ?[*:0]u16,
    subscription_name_buffer_used: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcOpenSubscription(
    subscription_name: ?[*:0]const u16,
    access_mask: u32,
    flags: u32,
) callconv(@import("std").os.windows.WINAPI) isize;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcSetSubscriptionProperty(
    subscription: isize,
    property_id: EC_SUBSCRIPTION_PROPERTY_ID,
    flags: u32,
    property_value: ?*EC_VARIANT,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcGetSubscriptionProperty(
    subscription: isize,
    property_id: EC_SUBSCRIPTION_PROPERTY_ID,
    flags: u32,
    property_value_buffer_size: u32,
    property_value_buffer: ?*EC_VARIANT,
    property_value_buffer_used: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcSaveSubscription(
    subscription: isize,
    flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcDeleteSubscription(
    subscription_name: ?[*:0]const u16,
    flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcGetObjectArraySize(
    object_array: isize,
    object_array_size: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcSetObjectArrayProperty(
    object_array: isize,
    property_id: EC_SUBSCRIPTION_PROPERTY_ID,
    array_index: u32,
    flags: u32,
    property_value: ?*EC_VARIANT,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcGetObjectArrayProperty(
    object_array: isize,
    property_id: EC_SUBSCRIPTION_PROPERTY_ID,
    array_index: u32,
    flags: u32,
    property_value_buffer_size: u32,
    property_value_buffer: ?*EC_VARIANT,
    property_value_buffer_used: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcInsertObjectArrayElement(
    object_array: isize,
    array_index: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcRemoveObjectArrayElement(
    object_array: isize,
    array_index: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcGetSubscriptionRunTimeStatus(
    subscription_name: ?[*:0]const u16,
    status_info_id: EC_SUBSCRIPTION_RUNTIME_STATUS_INFO_ID,
    event_source_name: ?[*:0]const u16,
    flags: u32,
    status_value_buffer_size: u32,
    status_value_buffer: ?*EC_VARIANT,
    status_value_buffer_used: ?*u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcRetrySubscription(
    subscription_name: ?[*:0]const u16,
    event_source_name: ?[*:0]const u16,
    flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows6.0.6000'
pub extern "wecapi" fn EcClose(
    object: isize,
) callconv(@import("std").os.windows.WINAPI) BOOL;

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
// Section: Imports (2)
//--------------------------------------------------------------------------------
const BOOL = @import("../foundation.zig").BOOL;
const PWSTR = @import("../foundation.zig").PWSTR;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
