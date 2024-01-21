//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (11)
//--------------------------------------------------------------------------------
pub const DEVPROP_OPERATOR = enum(u32) {
    MODIFIER_NOT = 65536,
    MODIFIER_IGNORE_CASE = 131072,
    NONE = 0,
    EXISTS = 1,
    NOT_EXISTS = 65537,
    EQUALS = 2,
    NOT_EQUALS = 65538,
    GREATER_THAN = 3,
    LESS_THAN = 4,
    GREATER_THAN_EQUALS = 5,
    LESS_THAN_EQUALS = 6,
    EQUALS_IGNORE_CASE = 131074,
    NOT_EQUALS_IGNORE_CASE = 196610,
    BITWISE_AND = 7,
    BITWISE_OR = 8,
    BEGINS_WITH = 9,
    ENDS_WITH = 10,
    CONTAINS = 11,
    BEGINS_WITH_IGNORE_CASE = 131081,
    ENDS_WITH_IGNORE_CASE = 131082,
    CONTAINS_IGNORE_CASE = 131083,
    LIST_CONTAINS = 4096,
    LIST_ELEMENT_BEGINS_WITH = 8192,
    LIST_ELEMENT_ENDS_WITH = 12288,
    LIST_ELEMENT_CONTAINS = 16384,
    LIST_CONTAINS_IGNORE_CASE = 135168,
    LIST_ELEMENT_BEGINS_WITH_IGNORE_CASE = 139264,
    LIST_ELEMENT_ENDS_WITH_IGNORE_CASE = 143360,
    LIST_ELEMENT_CONTAINS_IGNORE_CASE = 147456,
    AND_OPEN = 1048576,
    AND_CLOSE = 2097152,
    OR_OPEN = 3145728,
    OR_CLOSE = 4194304,
    NOT_OPEN = 5242880,
    NOT_CLOSE = 6291456,
    ARRAY_CONTAINS = 268435456,
    MASK_EVAL = 4095,
    MASK_LIST = 61440,
    MASK_MODIFIER = 983040,
    MASK_NOT_LOGICAL = 4027580415,
    MASK_LOGICAL = 267386880,
    MASK_ARRAY = 4026531840,
    _,
    pub fn initFlags(o: struct {
        MODIFIER_NOT: u1 = 0,
        MODIFIER_IGNORE_CASE: u1 = 0,
        NONE: u1 = 0,
        EXISTS: u1 = 0,
        NOT_EXISTS: u1 = 0,
        EQUALS: u1 = 0,
        NOT_EQUALS: u1 = 0,
        GREATER_THAN: u1 = 0,
        LESS_THAN: u1 = 0,
        GREATER_THAN_EQUALS: u1 = 0,
        LESS_THAN_EQUALS: u1 = 0,
        EQUALS_IGNORE_CASE: u1 = 0,
        NOT_EQUALS_IGNORE_CASE: u1 = 0,
        BITWISE_AND: u1 = 0,
        BITWISE_OR: u1 = 0,
        BEGINS_WITH: u1 = 0,
        ENDS_WITH: u1 = 0,
        CONTAINS: u1 = 0,
        BEGINS_WITH_IGNORE_CASE: u1 = 0,
        ENDS_WITH_IGNORE_CASE: u1 = 0,
        CONTAINS_IGNORE_CASE: u1 = 0,
        LIST_CONTAINS: u1 = 0,
        LIST_ELEMENT_BEGINS_WITH: u1 = 0,
        LIST_ELEMENT_ENDS_WITH: u1 = 0,
        LIST_ELEMENT_CONTAINS: u1 = 0,
        LIST_CONTAINS_IGNORE_CASE: u1 = 0,
        LIST_ELEMENT_BEGINS_WITH_IGNORE_CASE: u1 = 0,
        LIST_ELEMENT_ENDS_WITH_IGNORE_CASE: u1 = 0,
        LIST_ELEMENT_CONTAINS_IGNORE_CASE: u1 = 0,
        AND_OPEN: u1 = 0,
        AND_CLOSE: u1 = 0,
        OR_OPEN: u1 = 0,
        OR_CLOSE: u1 = 0,
        NOT_OPEN: u1 = 0,
        NOT_CLOSE: u1 = 0,
        ARRAY_CONTAINS: u1 = 0,
        MASK_EVAL: u1 = 0,
        MASK_LIST: u1 = 0,
        MASK_MODIFIER: u1 = 0,
        MASK_NOT_LOGICAL: u1 = 0,
        MASK_LOGICAL: u1 = 0,
        MASK_ARRAY: u1 = 0,
    }) DEVPROP_OPERATOR {
        return @as(DEVPROP_OPERATOR, @enumFromInt((if (o.MODIFIER_NOT == 1) @intFromEnum(DEVPROP_OPERATOR.MODIFIER_NOT) else 0) | (if (o.MODIFIER_IGNORE_CASE == 1) @intFromEnum(DEVPROP_OPERATOR.MODIFIER_IGNORE_CASE) else 0) | (if (o.NONE == 1) @intFromEnum(DEVPROP_OPERATOR.NONE) else 0) | (if (o.EXISTS == 1) @intFromEnum(DEVPROP_OPERATOR.EXISTS) else 0) | (if (o.NOT_EXISTS == 1) @intFromEnum(DEVPROP_OPERATOR.NOT_EXISTS) else 0) | (if (o.EQUALS == 1) @intFromEnum(DEVPROP_OPERATOR.EQUALS) else 0) | (if (o.NOT_EQUALS == 1) @intFromEnum(DEVPROP_OPERATOR.NOT_EQUALS) else 0) | (if (o.GREATER_THAN == 1) @intFromEnum(DEVPROP_OPERATOR.GREATER_THAN) else 0) | (if (o.LESS_THAN == 1) @intFromEnum(DEVPROP_OPERATOR.LESS_THAN) else 0) | (if (o.GREATER_THAN_EQUALS == 1) @intFromEnum(DEVPROP_OPERATOR.GREATER_THAN_EQUALS) else 0) | (if (o.LESS_THAN_EQUALS == 1) @intFromEnum(DEVPROP_OPERATOR.LESS_THAN_EQUALS) else 0) | (if (o.EQUALS_IGNORE_CASE == 1) @intFromEnum(DEVPROP_OPERATOR.EQUALS_IGNORE_CASE) else 0) | (if (o.NOT_EQUALS_IGNORE_CASE == 1) @intFromEnum(DEVPROP_OPERATOR.NOT_EQUALS_IGNORE_CASE) else 0) | (if (o.BITWISE_AND == 1) @intFromEnum(DEVPROP_OPERATOR.BITWISE_AND) else 0) | (if (o.BITWISE_OR == 1) @intFromEnum(DEVPROP_OPERATOR.BITWISE_OR) else 0) | (if (o.BEGINS_WITH == 1) @intFromEnum(DEVPROP_OPERATOR.BEGINS_WITH) else 0) | (if (o.ENDS_WITH == 1) @intFromEnum(DEVPROP_OPERATOR.ENDS_WITH) else 0) | (if (o.CONTAINS == 1) @intFromEnum(DEVPROP_OPERATOR.CONTAINS) else 0) | (if (o.BEGINS_WITH_IGNORE_CASE == 1) @intFromEnum(DEVPROP_OPERATOR.BEGINS_WITH_IGNORE_CASE) else 0) | (if (o.ENDS_WITH_IGNORE_CASE == 1) @intFromEnum(DEVPROP_OPERATOR.ENDS_WITH_IGNORE_CASE) else 0) | (if (o.CONTAINS_IGNORE_CASE == 1) @intFromEnum(DEVPROP_OPERATOR.CONTAINS_IGNORE_CASE) else 0) | (if (o.LIST_CONTAINS == 1) @intFromEnum(DEVPROP_OPERATOR.LIST_CONTAINS) else 0) | (if (o.LIST_ELEMENT_BEGINS_WITH == 1) @intFromEnum(DEVPROP_OPERATOR.LIST_ELEMENT_BEGINS_WITH) else 0) | (if (o.LIST_ELEMENT_ENDS_WITH == 1) @intFromEnum(DEVPROP_OPERATOR.LIST_ELEMENT_ENDS_WITH) else 0) | (if (o.LIST_ELEMENT_CONTAINS == 1) @intFromEnum(DEVPROP_OPERATOR.LIST_ELEMENT_CONTAINS) else 0) | (if (o.LIST_CONTAINS_IGNORE_CASE == 1) @intFromEnum(DEVPROP_OPERATOR.LIST_CONTAINS_IGNORE_CASE) else 0) | (if (o.LIST_ELEMENT_BEGINS_WITH_IGNORE_CASE == 1) @intFromEnum(DEVPROP_OPERATOR.LIST_ELEMENT_BEGINS_WITH_IGNORE_CASE) else 0) | (if (o.LIST_ELEMENT_ENDS_WITH_IGNORE_CASE == 1) @intFromEnum(DEVPROP_OPERATOR.LIST_ELEMENT_ENDS_WITH_IGNORE_CASE) else 0) | (if (o.LIST_ELEMENT_CONTAINS_IGNORE_CASE == 1) @intFromEnum(DEVPROP_OPERATOR.LIST_ELEMENT_CONTAINS_IGNORE_CASE) else 0) | (if (o.AND_OPEN == 1) @intFromEnum(DEVPROP_OPERATOR.AND_OPEN) else 0) | (if (o.AND_CLOSE == 1) @intFromEnum(DEVPROP_OPERATOR.AND_CLOSE) else 0) | (if (o.OR_OPEN == 1) @intFromEnum(DEVPROP_OPERATOR.OR_OPEN) else 0) | (if (o.OR_CLOSE == 1) @intFromEnum(DEVPROP_OPERATOR.OR_CLOSE) else 0) | (if (o.NOT_OPEN == 1) @intFromEnum(DEVPROP_OPERATOR.NOT_OPEN) else 0) | (if (o.NOT_CLOSE == 1) @intFromEnum(DEVPROP_OPERATOR.NOT_CLOSE) else 0) | (if (o.ARRAY_CONTAINS == 1) @intFromEnum(DEVPROP_OPERATOR.ARRAY_CONTAINS) else 0) | (if (o.MASK_EVAL == 1) @intFromEnum(DEVPROP_OPERATOR.MASK_EVAL) else 0) | (if (o.MASK_LIST == 1) @intFromEnum(DEVPROP_OPERATOR.MASK_LIST) else 0) | (if (o.MASK_MODIFIER == 1) @intFromEnum(DEVPROP_OPERATOR.MASK_MODIFIER) else 0) | (if (o.MASK_NOT_LOGICAL == 1) @intFromEnum(DEVPROP_OPERATOR.MASK_NOT_LOGICAL) else 0) | (if (o.MASK_LOGICAL == 1) @intFromEnum(DEVPROP_OPERATOR.MASK_LOGICAL) else 0) | (if (o.MASK_ARRAY == 1) @intFromEnum(DEVPROP_OPERATOR.MASK_ARRAY) else 0)));
    }
};
pub const DEVPROP_OPERATOR_MODIFIER_NOT = DEVPROP_OPERATOR.MODIFIER_NOT;
pub const DEVPROP_OPERATOR_MODIFIER_IGNORE_CASE = DEVPROP_OPERATOR.MODIFIER_IGNORE_CASE;
pub const DEVPROP_OPERATOR_NONE = DEVPROP_OPERATOR.NONE;
pub const DEVPROP_OPERATOR_EXISTS = DEVPROP_OPERATOR.EXISTS;
pub const DEVPROP_OPERATOR_NOT_EXISTS = DEVPROP_OPERATOR.NOT_EXISTS;
pub const DEVPROP_OPERATOR_EQUALS = DEVPROP_OPERATOR.EQUALS;
pub const DEVPROP_OPERATOR_NOT_EQUALS = DEVPROP_OPERATOR.NOT_EQUALS;
pub const DEVPROP_OPERATOR_GREATER_THAN = DEVPROP_OPERATOR.GREATER_THAN;
pub const DEVPROP_OPERATOR_LESS_THAN = DEVPROP_OPERATOR.LESS_THAN;
pub const DEVPROP_OPERATOR_GREATER_THAN_EQUALS = DEVPROP_OPERATOR.GREATER_THAN_EQUALS;
pub const DEVPROP_OPERATOR_LESS_THAN_EQUALS = DEVPROP_OPERATOR.LESS_THAN_EQUALS;
pub const DEVPROP_OPERATOR_EQUALS_IGNORE_CASE = DEVPROP_OPERATOR.EQUALS_IGNORE_CASE;
pub const DEVPROP_OPERATOR_NOT_EQUALS_IGNORE_CASE = DEVPROP_OPERATOR.NOT_EQUALS_IGNORE_CASE;
pub const DEVPROP_OPERATOR_BITWISE_AND = DEVPROP_OPERATOR.BITWISE_AND;
pub const DEVPROP_OPERATOR_BITWISE_OR = DEVPROP_OPERATOR.BITWISE_OR;
pub const DEVPROP_OPERATOR_BEGINS_WITH = DEVPROP_OPERATOR.BEGINS_WITH;
pub const DEVPROP_OPERATOR_ENDS_WITH = DEVPROP_OPERATOR.ENDS_WITH;
pub const DEVPROP_OPERATOR_CONTAINS = DEVPROP_OPERATOR.CONTAINS;
pub const DEVPROP_OPERATOR_BEGINS_WITH_IGNORE_CASE = DEVPROP_OPERATOR.BEGINS_WITH_IGNORE_CASE;
pub const DEVPROP_OPERATOR_ENDS_WITH_IGNORE_CASE = DEVPROP_OPERATOR.ENDS_WITH_IGNORE_CASE;
pub const DEVPROP_OPERATOR_CONTAINS_IGNORE_CASE = DEVPROP_OPERATOR.CONTAINS_IGNORE_CASE;
pub const DEVPROP_OPERATOR_LIST_CONTAINS = DEVPROP_OPERATOR.LIST_CONTAINS;
pub const DEVPROP_OPERATOR_LIST_ELEMENT_BEGINS_WITH = DEVPROP_OPERATOR.LIST_ELEMENT_BEGINS_WITH;
pub const DEVPROP_OPERATOR_LIST_ELEMENT_ENDS_WITH = DEVPROP_OPERATOR.LIST_ELEMENT_ENDS_WITH;
pub const DEVPROP_OPERATOR_LIST_ELEMENT_CONTAINS = DEVPROP_OPERATOR.LIST_ELEMENT_CONTAINS;
pub const DEVPROP_OPERATOR_LIST_CONTAINS_IGNORE_CASE = DEVPROP_OPERATOR.LIST_CONTAINS_IGNORE_CASE;
pub const DEVPROP_OPERATOR_LIST_ELEMENT_BEGINS_WITH_IGNORE_CASE = DEVPROP_OPERATOR.LIST_ELEMENT_BEGINS_WITH_IGNORE_CASE;
pub const DEVPROP_OPERATOR_LIST_ELEMENT_ENDS_WITH_IGNORE_CASE = DEVPROP_OPERATOR.LIST_ELEMENT_ENDS_WITH_IGNORE_CASE;
pub const DEVPROP_OPERATOR_LIST_ELEMENT_CONTAINS_IGNORE_CASE = DEVPROP_OPERATOR.LIST_ELEMENT_CONTAINS_IGNORE_CASE;
pub const DEVPROP_OPERATOR_AND_OPEN = DEVPROP_OPERATOR.AND_OPEN;
pub const DEVPROP_OPERATOR_AND_CLOSE = DEVPROP_OPERATOR.AND_CLOSE;
pub const DEVPROP_OPERATOR_OR_OPEN = DEVPROP_OPERATOR.OR_OPEN;
pub const DEVPROP_OPERATOR_OR_CLOSE = DEVPROP_OPERATOR.OR_CLOSE;
pub const DEVPROP_OPERATOR_NOT_OPEN = DEVPROP_OPERATOR.NOT_OPEN;
pub const DEVPROP_OPERATOR_NOT_CLOSE = DEVPROP_OPERATOR.NOT_CLOSE;
pub const DEVPROP_OPERATOR_ARRAY_CONTAINS = DEVPROP_OPERATOR.ARRAY_CONTAINS;
pub const DEVPROP_OPERATOR_MASK_EVAL = DEVPROP_OPERATOR.MASK_EVAL;
pub const DEVPROP_OPERATOR_MASK_LIST = DEVPROP_OPERATOR.MASK_LIST;
pub const DEVPROP_OPERATOR_MASK_MODIFIER = DEVPROP_OPERATOR.MASK_MODIFIER;
pub const DEVPROP_OPERATOR_MASK_NOT_LOGICAL = DEVPROP_OPERATOR.MASK_NOT_LOGICAL;
pub const DEVPROP_OPERATOR_MASK_LOGICAL = DEVPROP_OPERATOR.MASK_LOGICAL;
pub const DEVPROP_OPERATOR_MASK_ARRAY = DEVPROP_OPERATOR.MASK_ARRAY;

pub const DEVPROP_FILTER_EXPRESSION = extern struct {
    Operator: DEVPROP_OPERATOR,
    Property: DEVPROPERTY,
};

pub const DEV_OBJECT_TYPE = enum(i32) {
    Unknown = 0,
    DeviceInterface = 1,
    DeviceContainer = 2,
    Device = 3,
    DeviceInterfaceClass = 4,
    AEP = 5,
    AEPContainer = 6,
    DeviceInstallerClass = 7,
    DeviceInterfaceDisplay = 8,
    DeviceContainerDisplay = 9,
    AEPService = 10,
    DevicePanel = 11,
};
pub const DevObjectTypeUnknown = DEV_OBJECT_TYPE.Unknown;
pub const DevObjectTypeDeviceInterface = DEV_OBJECT_TYPE.DeviceInterface;
pub const DevObjectTypeDeviceContainer = DEV_OBJECT_TYPE.DeviceContainer;
pub const DevObjectTypeDevice = DEV_OBJECT_TYPE.Device;
pub const DevObjectTypeDeviceInterfaceClass = DEV_OBJECT_TYPE.DeviceInterfaceClass;
pub const DevObjectTypeAEP = DEV_OBJECT_TYPE.AEP;
pub const DevObjectTypeAEPContainer = DEV_OBJECT_TYPE.AEPContainer;
pub const DevObjectTypeDeviceInstallerClass = DEV_OBJECT_TYPE.DeviceInstallerClass;
pub const DevObjectTypeDeviceInterfaceDisplay = DEV_OBJECT_TYPE.DeviceInterfaceDisplay;
pub const DevObjectTypeDeviceContainerDisplay = DEV_OBJECT_TYPE.DeviceContainerDisplay;
pub const DevObjectTypeAEPService = DEV_OBJECT_TYPE.AEPService;
pub const DevObjectTypeDevicePanel = DEV_OBJECT_TYPE.DevicePanel;

pub const DEV_QUERY_FLAGS = enum(i32) {
    None = 0,
    UpdateResults = 1,
    AllProperties = 2,
    Localize = 4,
    AsyncClose = 8,
};
pub const DevQueryFlagNone = DEV_QUERY_FLAGS.None;
pub const DevQueryFlagUpdateResults = DEV_QUERY_FLAGS.UpdateResults;
pub const DevQueryFlagAllProperties = DEV_QUERY_FLAGS.AllProperties;
pub const DevQueryFlagLocalize = DEV_QUERY_FLAGS.Localize;
pub const DevQueryFlagAsyncClose = DEV_QUERY_FLAGS.AsyncClose;

pub const DEV_QUERY_STATE = enum(i32) {
    Initialized = 0,
    EnumCompleted = 1,
    Aborted = 2,
    Closed = 3,
};
pub const DevQueryStateInitialized = DEV_QUERY_STATE.Initialized;
pub const DevQueryStateEnumCompleted = DEV_QUERY_STATE.EnumCompleted;
pub const DevQueryStateAborted = DEV_QUERY_STATE.Aborted;
pub const DevQueryStateClosed = DEV_QUERY_STATE.Closed;

pub const DEV_QUERY_RESULT_ACTION = enum(i32) {
    StateChange = 0,
    Add = 1,
    Update = 2,
    Remove = 3,
};
pub const DevQueryResultStateChange = DEV_QUERY_RESULT_ACTION.StateChange;
pub const DevQueryResultAdd = DEV_QUERY_RESULT_ACTION.Add;
pub const DevQueryResultUpdate = DEV_QUERY_RESULT_ACTION.Update;
pub const DevQueryResultRemove = DEV_QUERY_RESULT_ACTION.Remove;

pub const DEV_OBJECT = extern struct {
    ObjectType: DEV_OBJECT_TYPE,
    pszObjectId: ?[*:0]const u16,
    cPropertyCount: u32,
    pProperties: ?*const DEVPROPERTY,
};

pub const DEV_QUERY_RESULT_ACTION_DATA = extern struct {
    pub const _DEV_QUERY_RESULT_UPDATE_PAYLOAD = extern union {
        State: DEV_QUERY_STATE,
        DeviceObject: DEV_OBJECT,
    };
    Action: DEV_QUERY_RESULT_ACTION,
    Data: _DEV_QUERY_RESULT_UPDATE_PAYLOAD,
};

pub const DEV_QUERY_PARAMETER = extern struct {
    Key: DEVPROPKEY,
    Type: u32,
    BufferSize: u32,
    Buffer: ?*anyopaque,
};

pub const HDEVQUERY__ = extern struct {
    unused: i32,
};

pub const PDEV_QUERY_RESULT_CALLBACK = *const fn (
    h_dev_query: ?*HDEVQUERY__,
    p_context: ?*anyopaque,
    p_action_data: ?*const DEV_QUERY_RESULT_ACTION_DATA,
) callconv(@import("std").os.windows.WINAPI) void;

//--------------------------------------------------------------------------------
// Section: Functions (14)
//--------------------------------------------------------------------------------
pub extern "api-ms-win-devices-query-l1-1-0" fn DevCreateObjectQuery(
    object_type: DEV_OBJECT_TYPE,
    query_flags: u32,
    c_requested_properties: u32,
    p_requested_properties: ?[*]const DEVPROPCOMPKEY,
    c_filter_expression_count: u32,
    p_filter: ?[*]const DEVPROP_FILTER_EXPRESSION,
    p_callback: ?PDEV_QUERY_RESULT_CALLBACK,
    p_context: ?*anyopaque,
    ph_dev_query: ?*?*HDEVQUERY__,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-devices-query-l1-1-1" fn DevCreateObjectQueryEx(
    object_type: DEV_OBJECT_TYPE,
    query_flags: u32,
    c_requested_properties: u32,
    p_requested_properties: ?[*]const DEVPROPCOMPKEY,
    c_filter_expression_count: u32,
    p_filter: ?[*]const DEVPROP_FILTER_EXPRESSION,
    c_extended_parameter_count: u32,
    p_extended_parameters: ?[*]const DEV_QUERY_PARAMETER,
    p_callback: ?PDEV_QUERY_RESULT_CALLBACK,
    p_context: ?*anyopaque,
    ph_dev_query: ?*?*HDEVQUERY__,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-devices-query-l1-1-0" fn DevCreateObjectQueryFromId(
    object_type: DEV_OBJECT_TYPE,
    psz_object_id: ?[*:0]const u16,
    query_flags: u32,
    c_requested_properties: u32,
    p_requested_properties: ?[*]const DEVPROPCOMPKEY,
    c_filter_expression_count: u32,
    p_filter: ?[*]const DEVPROP_FILTER_EXPRESSION,
    p_callback: ?PDEV_QUERY_RESULT_CALLBACK,
    p_context: ?*anyopaque,
    ph_dev_query: ?*?*HDEVQUERY__,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-devices-query-l1-1-1" fn DevCreateObjectQueryFromIdEx(
    object_type: DEV_OBJECT_TYPE,
    psz_object_id: ?[*:0]const u16,
    query_flags: u32,
    c_requested_properties: u32,
    p_requested_properties: ?[*]const DEVPROPCOMPKEY,
    c_filter_expression_count: u32,
    p_filter: ?[*]const DEVPROP_FILTER_EXPRESSION,
    c_extended_parameter_count: u32,
    p_extended_parameters: ?[*]const DEV_QUERY_PARAMETER,
    p_callback: ?PDEV_QUERY_RESULT_CALLBACK,
    p_context: ?*anyopaque,
    ph_dev_query: ?*?*HDEVQUERY__,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-devices-query-l1-1-0" fn DevCreateObjectQueryFromIds(
    object_type: DEV_OBJECT_TYPE,
    pszz_object_ids: ?[*]const u16,
    query_flags: u32,
    c_requested_properties: u32,
    p_requested_properties: ?[*]const DEVPROPCOMPKEY,
    c_filter_expression_count: u32,
    p_filter: ?[*]const DEVPROP_FILTER_EXPRESSION,
    p_callback: ?PDEV_QUERY_RESULT_CALLBACK,
    p_context: ?*anyopaque,
    ph_dev_query: ?*?*HDEVQUERY__,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-devices-query-l1-1-1" fn DevCreateObjectQueryFromIdsEx(
    object_type: DEV_OBJECT_TYPE,
    pszz_object_ids: ?[*]const u16,
    query_flags: u32,
    c_requested_properties: u32,
    p_requested_properties: ?[*]const DEVPROPCOMPKEY,
    c_filter_expression_count: u32,
    p_filter: ?[*]const DEVPROP_FILTER_EXPRESSION,
    c_extended_parameter_count: u32,
    p_extended_parameters: ?[*]const DEV_QUERY_PARAMETER,
    p_callback: ?PDEV_QUERY_RESULT_CALLBACK,
    p_context: ?*anyopaque,
    ph_dev_query: ?*?*HDEVQUERY__,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-devices-query-l1-1-0" fn DevCloseObjectQuery(
    h_dev_query: ?*HDEVQUERY__,
) callconv(@import("std").os.windows.WINAPI) void;

pub extern "api-ms-win-devices-query-l1-1-0" fn DevGetObjects(
    object_type: DEV_OBJECT_TYPE,
    query_flags: u32,
    c_requested_properties: u32,
    p_requested_properties: ?[*]const DEVPROPCOMPKEY,
    c_filter_expression_count: u32,
    p_filter: ?[*]const DEVPROP_FILTER_EXPRESSION,
    pc_object_count: ?*u32,
    pp_objects: ?*const ?*DEV_OBJECT,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-devices-query-l1-1-1" fn DevGetObjectsEx(
    object_type: DEV_OBJECT_TYPE,
    query_flags: u32,
    c_requested_properties: u32,
    p_requested_properties: ?[*]const DEVPROPCOMPKEY,
    c_filter_expression_count: u32,
    p_filter: ?[*]const DEVPROP_FILTER_EXPRESSION,
    c_extended_parameter_count: u32,
    p_extended_parameters: ?[*]const DEV_QUERY_PARAMETER,
    pc_object_count: ?*u32,
    pp_objects: ?*const ?*DEV_OBJECT,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-devices-query-l1-1-0" fn DevFreeObjects(
    c_object_count: u32,
    p_objects: [*]const DEV_OBJECT,
) callconv(@import("std").os.windows.WINAPI) void;

pub extern "api-ms-win-devices-query-l1-1-0" fn DevGetObjectProperties(
    object_type: DEV_OBJECT_TYPE,
    psz_object_id: ?[*:0]const u16,
    query_flags: u32,
    c_requested_properties: u32,
    p_requested_properties: [*]const DEVPROPCOMPKEY,
    pc_property_count: ?*u32,
    pp_properties: ?*const ?*DEVPROPERTY,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-devices-query-l1-1-1" fn DevGetObjectPropertiesEx(
    object_type: DEV_OBJECT_TYPE,
    psz_object_id: ?[*:0]const u16,
    query_flags: u32,
    c_requested_properties: u32,
    p_requested_properties: [*]const DEVPROPCOMPKEY,
    c_extended_parameter_count: u32,
    p_extended_parameters: ?[*]const DEV_QUERY_PARAMETER,
    pc_property_count: ?*u32,
    pp_properties: ?*const ?*DEVPROPERTY,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "api-ms-win-devices-query-l1-1-0" fn DevFreeObjectProperties(
    c_property_count: u32,
    p_properties: [*]const DEVPROPERTY,
) callconv(@import("std").os.windows.WINAPI) void;

pub extern "api-ms-win-devices-query-l1-1-0" fn DevFindProperty(
    p_key: ?*const DEVPROPKEY,
    store: DEVPROPSTORE,
    psz_locale_name: ?[*:0]const u16,
    c_properties: u32,
    p_properties: ?[*]const DEVPROPERTY,
) callconv(@import("std").os.windows.WINAPI) ?*DEVPROPERTY;

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
// Section: Imports (6)
//--------------------------------------------------------------------------------
const DEVPROPCOMPKEY = @import("../devices/properties.zig").DEVPROPCOMPKEY;
const DEVPROPERTY = @import("../devices/properties.zig").DEVPROPERTY;
const DEVPROPKEY = @import("../devices/properties.zig").DEVPROPKEY;
const DEVPROPSTORE = @import("../devices/properties.zig").DEVPROPSTORE;
const HRESULT = @import("../foundation.zig").HRESULT;
const PWSTR = @import("../foundation.zig").PWSTR;

test {
    // The following '_ = <FuncPtrType>' lines are a workaround for https://github.com/ziglang/zig/issues/4476
    if (@hasDecl(@This(), "PDEV_QUERY_RESULT_CALLBACK")) {
        _ = PDEV_QUERY_RESULT_CALLBACK;
    }

    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
