//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (6)
//--------------------------------------------------------------------------------
pub const HCN_NOTIFICATIONS = enum(i32) {
    Invalid = 0,
    NetworkPreCreate = 1,
    NetworkCreate = 2,
    NetworkPreDelete = 3,
    NetworkDelete = 4,
    NamespaceCreate = 5,
    NamespaceDelete = 6,
    GuestNetworkServiceCreate = 7,
    GuestNetworkServiceDelete = 8,
    NetworkEndpointAttached = 9,
    NetworkEndpointDetached = 16,
    GuestNetworkServiceStateChanged = 17,
    GuestNetworkServiceInterfaceStateChanged = 18,
    ServiceDisconnect = 16777216,
    FlagsReserved = -268435456,
};
pub const HcnNotificationInvalid = HCN_NOTIFICATIONS.Invalid;
pub const HcnNotificationNetworkPreCreate = HCN_NOTIFICATIONS.NetworkPreCreate;
pub const HcnNotificationNetworkCreate = HCN_NOTIFICATIONS.NetworkCreate;
pub const HcnNotificationNetworkPreDelete = HCN_NOTIFICATIONS.NetworkPreDelete;
pub const HcnNotificationNetworkDelete = HCN_NOTIFICATIONS.NetworkDelete;
pub const HcnNotificationNamespaceCreate = HCN_NOTIFICATIONS.NamespaceCreate;
pub const HcnNotificationNamespaceDelete = HCN_NOTIFICATIONS.NamespaceDelete;
pub const HcnNotificationGuestNetworkServiceCreate = HCN_NOTIFICATIONS.GuestNetworkServiceCreate;
pub const HcnNotificationGuestNetworkServiceDelete = HCN_NOTIFICATIONS.GuestNetworkServiceDelete;
pub const HcnNotificationNetworkEndpointAttached = HCN_NOTIFICATIONS.NetworkEndpointAttached;
pub const HcnNotificationNetworkEndpointDetached = HCN_NOTIFICATIONS.NetworkEndpointDetached;
pub const HcnNotificationGuestNetworkServiceStateChanged = HCN_NOTIFICATIONS.GuestNetworkServiceStateChanged;
pub const HcnNotificationGuestNetworkServiceInterfaceStateChanged = HCN_NOTIFICATIONS.GuestNetworkServiceInterfaceStateChanged;
pub const HcnNotificationServiceDisconnect = HCN_NOTIFICATIONS.ServiceDisconnect;
pub const HcnNotificationFlagsReserved = HCN_NOTIFICATIONS.FlagsReserved;

pub const HCN_NOTIFICATION_CALLBACK = *const fn (
    notification_type: u32,
    context: ?*anyopaque,
    notification_status: HRESULT,
    notification_data: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) void;

pub const HCN_PORT_PROTOCOL = enum(i32) {
    TCP = 1,
    UDP = 2,
    BOTH = 3,
};
pub const HCN_PORT_PROTOCOL_TCP = HCN_PORT_PROTOCOL.TCP;
pub const HCN_PORT_PROTOCOL_UDP = HCN_PORT_PROTOCOL.UDP;
pub const HCN_PORT_PROTOCOL_BOTH = HCN_PORT_PROTOCOL.BOTH;

pub const HCN_PORT_ACCESS = enum(i32) {
    EXCLUSIVE = 1,
    SHARED = 2,
};
pub const HCN_PORT_ACCESS_EXCLUSIVE = HCN_PORT_ACCESS.EXCLUSIVE;
pub const HCN_PORT_ACCESS_SHARED = HCN_PORT_ACCESS.SHARED;

pub const HCN_PORT_RANGE_RESERVATION = extern struct {
    startingPort: u16,
    endingPort: u16,
};

pub const HCN_PORT_RANGE_ENTRY = extern struct {
    OwningPartitionId: Guid,
    TargetPartitionId: Guid,
    Protocol: HCN_PORT_PROTOCOL,
    Priority: u64,
    ReservationType: u32,
    SharingFlags: u32,
    DeliveryMode: u32,
    StartingPort: u16,
    EndingPort: u16,
};

//--------------------------------------------------------------------------------
// Section: Functions (41)
//--------------------------------------------------------------------------------
pub extern "computenetwork" fn HcnEnumerateNetworks(
    query: ?[*:0]const u16,
    networks: ?*?PWSTR,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnCreateNetwork(
    id: ?*const Guid,
    settings: ?[*:0]const u16,
    network: ?*?*anyopaque,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnOpenNetwork(
    id: ?*const Guid,
    network: ?*?*anyopaque,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnModifyNetwork(
    network: ?*anyopaque,
    settings: ?[*:0]const u16,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnQueryNetworkProperties(
    network: ?*anyopaque,
    query: ?[*:0]const u16,
    properties: ?*?PWSTR,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnDeleteNetwork(
    id: ?*const Guid,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnCloseNetwork(
    network: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnEnumerateNamespaces(
    query: ?[*:0]const u16,
    namespaces: ?*?PWSTR,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnCreateNamespace(
    id: ?*const Guid,
    settings: ?[*:0]const u16,
    namespace: ?*?*anyopaque,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnOpenNamespace(
    id: ?*const Guid,
    namespace: ?*?*anyopaque,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnModifyNamespace(
    namespace: ?*anyopaque,
    settings: ?[*:0]const u16,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnQueryNamespaceProperties(
    namespace: ?*anyopaque,
    query: ?[*:0]const u16,
    properties: ?*?PWSTR,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnDeleteNamespace(
    id: ?*const Guid,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnCloseNamespace(
    namespace: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnEnumerateEndpoints(
    query: ?[*:0]const u16,
    endpoints: ?*?PWSTR,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnCreateEndpoint(
    network: ?*anyopaque,
    id: ?*const Guid,
    settings: ?[*:0]const u16,
    endpoint: ?*?*anyopaque,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnOpenEndpoint(
    id: ?*const Guid,
    endpoint: ?*?*anyopaque,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnModifyEndpoint(
    endpoint: ?*anyopaque,
    settings: ?[*:0]const u16,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnQueryEndpointProperties(
    endpoint: ?*anyopaque,
    query: ?[*:0]const u16,
    properties: ?*?PWSTR,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnDeleteEndpoint(
    id: ?*const Guid,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnCloseEndpoint(
    endpoint: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnEnumerateLoadBalancers(
    query: ?[*:0]const u16,
    load_balancer: ?*?PWSTR,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnCreateLoadBalancer(
    id: ?*const Guid,
    settings: ?[*:0]const u16,
    load_balancer: ?*?*anyopaque,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnOpenLoadBalancer(
    id: ?*const Guid,
    load_balancer: ?*?*anyopaque,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnModifyLoadBalancer(
    load_balancer: ?*anyopaque,
    settings: ?[*:0]const u16,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnQueryLoadBalancerProperties(
    load_balancer: ?*anyopaque,
    query: ?[*:0]const u16,
    properties: ?*?PWSTR,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnDeleteLoadBalancer(
    id: ?*const Guid,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnCloseLoadBalancer(
    load_balancer: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnRegisterServiceCallback(
    callback: ?HCN_NOTIFICATION_CALLBACK,
    context: ?*anyopaque,
    callback_handle: ?*?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnUnregisterServiceCallback(
    callback_handle: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnRegisterGuestNetworkServiceCallback(
    guest_network_service: ?*anyopaque,
    callback: ?HCN_NOTIFICATION_CALLBACK,
    context: ?*anyopaque,
    callback_handle: ?*?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnUnregisterGuestNetworkServiceCallback(
    callback_handle: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnCreateGuestNetworkService(
    id: ?*const Guid,
    settings: ?[*:0]const u16,
    guest_network_service: ?*?*anyopaque,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnCloseGuestNetworkService(
    guest_network_service: ?*anyopaque,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnModifyGuestNetworkService(
    guest_network_service: ?*anyopaque,
    settings: ?[*:0]const u16,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnDeleteGuestNetworkService(
    id: ?*const Guid,
    error_record: ?*?PWSTR,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnReserveGuestNetworkServicePort(
    guest_network_service: ?*anyopaque,
    protocol: HCN_PORT_PROTOCOL,
    access: HCN_PORT_ACCESS,
    port: u16,
    port_reservation_handle: ?*?HANDLE,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnReserveGuestNetworkServicePortRange(
    guest_network_service: ?*anyopaque,
    port_count: u16,
    port_range_reservation: ?*HCN_PORT_RANGE_RESERVATION,
    port_reservation_handle: ?*?HANDLE,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnReleaseGuestNetworkServicePortReservationHandle(
    port_reservation_handle: ?HANDLE,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnEnumerateGuestNetworkPortReservations(
    return_count: ?*u32,
    port_entries: ?*?*HCN_PORT_RANGE_ENTRY,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

pub extern "computenetwork" fn HcnFreeGuestNetworkPortReservations(
    port_entries: ?*HCN_PORT_RANGE_ENTRY,
) callconv(@import("std").os.windows.WINAPI) void;

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
// Section: Imports (4)
//--------------------------------------------------------------------------------
const Guid = @import("../zig.zig").Guid;
const HANDLE = @import("../foundation.zig").HANDLE;
const HRESULT = @import("../foundation.zig").HRESULT;
const PWSTR = @import("../foundation.zig").PWSTR;

test {
    // The following '_ = <FuncPtrType>' lines are a workaround for https://github.com/ziglang/zig/issues/4476
    if (@hasDecl(@This(), "HCN_NOTIFICATION_CALLBACK")) {
        _ = HCN_NOTIFICATION_CALLBACK;
    }

    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
