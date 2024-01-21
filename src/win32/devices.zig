//! NOTE: this file is autogenerated, DO NOT MODIFY
pub const all_joyn = @import("devices/all_joyn.zig");
pub const biometric_framework = @import("devices/biometric_framework.zig");
pub const bluetooth = @import("devices/bluetooth.zig");
pub const communication = @import("devices/communication.zig");
pub const device_access = @import("devices/device_access.zig");
pub const device_and_driver_installation = @import("devices/device_and_driver_installation.zig");
pub const device_query = @import("devices/device_query.zig");
pub const display = @import("devices/display.zig");
pub const enumeration = @import("devices/enumeration.zig");
pub const fax = @import("devices/fax.zig");
pub const function_discovery = @import("devices/function_discovery.zig");
pub const geolocation = @import("devices/geolocation.zig");
pub const human_interface_device = @import("devices/human_interface_device.zig");
pub const image_acquisition = @import("devices/image_acquisition.zig");
pub const portable_devices = @import("devices/portable_devices.zig");
pub const properties = @import("devices/properties.zig");
pub const pwm = @import("devices/pwm.zig");
pub const sensors = @import("devices/sensors.zig");
pub const serial_communication = @import("devices/serial_communication.zig");
pub const tapi = @import("devices/tapi.zig");
pub const usb = @import("devices/usb.zig");
pub const web_services_on_devices = @import("devices/web_services_on_devices.zig");
test {
    @import("std").testing.refAllDecls(@This());
}
