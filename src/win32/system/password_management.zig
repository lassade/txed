//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (0)
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// Section: Types (4)
//--------------------------------------------------------------------------------
pub const CYPHER_BLOCK = extern struct {
    data: [8]CHAR,
};

pub const LM_OWF_PASSWORD = extern struct {
    data: [2]CYPHER_BLOCK,
};

pub const SAMPR_ENCRYPTED_USER_PASSWORD = extern struct {
    Buffer: [516]u8,
};

pub const ENCRYPTED_LM_OWF_PASSWORD = extern struct {
    data: [2]CYPHER_BLOCK,
};

//--------------------------------------------------------------------------------
// Section: Functions (2)
//--------------------------------------------------------------------------------
// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "advapi32" fn MSChapSrvChangePassword(
    server_name: ?PWSTR,
    user_name: ?PWSTR,
    lm_old_present: BOOLEAN,
    lm_old_owf_password: ?*LM_OWF_PASSWORD,
    lm_new_owf_password: ?*LM_OWF_PASSWORD,
    nt_old_owf_password: ?*LM_OWF_PASSWORD,
    nt_new_owf_password: ?*LM_OWF_PASSWORD,
) callconv(@import("std").os.windows.WINAPI) u32;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "advapi32" fn MSChapSrvChangePassword2(
    server_name: ?PWSTR,
    user_name: ?PWSTR,
    new_password_encrypted_with_old_nt: ?*SAMPR_ENCRYPTED_USER_PASSWORD,
    old_nt_owf_password_encrypted_with_new_nt: ?*ENCRYPTED_LM_OWF_PASSWORD,
    lm_present: BOOLEAN,
    new_password_encrypted_with_old_lm: ?*SAMPR_ENCRYPTED_USER_PASSWORD,
    old_lm_owf_password_encrypted_with_new_lm_or_nt: ?*ENCRYPTED_LM_OWF_PASSWORD,
) callconv(@import("std").os.windows.WINAPI) u32;

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
// Section: Imports (3)
//--------------------------------------------------------------------------------
const BOOLEAN = @import("../foundation.zig").BOOLEAN;
const CHAR = @import("../foundation.zig").CHAR;
const PWSTR = @import("../foundation.zig").PWSTR;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
