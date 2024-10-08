//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (28)
//--------------------------------------------------------------------------------
pub const szOID_CATALOG_LIST = "1.3.6.1.4.1.311.12.1.1";
pub const szOID_CATALOG_LIST_MEMBER = "1.3.6.1.4.1.311.12.1.2";
pub const szOID_CATALOG_LIST_MEMBER2 = "1.3.6.1.4.1.311.12.1.3";
pub const CRYPTCAT_FILEEXT = "CAT";
pub const CRYPTCAT_MAX_MEMBERTAG = @as(u32, 64);
pub const CRYPTCAT_MEMBER_SORTED = @as(u32, 1073741824);
pub const CRYPTCAT_ATTR_AUTHENTICATED = @as(u32, 268435456);
pub const CRYPTCAT_ATTR_UNAUTHENTICATED = @as(u32, 536870912);
pub const CRYPTCAT_ATTR_NAMEASCII = @as(u32, 1);
pub const CRYPTCAT_ATTR_NAMEOBJID = @as(u32, 2);
pub const CRYPTCAT_ATTR_DATAASCII = @as(u32, 65536);
pub const CRYPTCAT_ATTR_DATABASE64 = @as(u32, 131072);
pub const CRYPTCAT_ATTR_DATAREPLACE = @as(u32, 262144);
pub const CRYPTCAT_ATTR_NO_AUTO_COMPAT_ENTRY = @as(u32, 16777216);
pub const CRYPTCAT_E_AREA_HEADER = @as(u32, 0);
pub const CRYPTCAT_E_AREA_MEMBER = @as(u32, 65536);
pub const CRYPTCAT_E_AREA_ATTRIBUTE = @as(u32, 131072);
pub const CRYPTCAT_E_CDF_UNSUPPORTED = @as(u32, 1);
pub const CRYPTCAT_E_CDF_DUPLICATE = @as(u32, 2);
pub const CRYPTCAT_E_CDF_TAGNOTFOUND = @as(u32, 4);
pub const CRYPTCAT_E_CDF_MEMBER_FILE_PATH = @as(u32, 65537);
pub const CRYPTCAT_E_CDF_MEMBER_INDIRECTDATA = @as(u32, 65538);
pub const CRYPTCAT_E_CDF_MEMBER_FILENOTFOUND = @as(u32, 65540);
pub const CRYPTCAT_E_CDF_BAD_GUID_CONV = @as(u32, 131073);
pub const CRYPTCAT_E_CDF_ATTR_TOOFEWVALUES = @as(u32, 131074);
pub const CRYPTCAT_E_CDF_ATTR_TYPECOMBO = @as(u32, 131076);
pub const CRYPTCAT_ADDCATALOG_NONE = @as(u32, 0);
pub const CRYPTCAT_ADDCATALOG_HARDLINK = @as(u32, 1);

//--------------------------------------------------------------------------------
// Section: Types (8)
//--------------------------------------------------------------------------------
pub const CRYPTCAT_VERSION = enum(u32) {
    @"1" = 256,
    @"2" = 512,
};
pub const CRYPTCAT_VERSION_1 = CRYPTCAT_VERSION.@"1";
pub const CRYPTCAT_VERSION_2 = CRYPTCAT_VERSION.@"2";

pub const CRYPTCAT_OPEN_FLAGS = enum(u32) {
    ALWAYS = 2,
    CREATENEW = 1,
    EXISTING = 4,
    EXCLUDE_PAGE_HASHES = 65536,
    INCLUDE_PAGE_HASHES = 131072,
    VERIFYSIGHASH = 268435456,
    NO_CONTENT_HCRYPTMSG = 536870912,
    SORTED = 1073741824,
    FLAGS_MASK = 4294901760,
    _,
    pub fn initFlags(o: struct {
        ALWAYS: u1 = 0,
        CREATENEW: u1 = 0,
        EXISTING: u1 = 0,
        EXCLUDE_PAGE_HASHES: u1 = 0,
        INCLUDE_PAGE_HASHES: u1 = 0,
        VERIFYSIGHASH: u1 = 0,
        NO_CONTENT_HCRYPTMSG: u1 = 0,
        SORTED: u1 = 0,
        FLAGS_MASK: u1 = 0,
    }) CRYPTCAT_OPEN_FLAGS {
        return @as(CRYPTCAT_OPEN_FLAGS, @enumFromInt((if (o.ALWAYS == 1) @intFromEnum(CRYPTCAT_OPEN_FLAGS.ALWAYS) else 0) | (if (o.CREATENEW == 1) @intFromEnum(CRYPTCAT_OPEN_FLAGS.CREATENEW) else 0) | (if (o.EXISTING == 1) @intFromEnum(CRYPTCAT_OPEN_FLAGS.EXISTING) else 0) | (if (o.EXCLUDE_PAGE_HASHES == 1) @intFromEnum(CRYPTCAT_OPEN_FLAGS.EXCLUDE_PAGE_HASHES) else 0) | (if (o.INCLUDE_PAGE_HASHES == 1) @intFromEnum(CRYPTCAT_OPEN_FLAGS.INCLUDE_PAGE_HASHES) else 0) | (if (o.VERIFYSIGHASH == 1) @intFromEnum(CRYPTCAT_OPEN_FLAGS.VERIFYSIGHASH) else 0) | (if (o.NO_CONTENT_HCRYPTMSG == 1) @intFromEnum(CRYPTCAT_OPEN_FLAGS.NO_CONTENT_HCRYPTMSG) else 0) | (if (o.SORTED == 1) @intFromEnum(CRYPTCAT_OPEN_FLAGS.SORTED) else 0) | (if (o.FLAGS_MASK == 1) @intFromEnum(CRYPTCAT_OPEN_FLAGS.FLAGS_MASK) else 0)));
    }
};
pub const CRYPTCAT_OPEN_ALWAYS = CRYPTCAT_OPEN_FLAGS.ALWAYS;
pub const CRYPTCAT_OPEN_CREATENEW = CRYPTCAT_OPEN_FLAGS.CREATENEW;
pub const CRYPTCAT_OPEN_EXISTING = CRYPTCAT_OPEN_FLAGS.EXISTING;
pub const CRYPTCAT_OPEN_EXCLUDE_PAGE_HASHES = CRYPTCAT_OPEN_FLAGS.EXCLUDE_PAGE_HASHES;
pub const CRYPTCAT_OPEN_INCLUDE_PAGE_HASHES = CRYPTCAT_OPEN_FLAGS.INCLUDE_PAGE_HASHES;
pub const CRYPTCAT_OPEN_VERIFYSIGHASH = CRYPTCAT_OPEN_FLAGS.VERIFYSIGHASH;
pub const CRYPTCAT_OPEN_NO_CONTENT_HCRYPTMSG = CRYPTCAT_OPEN_FLAGS.NO_CONTENT_HCRYPTMSG;
pub const CRYPTCAT_OPEN_SORTED = CRYPTCAT_OPEN_FLAGS.SORTED;
pub const CRYPTCAT_OPEN_FLAGS_MASK = CRYPTCAT_OPEN_FLAGS.FLAGS_MASK;

pub const CRYPTCATSTORE = extern struct {
    cbStruct: u32,
    dwPublicVersion: u32,
    pwszP7File: ?PWSTR,
    hProv: usize,
    dwEncodingType: u32,
    fdwStoreFlags: CRYPTCAT_OPEN_FLAGS,
    hReserved: ?HANDLE,
    hAttrs: ?HANDLE,
    hCryptMsg: ?*anyopaque,
    hSorted: ?HANDLE,
};

pub const CRYPTCATMEMBER = extern struct {
    cbStruct: u32,
    pwszReferenceTag: ?PWSTR,
    pwszFileName: ?PWSTR,
    gSubjectType: Guid,
    fdwMemberFlags: u32,
    pIndirectData: ?*SIP_INDIRECT_DATA,
    dwCertVersion: u32,
    dwReserved: u32,
    hReserved: ?HANDLE,
    sEncodedIndirectData: CRYPTOAPI_BLOB,
    sEncodedMemberInfo: CRYPTOAPI_BLOB,
};

pub const CRYPTCATATTRIBUTE = extern struct {
    cbStruct: u32,
    pwszReferenceTag: ?PWSTR,
    dwAttrTypeAndAction: u32,
    cbValue: u32,
    pbValue: ?*u8,
    dwReserved: u32,
};

pub const CRYPTCATCDF = extern struct {
    cbStruct: u32,
    hFile: ?HANDLE,
    dwCurFilePos: u32,
    dwLastMemberOffset: u32,
    fEOF: BOOL,
    pwszResultDir: ?PWSTR,
    hCATStore: ?HANDLE,
};

pub const CATALOG_INFO = extern struct {
    cbStruct: u32,
    wszCatalogFile: [260]u16,
};

pub const PFN_CDF_PARSE_ERROR_CALLBACK = *const fn (
    dw_error_area: u32,
    dw_local_error: u32,
    pwsz_line: ?PWSTR,
) callconv(@import("std").os.windows.WINAPI) void;

//--------------------------------------------------------------------------------
// Section: Functions (34)
//--------------------------------------------------------------------------------
// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATOpen(
    pwsz_file_name: ?PWSTR,
    fdw_open_flags: CRYPTCAT_OPEN_FLAGS,
    h_prov: usize,
    dw_public_version: CRYPTCAT_VERSION,
    dw_encoding_type: u32,
) callconv(@import("std").os.windows.WINAPI) ?HANDLE;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATClose(
    h_catalog: ?HANDLE,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATStoreFromHandle(
    h_catalog: ?HANDLE,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATSTORE;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATHandleFromStore(
    p_cat_store: ?*CRYPTCATSTORE,
) callconv(@import("std").os.windows.WINAPI) ?HANDLE;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATPersistStore(
    h_catalog: ?HANDLE,
) callconv(@import("std").os.windows.WINAPI) BOOL;

pub extern "wintrust" fn CryptCATGetCatAttrInfo(
    h_catalog: ?HANDLE,
    pwsz_reference_tag: ?PWSTR,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATATTRIBUTE;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATPutCatAttrInfo(
    h_catalog: ?HANDLE,
    pwsz_reference_tag: ?PWSTR,
    dw_attr_type_and_action: u32,
    cb_data: u32,
    pb_data: ?*u8,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATATTRIBUTE;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATEnumerateCatAttr(
    h_catalog: ?HANDLE,
    p_prev_attr: ?*CRYPTCATATTRIBUTE,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATATTRIBUTE;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATGetMemberInfo(
    h_catalog: ?HANDLE,
    pwsz_reference_tag: ?PWSTR,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATMEMBER;

pub extern "wintrust" fn CryptCATAllocSortedMemberInfo(
    h_catalog: ?HANDLE,
    pwsz_reference_tag: ?PWSTR,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATMEMBER;

pub extern "wintrust" fn CryptCATFreeSortedMemberInfo(
    h_catalog: ?HANDLE,
    p_cat_member: ?*CRYPTCATMEMBER,
) callconv(@import("std").os.windows.WINAPI) void;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATGetAttrInfo(
    h_catalog: ?HANDLE,
    p_cat_member: ?*CRYPTCATMEMBER,
    pwsz_reference_tag: ?PWSTR,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATATTRIBUTE;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATPutMemberInfo(
    h_catalog: ?HANDLE,
    pwsz_file_name: ?PWSTR,
    pwsz_reference_tag: ?PWSTR,
    pg_subject_type: ?*Guid,
    dw_cert_version: u32,
    cb_s_i_p_indirect_data: u32,
    pb_s_i_p_indirect_data: ?*u8,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATMEMBER;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATPutAttrInfo(
    h_catalog: ?HANDLE,
    p_cat_member: ?*CRYPTCATMEMBER,
    pwsz_reference_tag: ?PWSTR,
    dw_attr_type_and_action: u32,
    cb_data: u32,
    pb_data: ?*u8,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATATTRIBUTE;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATEnumerateMember(
    h_catalog: ?HANDLE,
    p_prev_member: ?*CRYPTCATMEMBER,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATMEMBER;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATEnumerateAttr(
    h_catalog: ?HANDLE,
    p_cat_member: ?*CRYPTCATMEMBER,
    p_prev_attr: ?*CRYPTCATATTRIBUTE,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATATTRIBUTE;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATCDFOpen(
    pwsz_file_path: ?PWSTR,
    pfn_parse_error: ?PFN_CDF_PARSE_ERROR_CALLBACK,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATCDF;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATCDFClose(
    p_c_d_f: ?*CRYPTCATCDF,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATCDFEnumCatAttributes(
    p_c_d_f: ?*CRYPTCATCDF,
    p_prev_attr: ?*CRYPTCATATTRIBUTE,
    pfn_parse_error: ?PFN_CDF_PARSE_ERROR_CALLBACK,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATATTRIBUTE;

pub extern "wintrust" fn CryptCATCDFEnumMembers(
    p_c_d_f: ?*CRYPTCATCDF,
    p_prev_member: ?*CRYPTCATMEMBER,
    pfn_parse_error: ?PFN_CDF_PARSE_ERROR_CALLBACK,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATMEMBER;

pub extern "wintrust" fn CryptCATCDFEnumAttributes(
    p_c_d_f: ?*CRYPTCATCDF,
    p_member: ?*CRYPTCATMEMBER,
    p_prev_attr: ?*CRYPTCATATTRIBUTE,
    pfn_parse_error: ?PFN_CDF_PARSE_ERROR_CALLBACK,
) callconv(@import("std").os.windows.WINAPI) ?*CRYPTCATATTRIBUTE;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn IsCatalogFile(
    h_file: ?HANDLE,
    pwsz_file_name: ?PWSTR,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATAdminAcquireContext(
    ph_cat_admin: ?*isize,
    pg_subsystem: ?*const Guid,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows8.0'
pub extern "wintrust" fn CryptCATAdminAcquireContext2(
    ph_cat_admin: ?*isize,
    pg_subsystem: ?*const Guid,
    pwsz_hash_algorithm: ?[*:0]const u16,
    p_strong_hash_policy: ?*CERT_STRONG_SIGN_PARA,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATAdminReleaseContext(
    h_cat_admin: isize,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATAdminReleaseCatalogContext(
    h_cat_admin: isize,
    h_cat_info: isize,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATAdminEnumCatalogFromHash(
    h_cat_admin: isize,
    // TODO: what to do with BytesParamIndex 2?
    pb_hash: ?*u8,
    cb_hash: u32,
    dw_flags: u32,
    ph_prev_cat_info: ?*isize,
) callconv(@import("std").os.windows.WINAPI) isize;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATAdminCalcHashFromFileHandle(
    h_file: ?HANDLE,
    pcb_hash: ?*u32,
    // TODO: what to do with BytesParamIndex 1?
    pb_hash: ?*u8,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows8.0'
pub extern "wintrust" fn CryptCATAdminCalcHashFromFileHandle2(
    h_cat_admin: isize,
    h_file: ?HANDLE,
    pcb_hash: ?*u32,
    // TODO: what to do with BytesParamIndex 2?
    pb_hash: ?*u8,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATAdminAddCatalog(
    h_cat_admin: isize,
    pwsz_catalog_file: ?PWSTR,
    pwsz_select_base_name: ?PWSTR,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) isize;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATAdminRemoveCatalog(
    h_cat_admin: isize,
    pwsz_catalog_file: ?[*:0]const u16,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATCatalogInfoFromContext(
    h_cat_info: isize,
    ps_cat_info: ?*CATALOG_INFO,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

// TODO: this type is limited to platform 'windows5.1.2600'
pub extern "wintrust" fn CryptCATAdminResolveCatalogPath(
    h_cat_admin: isize,
    pwsz_catalog_file: ?PWSTR,
    ps_cat_info: ?*CATALOG_INFO,
    dw_flags: u32,
) callconv(@import("std").os.windows.WINAPI) BOOL;

pub extern "wintrust" fn CryptCATAdminPauseServiceForBackup(
    dw_flags: u32,
    f_resume: BOOL,
) callconv(@import("std").os.windows.WINAPI) BOOL;

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
// Section: Imports (7)
//--------------------------------------------------------------------------------
const Guid = @import("../../zig.zig").Guid;
const BOOL = @import("../../foundation.zig").BOOL;
const CERT_STRONG_SIGN_PARA = @import("../../security/cryptography.zig").CERT_STRONG_SIGN_PARA;
const CRYPTOAPI_BLOB = @import("../../security/cryptography.zig").CRYPTOAPI_BLOB;
const HANDLE = @import("../../foundation.zig").HANDLE;
const PWSTR = @import("../../foundation.zig").PWSTR;
const SIP_INDIRECT_DATA = @import("../../security/cryptography/sip.zig").SIP_INDIRECT_DATA;

test {
    // The following '_ = <FuncPtrType>' lines are a workaround for https://github.com/ziglang/zig/issues/4476
    if (@hasDecl(@This(), "PFN_CDF_PARSE_ERROR_CALLBACK")) {
        _ = PFN_CDF_PARSE_ERROR_CALLBACK;
    }

    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
