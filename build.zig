const std = @import("std");
const Step = std.Build.Step;
const LazyPath = std.Build.LazyPath;

const win = @import("src/win32.zig");
const L = win.zig.L;
const FAILED = win.zig.FAILED;

pub fn build(b: *std.Build) !void {
    // windows only
    const target = b.resolveTargetQuery(.{
        .cpu_arch = .x86_64,
        .os_tag = .windows,
        .abi = .msvc,
    });
    const optimize = b.standardOptimizeOption(.{});

    b.exe_dir = b.pathJoin(&.{ "zig-out", @tagName(optimize) });

    const wsdk = try WinSdk.init(b);

    const exe = b.addExecutable(.{
        .name = "txed",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });
    // exe.subsystem = .Windows;
    exe.linkLibC();
    //if (optimize == .ReleaseFast) exe.strip = true;

    exe.addCSourceFile(.{ .file = .{ .path = "src/stb/stb.c" }, .flags = &.{"-O3"} });

    // exe.addLibraryPath(.{ .path = b.bmt("{s}\\um\\x64", wsdk.lib) });
    // exe.linkSystemLibrary("d3d12");
    // exe.linkSystemLibrary("dxgi");
    // exe.linkSystemLibrary("d3dcompiler");

    // const dxc_step = b.step("dxc", "Build shaders");
    // dxc_step.dependOn(wsdk.compileShader(b, .{ .path = "src/shaders.hlsl" }, "shader.vs.cso", "VSMain", .vs_6_0, &.{}));
    // dxc_step.dependOn(wsdk.compileShader(b, .{ .path = "src/shaders.hlsl" }, "shader.ps.cso", "PSMain", .ps_6_0, &.{}));

    wsdk.embedShader(b, .{ .path = "src/shaders.hlsl" }, exe, "shader.vs.cso", "VSMain", .vs_6_0, &.{});
    wsdk.embedShader(b, .{ .path = "src/shaders.hlsl" }, exe, "shader.ps.cso", "PSMain", .ps_6_0, &.{});

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "./src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}

const WinSdk = struct {
    bin: []u8,
    lib: []u8,

    dxc: []u8,

    fn init(b: *std.Build) !WinSdk {
        var hkey: win.system.registry.HKEY = undefined;
        if (win.system.registry.RegOpenKeyExA(
            win.system.registry.HKEY_LOCAL_MACHINE,
            "SOFTWARE\\WOW6432Node\\Microsoft\\Microsoft SDKs\\Windows\\v10.0",
            0,
            .READ,
            @ptrCast(&hkey),
        ) != .NO_ERROR) {
            return error.WinSdkNotFound;
        }
        defer _ = win.system.registry.RegCloseKey(hkey);

        var path: [256]u8 = undefined;
        var path_len: u32 = @intCast(path.len);
        if (win.system.registry.RegQueryValueExA(
            hkey,
            "InstallationFolder",
            null,
            null,
            @ptrCast(&path),
            &path_len,
        ) != .NO_ERROR) {
            return error.WinSdkInstallationFolderValueNotFound;
        }
        path_len -= 1; // account for the null terminator

        var ver: [32]u8 = undefined;
        var ver_len: u32 = @intCast(ver.len);
        if (win.system.registry.RegQueryValueExA(
            hkey,
            "ProductVersion",
            null,
            null,
            @ptrCast(&ver),
            &ver_len,
        ) != .NO_ERROR) {
            return error.WinSdkProductVersionValueNotFound;
        }
        ver_len -= 1; // account for the null terminator

        const bin = b.fmt("{s}bin\\{s}.0", .{ path[0..path_len], ver[0..ver_len] });

        return .{
            .bin = bin,
            .lib = b.fmt("{s}Lib\\{s}.0", .{ path[0..path_len], ver[0..ver_len] }),
            .dxc = b.fmt("{s}\\x64\\dxc.exe", .{bin}),
        };
    }

    pub fn compileShader(
        self: *const WinSdk,
        b: *std.Build,
        input: LazyPath,
        output: []const u8,
        entry_point: []const u8,
        profile: ShaderProfile,
        defines: []const []const u8,
    ) *Step {
        var compile = b.addSystemCommand(&[_][]const u8{self.dxc});
        compile.setName("dxc");
        compile.addFileArg(input);
        compile.addArg("/E");
        compile.addArg(entry_point);
        compile.addArg("/Fo");
        const output_file = compile.addOutputFileArg(std.fs.path.basename(output));
        compile.addArg("/T");
        compile.addArg(@tagName(profile));
        for (defines) |def| {
            compile.addArg("/D");
            compile.addArg(def);
        }
        compile.addArg("/WX");
        compile.addArg("/Ges");
        compile.addArg("/O3");

        // todo: on debug builds skip optimizations and compile debug

        var install = b.addInstallFileWithDir(output_file, .bin, output);
        return &install.step;
    }

    pub fn embedShader(
        self: *const WinSdk,
        b: *std.Build,
        input: LazyPath,
        target: *Step.Compile,
        name: []const u8,
        entry_point: []const u8,
        profile: ShaderProfile,
        defines: []const []const u8,
    ) void {
        var compile = b.addSystemCommand(&[_][]const u8{self.dxc});
        compile.setName("dxc");
        compile.addFileArg(input);
        compile.addArg("/E");
        compile.addArg(entry_point);
        compile.addArg("/Fo");
        const output_file = compile.addOutputFileArg(name);
        compile.addArg("/T");
        compile.addArg(@tagName(profile));
        for (defines) |def| {
            compile.addArg("/D");
            compile.addArg(def);
        }
        compile.addArg("/WX");
        compile.addArg("/Ges");
        compile.addArg("/O3");

        // todo: on debug builds skip optimizations and compile debug

        target.root_module.addAnonymousImport(name, .{ .root_source_file = output_file });
        //target.step.dependOn(&compile.step);
    }

    pub const ShaderProfile = enum { vs_6_0, ps_6_0 };
};
