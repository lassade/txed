const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const log = std.log;

const win = @import("win32.zig");
const wf = win.foundation;
const wm = win.ui.windows_and_messaging;
const kbm = win.ui.input.keyboard_and_mouse;
const gdi = win.graphics.gdi;
const dx = win.graphics.direct3d;
const dx12 = win.graphics.direct3d12;
const dxgi = win.graphics.dxgi;
const wz = win.zig;
const L = win.zig.L;

const stb = @import("stb.zig");
const tt = stb.truetype;

const help = @import("help.zig");
const hrErrorOnFail = help.hrErrorOnFail;
const GPUBuffer = help.GPUBuffer;
const GPUStagingBuffer = help.GPUStagingBuffer;
const GPUDescHeap = help.GPUDescHeap;

const TextFile = @import("TextFile.zig");

const Font = struct {
    info: tt.FontInfo,
    data: []u8,

    fn initSystemFont(allocator: Allocator, hwnd: wf.HWND, font_name: [:0]const u8) !Font {
        var ps: gdi.PAINTSTRUCT = undefined;
        const hdc = gdi.BeginPaint(hwnd, &ps);
        const hfont = gdi.CreateFontA(
            12,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            .DEFAULT_PRECIS,
            .DEFAULT_PRECIS,
            .DEFAULT_QUALITY,
            .DONTCARE,
            font_name,
        );
        if (hfont == null) return error.FontNotFound;

        _ = gdi.SelectObject(hdc, hfont);
        const fond_data_size = gdi.GetFontData(hdc, 0, 0, null, 0);

        const data = try allocator.alloc(u8, fond_data_size);
        _ = gdi.GetFontData(hdc, 0, 0, @ptrCast(data.ptr), fond_data_size);
        _ = gdi.EndPaint(hwnd, &ps);

        return Font{
            .info = try tt.FontInfo.init(data, 0),
            .data = data,
        };
    }

    fn deinit(self: *Font, allocator: Allocator) void {
        allocator.free(self.data);
    }
};

const App = struct {
    const frame_count = 2;

    const Vert = extern struct {
        pos: [3]f32,
        uv: [2]f32,
    };

    const ColorEnum = enum(u4) {
        text = 0,
        bg,
    };

    const Char = packed struct {
        index: u32 = 0,

        // colors
        fg_color: ColorEnum = .text,
        bg_color: ColorEnum = .bg,

        // various styles
        cursor_line: bool = false,
        cursor_block: bool = false,
        box: bool = false,

        scope: bool = false,
        scope_hint: bool = false,

        selected: bool = false,
        line_over: bool = false,

        unused: u17 = 0,
    };

    const Config = extern struct {
        slot_size: [2]u32,
        font_atlas_size: [2]u32,
        console_size: [2]u32,
        // note aligment requirement
        colors: [16]@Vector(4, f32),
    };

    const ChangedFlags = packed struct(u32) {
        console: bool = false,
        file_view: bool = false,
        // todo: line: bool = false,
        unused: u30 = 0,
    };

    allocator: Allocator,

    hwnd: wf.HWND,

    screen_size: [2]u32,
    aspect_ratio: f32,
    viewport: dx12.D3D12_VIEWPORT,
    scissor_rect: wf.RECT,

    changed: ChangedFlags = .{},

    device: *dx12.ID3D12Device,
    command_queue: *dx12.ID3D12CommandQueue,
    swap_chain: *dxgi.IDXGISwapChain3,
    srv_heap: GPUDescHeap,
    rtv_heap: *dx12.ID3D12DescriptorHeap,
    rtv_desc_size: u32,
    render_targets: [frame_count]*dx12.ID3D12Resource,
    command_allocator: *dx12.ID3D12CommandAllocator,
    command_list: *dx12.ID3D12GraphicsCommandList,

    frame_index: u32,
    fence_event: wf.HANDLE,
    fence: *dx12.ID3D12Fence,
    fence_value: u64,

    staging_buffers: [frame_count]GPUStagingBuffer,

    root_sig: *dx12.ID3D12RootSignature,
    pipeline_state: *dx12.ID3D12PipelineState,

    font: Font,
    font_size: f32,
    font_scale: f32,
    slot_size: [2]u32,
    font_atlas: *dx12.ID3D12Resource,
    font_atlas_desc_offset: u64,
    font_atlas_size: [2]u32,
    font_atlas_slot_pos: [2]u32,
    font_atlas_slot_count_per_dim: [2]u32,
    font_atlas_slot_count: u32,
    unicode_map: std.ArrayListUnmanaged(u32),

    config_buffer: GPUBuffer,

    console: []Char,
    console_buffer: GPUBuffer,
    console_size: [2]u32,

    console_output: *dx12.ID3D12Resource,
    console_output_desc_offset: u64,

    files: std.ArrayListUnmanaged(TextFile) = .{},
    file_index: u32 = 0xffffffff,

    fn init(allocator: Allocator, hwnd: wf.HWND, size: [2]u32) !App {
        var dxgi_factory_flags: u32 = 0;

        if (builtin.mode == .Debug) {
            var debug_ctrl: *dx12.ID3D12Debug = undefined;
            if (wz.SUCCEEDED(dx12.D3D12GetDebugInterface(dx12.IID_ID3D12Debug, @ptrCast(&debug_ctrl)))) {
                debug_ctrl.enableDebugLayer();

                // enable additional debug layers.
                dxgi_factory_flags |= dxgi.DXGI_CREATE_FACTORY_DEBUG;
                _ = debug_ctrl.release();
            }
        }

        var factory: *dxgi.IDXGIFactory4 = undefined;
        try hrErrorOnFail(dxgi.CreateDXGIFactory2(
            dxgi_factory_flags,
            dxgi.IID_IDXGIFactory4,
            @ptrCast(&factory),
        ));
        defer _ = factory.release();

        var hardwareAdapter: *dxgi.IDXGIAdapter1 = undefined;
        getHardwareAdapter: {
            var adapterIndex: u32 = undefined;
            var adapter: *dxgi.IDXGIAdapter1 = undefined;

            var factory6: *dxgi.IDXGIFactory6 = undefined;
            if (wz.SUCCEEDED(factory.queryInterface(dxgi.IID_IDXGIFactory6, @ptrCast(&factory6)))) {
                defer _ = factory6.release();

                adapterIndex = 0;
                while (wz.SUCCEEDED(factory6.enumAdapterByGpuPreference(
                    adapterIndex,
                    dxgi.DXGI_GPU_PREFERENCE_UNSPECIFIED,
                    dxgi.IID_IDXGIAdapter1,
                    @ptrCast(&adapter),
                ))) : (adapterIndex += 1) {
                    var desc: dxgi.DXGI_ADAPTER_DESC1 = undefined;
                    try hrErrorOnFail(adapter.getDesc1(&desc));

                    // don't select the Basic Render Driver adapter
                    // if you want a software adapter, pass in "/warp" on the command line
                    if ((desc.Flags & @as(u32, @intFromEnum(dxgi.DXGI_ADAPTER_FLAG_SOFTWARE))) != 0) {
                        // check to see whether the adapter supports Direct3D 12, but don't create the actual device yet.
                        if (wz.SUCCEEDED(dx12.D3D12CreateDevice(@ptrCast(adapter), dx.D3D_FEATURE_LEVEL_11_0, dx12.IID_ID3D12Device, null))) {
                            hardwareAdapter = adapter;
                            break :getHardwareAdapter;
                        }
                    }

                    // release the unused adapter
                    _ = adapter.release();
                }
            }

            // fallback method
            adapterIndex = 0;
            while (wz.SUCCEEDED(factory.enumAdapters1(adapterIndex, @ptrCast(&adapter)))) : (adapterIndex += 1) {
                var desc: dxgi.DXGI_ADAPTER_DESC1 = undefined;
                _ = adapter.getDesc1(&desc);

                if ((desc.Flags & @as(u32, @intFromEnum(dxgi.DXGI_ADAPTER_FLAG_SOFTWARE))) != 0) {
                    if (wz.SUCCEEDED(dx12.D3D12CreateDevice(@ptrCast(adapter), dx.D3D_FEATURE_LEVEL_11_0, dx12.IID_ID3D12Device, null))) {
                        hardwareAdapter = adapter;
                        break :getHardwareAdapter;
                    }
                }

                _ = adapter.release();
            }
        }
        defer _ = hardwareAdapter.release();

        var device: *dx12.ID3D12Device = undefined;
        try hrErrorOnFail(dx12.D3D12CreateDevice(
            @ptrCast(hardwareAdapter),
            dx.D3D_FEATURE_LEVEL_11_0,
            dx12.IID_ID3D12Device,
            @ptrCast(&device),
        ));
        errdefer _ = device.release();

        const queue_desc = dx12.D3D12_COMMAND_QUEUE_DESC{
            .Type = .DIRECT,
            .Priority = 0,
            .Flags = .NONE,
            .NodeMask = 0,
        };
        var command_queue: *dx12.ID3D12CommandQueue = undefined;
        try hrErrorOnFail(device.createCommandQueue(
            &queue_desc,
            dx12.IID_ID3D12CommandQueue,
            @ptrCast(&command_queue),
        ));
        errdefer _ = command_queue.release();

        var swap_chain3: *dxgi.IDXGISwapChain3 = undefined;
        {
            var swap_chain: *dxgi.IDXGISwapChain1 = undefined;
            try hrErrorOnFail(factory.createSwapChainForHwnd(
                @ptrCast(command_queue), // Swap chain needs the queue so that it can force a flush on it.
                hwnd,
                &dxgi.DXGI_SWAP_CHAIN_DESC1{
                    .Width = size[0],
                    .Height = size[1],
                    .Format = .R8G8B8A8_UNORM,
                    .Stereo = 0,
                    .SampleDesc = .{
                        .Count = 1,
                        .Quality = 0,
                    },
                    .BufferUsage = dxgi.DXGI_USAGE_RENDER_TARGET_OUTPUT,
                    .BufferCount = frame_count, // duble buffered
                    .Scaling = .STRETCH,
                    .SwapEffect = .FLIP_DISCARD,
                    .AlphaMode = .UNSPECIFIED,
                    .Flags = 0,
                },
                null,
                null,
                @ptrCast(&swap_chain),
            ));
            errdefer _ = swap_chain.release();

            try hrErrorOnFail(swap_chain.queryInterface(dxgi.IID_IDXGISwapChain3, @ptrCast(&swap_chain3)));
            _ = swap_chain.release();
        }

        // this sample does not support fullscreen transitions.
        _ = factory.makeWindowAssociation(hwnd, dxgi.DXGI_MWA_NO_ALT_ENTER);

        // create descriptor heaps.
        // describe and create a render target view (RTV) descriptor heap.
        var rtv_heap: *dx12.ID3D12DescriptorHeap = undefined;
        try hrErrorOnFail(device.createDescriptorHeap(
            &dx12.D3D12_DESCRIPTOR_HEAP_DESC{
                .Type = .RTV,
                .NumDescriptors = frame_count,
                .Flags = .NONE,
                .NodeMask = 0,
            },
            dx12.IID_ID3D12DescriptorHeap,
            @ptrCast(&rtv_heap),
        ));
        errdefer _ = rtv_heap.release();

        const rtv_desc_size = device.getDescriptorHandleIncrementSize(.RTV);

        // create frame resources.
        var render_targets: [2]*dx12.ID3D12Resource = undefined;
        var render_targets_len: usize = 0;
        errdefer {
            for (0..render_targets_len) |i| {
                _ = render_targets[i].release();
            }
        }
        var rtv_handle = rtv_heap.getCPUDescriptorHandleForHeapStart();
        // create a RTV for each frame.
        for (0..frame_count) |n| {
            try hrErrorOnFail(swap_chain3.getBuffer(
                @intCast(n),
                dx12.IID_ID3D12Resource,
                @ptrCast(&render_targets[n]),
            ));
            device.createRenderTargetView(@ptrCast(render_targets[n]), null, rtv_handle);
            rtv_handle.ptr += rtv_desc_size;
            render_targets_len += 1;
        }

        var command_allocator: *dx12.ID3D12CommandAllocator = undefined;
        try hrErrorOnFail(device.createCommandAllocator(
            .DIRECT,
            dx12.IID_ID3D12CommandAllocator,
            @ptrCast(&command_allocator),
        ));
        errdefer _ = command_allocator.release();

        // create an empty root signature.
        var root_sig: *dx12.ID3D12RootSignature = undefined;
        {
            const params = [_]dx12.D3D12_ROOT_PARAMETER{
                dx12.D3D12_ROOT_PARAMETER.initDescriptorTable(&.{dx12.D3D12_DESCRIPTOR_RANGE.init(.CBV, 1, 0)}, .ALL),
                dx12.D3D12_ROOT_PARAMETER.initDescriptorTable(&.{dx12.D3D12_DESCRIPTOR_RANGE.init(.SRV, 1, 0)}, .ALL),
                dx12.D3D12_ROOT_PARAMETER.initDescriptorTable(&.{dx12.D3D12_DESCRIPTOR_RANGE.init(.SRV, 1, 1)}, .ALL),
                dx12.D3D12_ROOT_PARAMETER.initDescriptorTable(&.{dx12.D3D12_DESCRIPTOR_RANGE.init(.UAV, 1, 0)}, .ALL),
            };

            var signature: *dx.ID3DBlob = undefined;
            var err: *dx.ID3DBlob = undefined;
            errdefer _ = err.release(); // otherwise err is null
            try hrErrorOnFail(dx12.D3D12SerializeRootSignature(
                &dx12.D3D12_ROOT_SIGNATURE_DESC{
                    .NumParameters = @intCast(params.len),
                    .pParameters = @ptrCast(&params),
                    .NumStaticSamplers = 0,
                    .pStaticSamplers = null,
                    .Flags = .ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
                },
                .v1_0,
                @ptrCast(&signature),
                @ptrCast(&err),
            ));
            defer _ = signature.release();
            try hrErrorOnFail(device.createRootSignature(
                0,
                @ptrCast(signature.getBufferPointer()),
                signature.getBufferSize(),
                dx12.IID_ID3D12RootSignature,
                @ptrCast(&root_sig),
            ));
        }
        errdefer _ = root_sig.release();

        // load shaders
        var pipeline_state: *dx12.ID3D12PipelineState = undefined;
        {
            const cs_cso = @embedFile("console.cs.cso");
            try hrErrorOnFail(device.createComputePipelineState(
                &dx12.D3D12_COMPUTE_PIPELINE_STATE_DESC{
                    .pRootSignature = root_sig,
                    .CS = .{ .pShaderBytecode = cs_cso, .BytecodeLength = cs_cso.len },
                    .NodeMask = 0,
                    .CachedPSO = .{},
                    .Flags = .NONE,
                },
                dx12.IID_ID3D12PipelineState,
                @ptrCast(&pipeline_state),
            ));
        }
        errdefer _ = pipeline_state.release();

        // create the command list.
        var command_list: *dx12.ID3D12GraphicsCommandList = undefined;
        try hrErrorOnFail(device.createCommandList(
            0,
            .DIRECT,
            command_allocator,
            null,
            dx12.IID_ID3D12GraphicsCommandList,
            @ptrCast(&command_list),
        ));
        errdefer _ = command_list.release();

        // create synchronization objects and wait until assets have been uploaded to the GPU.
        var fence: *dx12.ID3D12Fence = undefined;
        try hrErrorOnFail(device.createFence(0, .NONE, dx12.IID_ID3D12Fence, @ptrCast(&fence)));
        errdefer _ = fence.release();

        const fence_value: u64 = 1;

        // create an event handle to use for frame synchronization.
        const fence_event = win.system.threading.CreateEventA(null, wz.FALSE, wz.FALSE, null) orelse {
            //if (wf.GetLastError() != .NO_ERROR) {}
            return error.CreateEventFailed;
        };
        errdefer _ = wf.CloseHandle(fence_event);

        var staging_buffers: [frame_count]GPUStagingBuffer = undefined;
        var staging_buffers_len: usize = 0;
        errdefer {
            for (0..staging_buffers_len) |i| {
                staging_buffers[i].deinit();
            }
        }
        for (0..frame_count) |i| {
            staging_buffers[i] = try GPUStagingBuffer.init(device, 64 * 1024);
            staging_buffers_len += 1;
        }

        var self = App{
            .allocator = allocator,

            .hwnd = hwnd,

            .screen_size = size,
            .aspect_ratio = @as(f32, @floatFromInt(size[0])) / @as(f32, @floatFromInt(size[1])),
            .viewport = .{
                .TopLeftX = 0.0,
                .TopLeftY = 0.0,
                .Width = @floatFromInt(size[0]),
                .Height = @floatFromInt(size[1]),
                .MinDepth = 0.0,
                .MaxDepth = 1.0,
            },
            .scissor_rect = .{
                .left = 0,
                .top = 0,
                .right = @intCast(size[0]),
                .bottom = @intCast(size[0]),
            },

            .device = device,
            .command_queue = command_queue,
            .swap_chain = swap_chain3,
            .srv_heap = undefined,
            .rtv_heap = rtv_heap,
            .rtv_desc_size = rtv_desc_size,
            .render_targets = render_targets,
            .command_allocator = command_allocator,
            .command_list = command_list,

            .frame_index = swap_chain3.getCurrentBackBufferIndex(),
            .fence_event = fence_event,
            .fence = fence,
            .fence_value = fence_value,

            .staging_buffers = staging_buffers,

            .root_sig = root_sig,
            .pipeline_state = pipeline_state,

            .font = undefined,
            .font_size = undefined,
            .font_scale = undefined,
            .slot_size = undefined,
            .font_atlas = undefined,
            .font_atlas_desc_offset = undefined,
            .font_atlas_size = undefined,
            .font_atlas_slot_pos = undefined,
            .font_atlas_slot_count_per_dim = undefined,
            .font_atlas_slot_count = undefined,
            .unicode_map = .{},

            .config_buffer = undefined,

            .console = undefined,
            .console_buffer = undefined,
            .console_size = undefined,

            .console_output = undefined,
            .console_output_desc_offset = undefined,
        };

        self.srv_heap = try GPUDescHeap.init(device, .CBV_SRV_UAV, 16, .SHADER_VISIBLE);
        const srv_cpu = self.srv_heap.heap.getCPUDescriptorHandleForHeapStart();

        self.config_buffer = try GPUBuffer.init(
            device,
            std.mem.alignForward(u64, @sizeOf(Config), 256),
            .DEFAULT,
            .COPY_DEST,
        );
        self.config_buffer.desc_offset = try self.srv_heap.alloc();
        self.device.createConstantBufferView(
            &dx12.D3D12_CONSTANT_BUFFER_VIEW_DESC{
                .BufferLocation = self.config_buffer.heap.getGPUVirtualAddress(),
                .SizeInBytes = @intCast(self.config_buffer.capacity),
            },
            srv_cpu.offset(self.config_buffer.desc_offset),
        );

        const staging_buffer = &self.staging_buffers[self.frame_index];

        // load font
        self.font = try Font.initSystemFont(allocator, hwnd, "Consolas");
        errdefer self.font.deinit(allocator);

        self.font_size = 16.0;
        self.font_atlas_size = .{ 256, 128 };

        const font_format = dxgi.common.DXGI_FORMAT.R8_UNORM;
        const font_stride = self.font_atlas_size[0];

        // assuming monospaced font, figure out the render metrics
        self.font_scale = self.font.info.scaleForPixelHeight(self.font_size);
        const default_codepoint = self.font.info.findCodepointIndex(@intCast('A')); // sometime the codepoint 0 isnt available
        const v_metrics = self.font.info.getFontVMetrics();
        const h_metrics = self.font.info.getHMetrics(default_codepoint);
        const ascent: u32 = @intFromFloat(@ceil(@as(f32, @floatFromInt(v_metrics.ascent)) * self.font_scale) + 1.0); // account for rounding error
        self.slot_size[0] = @intFromFloat(@ceil(@as(f32, @floatFromInt(h_metrics.advance_width)) * self.font_scale));
        self.slot_size[1] = @intFromFloat(@ceil(@as(f32, @floatFromInt(v_metrics.ascent - v_metrics.descent)) * self.font_scale) + 1.0);
        self.font_atlas_slot_pos = .{ 1, 0 };
        self.font_atlas_slot_count_per_dim[0] = self.font_atlas_size[0] / self.slot_size[0];
        self.font_atlas_slot_count_per_dim[1] = self.font_atlas_size[1] / self.slot_size[1];
        self.font_atlas_slot_count = self.font_atlas_slot_count_per_dim[0] * self.font_atlas_slot_count_per_dim[1];
        try self.unicode_map.append(allocator, 0xffff_ffff);

        // update console
        // todo: free previous console on resize
        self.console_size = .{ self.screen_size[0] / self.slot_size[0] + 1, self.screen_size[1] / self.slot_size[1] + 1 };
        self.console = try allocator.alloc(Char, self.console_size[0] * self.console_size[1]);
        const console_byte_len: u32 = @intCast(@sizeOf(Char) * self.console.len);
        // todo: console buffer only grows
        self.console_buffer = try GPUBuffer.init(self.device, console_byte_len, .DEFAULT, .COPY_DEST);
        // note: allocation on the srv heap must be only done once
        self.console_buffer.desc_offset = try self.srv_heap.alloc();
        self.device.createShaderResourceView(
            self.console_buffer.heap,
            &dx12.D3D12_SHADER_RESOURCE_VIEW_DESC{
                .Format = .UNKNOWN,
                .ViewDimension = .BUFFER,
                .Shader4ComponentMapping = dx12.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .Anonymous = .{
                    .Buffer = .{
                        .FirstElement = 0,
                        .NumElements = @intCast(self.console.len),
                        .StructureByteStride = @sizeOf(Char),
                        .Flags = .NONE,
                    },
                },
            },
            srv_cpu.offset(self.console_buffer.desc_offset),
        );
        errdefer allocator.free(self.console);
        errdefer self.console_buffer.deinit();

        self.clearConsole();

        // update config buffer
        {
            var config = Config{
                .slot_size = self.slot_size,
                .font_atlas_size = self.font_atlas_slot_count_per_dim,
                .console_size = self.console_size,
                .colors = undefined,
            };
            config.colors[@intFromEnum(ColorEnum.text)] = .{ 1.0, 1.0, 1.0, 1.0 };
            config.colors[@intFromEnum(ColorEnum.bg)] = .{ 39.0 / 255.0, 40.0 / 255.0, 34.0 / 255.0, 1.0 };

            const result = try staging_buffer.alloc(@sizeOf(Config));
            @memcpy(result.cpu_slice, std.mem.asBytes(&config));
            self.command_list.copyBufferRegion(
                self.config_buffer.heap,
                0,
                staging_buffer.buffer.heap,
                result.offset,
                @sizeOf(Config),
            );
        }

        // pre cache font_strip characteres
        const strip_height = (128 / self.font_atlas_slot_count_per_dim[0] + 1) * self.slot_size[1];
        var font_strip: []u8 = &.{};
        {
            font_strip = try allocator.alloc(u8, self.font_atlas_size[0] * strip_height);
            @memset(font_strip, 0);

            try self.unicode_map.ensureTotalCapacity(allocator, 128);
            // self.unicode_map.appendAssumeCapacity(0xffffffff);

            for (0..128) |unicode| {
                const c: u8 = @intCast(unicode);
                if (c == ' ' or std.ascii.isControl(c)) continue;

                self.unicode_map.appendAssumeCapacity(c);

                const i = self.font.info.findCodepointIndex(@intCast(unicode));
                const font_box = self.font.info.getBitmapBox(i, self.font_scale, self.font_scale);

                // fix the yeird font coordinate system
                var x0: u32 = self.font_atlas_slot_pos[0] * self.slot_size[0];
                x0 += @intCast(font_box.x0);
                var y0: u32 = self.font_atlas_slot_pos[1] * self.slot_size[1];
                y0 += ascent;
                if (font_box.y0 < 0) {
                    y0 -= @intCast(-font_box.y0);
                } else {
                    y0 += @intCast(font_box.y0);
                }
                y0 *= font_stride;

                self.font.info.makeBitmap(
                    font_strip.ptr + (x0 + y0),
                    font_box.x1 - font_box.x0,
                    font_box.y1 - font_box.y0,
                    @intCast(font_stride),
                    self.font_scale,
                    self.font_scale,
                    i,
                );

                self.font_atlas_slot_pos[0] += 1;
                if (self.font_atlas_slot_pos[0] == self.font_atlas_slot_count_per_dim[0]) {
                    self.font_atlas_slot_pos[0] = 0;
                    self.font_atlas_slot_pos[1] += 1;
                }
            }
        }
        defer allocator.free(font_strip);

        // create font atlas texture
        try hrErrorOnFail(self.device.createCommittedResource(
            &dx12.D3D12_HEAP_PROPERTIES{},
            .NONE,
            &dx12.D3D12_RESOURCE_DESC{
                .Dimension = .TEXTURE2D,
                .Alignment = 0,
                .Width = @intCast(self.font_atlas_size[0]),
                .Height = @intCast(self.font_atlas_size[1]),
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = font_format,
                .SampleDesc = .{ .Count = 1, .Quality = 0 },
                .Layout = .UNKNOWN,
                .Flags = .NONE,
            },
            .COPY_DEST,
            null,
            dx12.IID_ID3D12Resource,
            @ptrCast(&self.font_atlas),
        ));
        errdefer _ = self.font_atlas.release();

        const result = try staging_buffer.alloc(font_strip.len);
        @memcpy(result.cpu_slice, font_strip);

        command_list.copyTextureRegion(
            &dx12.D3D12_TEXTURE_COPY_LOCATION{
                .pResource = self.font_atlas,
                .Type = .SUBRESOURCE_INDEX,
                .Anonymous = .{ .SubresourceIndex = 0 },
            },
            0,
            0,
            0,
            &dx12.D3D12_TEXTURE_COPY_LOCATION{
                .pResource = staging_buffer.buffer.heap,
                .Type = .PLACED_FOOTPRINT,
                .Anonymous = .{
                    .PlacedFootprint = dx12.D3D12_PLACED_SUBRESOURCE_FOOTPRINT{
                        .Offset = result.offset,
                        .Footprint = dx12.D3D12_SUBRESOURCE_FOOTPRINT{
                            .Format = font_format,
                            .Width = @intCast(self.font_atlas_size[0]),
                            .Height = @intCast(strip_height),
                            .Depth = 1,
                            .RowPitch = @intCast(font_stride),
                        },
                    },
                },
            },
            null,
        );

        command_list.resourceBarrier(
            1,
            @ptrCast(&dx12.D3D12_RESOURCE_BARRIER{
                .Type = .TRANSITION,
                .Flags = .NONE,
                .Anonymous = .{
                    .Transition = dx12.D3D12_RESOURCE_TRANSITION_BARRIER{
                        .pResource = self.font_atlas,
                        .Subresource = 0,
                        .StateBefore = .COPY_DEST,
                        .StateAfter = .PIXEL_SHADER_RESOURCE,
                    },
                },
            }),
        );

        self.font_atlas_desc_offset = try self.srv_heap.alloc();
        self.device.createShaderResourceView(
            self.font_atlas,
            &dx12.D3D12_SHADER_RESOURCE_VIEW_DESC{
                .Shader4ComponentMapping = dx12.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .Format = font_format,
                .ViewDimension = .TEXTURE2D,
                .Anonymous = .{
                    .Texture2D = .{
                        .MostDetailedMip = 0,
                        .MipLevels = 1,
                        .PlaneSlice = 0,
                        .ResourceMinLODClamp = 0.0,
                    },
                },
            },
            srv_cpu.offset(self.font_atlas_desc_offset),
        );

        // create compute output texture
        try hrErrorOnFail(self.device.createCommittedResource(
            &dx12.D3D12_HEAP_PROPERTIES{},
            .NONE,
            &dx12.D3D12_RESOURCE_DESC{
                .Dimension = .TEXTURE2D,
                .Alignment = 0,
                .Width = @intCast(self.console_size[0] * self.slot_size[0]),
                .Height = @intCast(self.console_size[1] * self.slot_size[1]),
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = .R8G8B8A8_UNORM,
                .SampleDesc = .{ .Count = 1, .Quality = 0 },
                .Layout = .UNKNOWN,
                .Flags = .ALLOW_UNORDERED_ACCESS,
            },
            .UNORDERED_ACCESS,
            null,
            dx12.IID_ID3D12Resource,
            @ptrCast(&self.console_output),
        ));
        errdefer _ = self.console_output.release();

        self.console_output_desc_offset = try self.srv_heap.alloc();
        self.device.createUnorderedAccessView(
            self.console_output,
            null,
            &dx12.D3D12_UNORDERED_ACCESS_VIEW_DESC{
                .Format = .R8G8B8A8_UNORM,
                .ViewDimension = .TEXTURE2D,
                .Anonymous = .{
                    .Texture2D = .{
                        .MipSlice = 0,
                        .PlaneSlice = 0,
                    },
                },
            },
            srv_cpu.offset(self.console_output_desc_offset),
        );

        if (TextFile.open(allocator, "src/main.zig")) |file| {
            try self.files.append(allocator, file);
            self.file_index = 0;
            self.changed.file_view = true;
        } else |_| {
            // do nothing
        }

        // upload resources
        {
            try hrErrorOnFail(self.command_list.close());

            // execute the command list.
            const command_lists = [_]*dx12.ID3D12GraphicsCommandList{self.command_list};
            self.command_queue.executeCommandLists(@intCast(command_lists.len), @constCast(@ptrCast(&command_lists)));
        }

        try self.waitForPreviousFrame();

        return self;
    }

    fn deinit(self: *App) void {
        self.waitForPreviousFrame() catch {};

        _ = self.console_output.release();

        for (self.files.items) |*file| {
            file.flush() catch {};
            file.deinit(self.allocator);
        }
        self.files.deinit(self.allocator);

        self.allocator.free(self.console);
        self.console_buffer.deinit();

        self.unicode_map.deinit(self.allocator);
        _ = self.font_atlas.release();
        self.font.deinit(self.allocator);

        _ = self.pipeline_state.release();
        _ = self.root_sig.release();

        self.config_buffer.deinit();

        self.srv_heap.deinit();

        for (0..frame_count) |i| {
            self.staging_buffers[i].deinit();
        }

        _ = self.fence.release();
        _ = wf.CloseHandle(self.fence_event);

        _ = self.command_allocator.release();
        _ = self.command_list.release();

        for (0..frame_count) |i| {
            _ = self.render_targets[i].release();
        }

        _ = self.rtv_heap.release();
        _ = self.swap_chain.release();
        _ = self.command_queue.release();
        _ = self.device.release();
    }

    fn clearConsole(self: *App) void {
        @memset(self.console, Char{});
        self.changed.console = true;
    }

    fn viewFile(self: *App, file_index: u32) void {
        if (self.files.items.len <= file_index) {
            self.clearConsole();
            return;
        }

        const file = &self.files.items[file_index];
        self.file_index = file_index;

        var i: usize = 0;
        for (0..self.console_size[1]) |y| {
            for (0..self.console_size[0]) |x| {
                var char = Char{ .index = 0 };

                const pos = [2]usize{
                    x + file.scroll_pos[0],
                    y + file.scroll_pos[1],
                };

                if (pos[1] < file.lines.len) {
                    const line = file.lines.items(.data)[pos[1]];
                    if (pos[0] < line.items.len) {
                        // todo: read unicode char
                        const unicode: u32 = line.items[pos[0]];
                        if (std.mem.indexOfScalar(u32, self.unicode_map.items, unicode)) |index| {
                            char.index = @intCast(index);
                        }
                    }
                }

                self.console[i] = char;
                i += 1;
            }
        }

        // display cursors
        for (file.cursors.items) |cursor| {
            if (cursor.x < file.scroll_pos[0]) continue;
            if (cursor.pos[1] < file.scroll_pos[1]) continue;

            const x = cursor.x - file.scroll_pos[0];
            if (x >= self.console_size[0]) continue;

            const y = cursor.pos[1] - file.scroll_pos[1];
            if (y >= self.console_size[1]) continue;

            i = self.console_size[0] * y + x;
            self.console[i].cursor_line = true;
        }

        self.changed.file_view = false;
        self.changed.console = true;
    }

    fn tick(self: *App) !void {
        try hrErrorOnFail(self.command_allocator.reset());
        try hrErrorOnFail(self.command_list.reset(self.command_allocator, self.pipeline_state));

        const staging_buffer = &self.staging_buffers[self.frame_index];

        if (self.changed.file_view) {
            self.viewFile(self.file_index);
            self.changed.file_view = false;
        }

        if (self.changed.console) {
            const console_bytes = std.mem.sliceAsBytes(self.console);
            const result = try staging_buffer.alloc(console_bytes.len);
            @memcpy(result.cpu_slice, console_bytes);
            self.command_list.copyBufferRegion(
                self.console_buffer.heap,
                0,
                staging_buffer.buffer.heap,
                result.offset,
                @intCast(console_bytes.len),
            );
            self.changed.console = false;
        }

        self.command_list.setComputeRootSignature(self.root_sig);
        const heaps = [_]*dx12.ID3D12DescriptorHeap{self.srv_heap.heap};
        self.command_list.setDescriptorHeaps(@intCast(heaps.len), @constCast(@ptrCast(&heaps)));
        const srv_gpu = self.srv_heap.heap.getGPUDescriptorHandleForHeapStart();
        self.command_list.setComputeRootDescriptorTable(0, srv_gpu.offset(self.config_buffer.desc_offset));
        self.command_list.setComputeRootDescriptorTable(1, srv_gpu.offset(self.console_buffer.desc_offset));
        self.command_list.setComputeRootDescriptorTable(2, srv_gpu.offset(self.font_atlas_desc_offset));
        self.command_list.setComputeRootDescriptorTable(3, srv_gpu.offset(self.console_output_desc_offset));
        self.command_list.dispatch((self.console_size[0] / 16) + 1, (self.console_size[1] / 16) + 1, 1);

        self.command_list.resourceBarrier(
            1,
            @ptrCast(&dx12.D3D12_RESOURCE_BARRIER{
                .Type = .TRANSITION,
                .Flags = .NONE,
                .Anonymous = .{ .Transition = .{
                    .pResource = self.console_output,
                    .StateBefore = .UNORDERED_ACCESS,
                    .StateAfter = .COPY_SOURCE,
                    .Subresource = 0xffffffff,
                } },
            }),
        );

        const render_target = self.render_targets[self.frame_index];

        self.command_list.resourceBarrier(
            1,
            @ptrCast(&dx12.D3D12_RESOURCE_BARRIER{
                .Type = .TRANSITION,
                .Flags = .NONE,
                .Anonymous = .{ .Transition = .{
                    .pResource = render_target,
                    .StateBefore = .COMMON,
                    .StateAfter = .COPY_DEST,
                    .Subresource = 0xffffffff,
                } },
            }),
        );

        self.command_list.copyTextureRegion(
            &dx12.D3D12_TEXTURE_COPY_LOCATION{
                .pResource = render_target,
                .Type = .SUBRESOURCE_INDEX,
                .Anonymous = .{ .SubresourceIndex = 0 },
            },
            0,
            0,
            0,
            &dx12.D3D12_TEXTURE_COPY_LOCATION{
                .pResource = self.console_output,
                .Type = .SUBRESOURCE_INDEX,
                .Anonymous = .{ .SubresourceIndex = 0 },
            },
            // todo: soft scroll
            &dx12.D3D12_BOX{
                .left = 0,
                .top = 0,
                .right = self.screen_size[0],
                .bottom = self.screen_size[1],
                .front = 0,
                .back = 1,
            },
        );

        self.command_list.resourceBarrier(
            1,
            @ptrCast(&dx12.D3D12_RESOURCE_BARRIER{
                .Type = .TRANSITION,
                .Flags = .NONE,
                .Anonymous = .{ .Transition = .{
                    .pResource = render_target,
                    .StateBefore = .COPY_DEST,
                    .StateAfter = .COMMON,
                    .Subresource = 0xffffffff,
                } },
            }),
        );

        self.command_list.resourceBarrier(
            1,
            @ptrCast(&dx12.D3D12_RESOURCE_BARRIER{
                .Type = .TRANSITION,
                .Flags = .NONE,
                .Anonymous = .{ .Transition = .{
                    .pResource = self.console_output,
                    .StateBefore = .COPY_SOURCE,
                    .StateAfter = .UNORDERED_ACCESS,
                    .Subresource = 0xffffffff,
                } },
            }),
        );

        try hrErrorOnFail(self.command_list.close());

        // execute the command list
        const command_lists = [_]*dx12.ID3D12GraphicsCommandList{self.command_list};
        self.command_queue.executeCommandLists(@intCast(command_lists.len), @constCast(@ptrCast(&command_lists)));

        // clear current staging buffer to be used in the next couple of frames
        staging_buffer.len = 0;

        // present the frame
        try hrErrorOnFail(self.swap_chain.present(1, 0));

        try self.waitForPreviousFrame();
    }

    fn waitForPreviousFrame(self: *App) !void {
        // WAITING FOR THE FRAME TO COMPLETE BEFORE CONTINUING IS NOT BEST PRACTICE.
        // this is code implemented as such for simplicity. The D3D12HelloFrameBuffering
        // sample illustrates how to use fences for efficient resource usage and to
        // maximize GPU utilization.

        // signal and increment the fence value.
        const fence: u64 = self.fence_value;
        try hrErrorOnFail(self.command_queue.signal(self.fence, fence));
        self.fence_value += 1;

        // Wait until the previous frame is finished.
        if (self.fence.getCompletedValue() < fence) {
            try hrErrorOnFail(self.fence.setEventOnCompletion(fence, self.fence_event));
            _ = win.system.threading.WaitForSingleObject(self.fence_event, 0xffffffff);
        }

        self.frame_index = self.swap_chain.getCurrentBackBufferIndex();
    }

    fn keyDown(self: *App, key: kbm.VIRTUAL_KEY) void {
        log.info("down: {s}", .{@tagName(key)});
        if (key == .ESCAPE) {
            _ = wm.DestroyWindow(self.hwnd);
        } else if (key == .PRIOR) {
            self.pageUp();
        } else if (key == .NEXT) {
            self.pageDown();
        } else if (key == .UP) {
            self.moveCursor(.{ 0, -1 });
        } else if (key == .DOWN) {
            self.moveCursor(.{ 0, 1 });
        } else if (key == .LEFT) {
            self.moveCursor(.{ -1, 0 });
        } else if (key == .RIGHT) {
            self.moveCursor(.{ 1, 0 });
        } else if (key == .END) {
            self.putCursor(.end);
        } else if (key == .HOME) {
            self.putCursor(.start);
        }
    }

    fn keyUp(self: *App, key: kbm.VIRTUAL_KEY) void {
        _ = self;
        log.info("up: {s}", .{@tagName(key)});
    }

    fn scrollRelative(self: *App, offset: [2]i32) void {
        if (offset[0] == 0 and offset[1] == 0) return;

        var file = &self.files.items[self.file_index];

        // todo: per-pixel scroll and soft scroll

        const o = file.scroll_pos;

        inline for (0..2) |coord| {
            if (offset[coord] >= 0) {
                file.scroll_pos[coord] +|= @intCast(offset[coord]);
                const limit: u32 = @intCast(file.lines.len - @divTrunc(self.console_size[coord], 2));
                if (file.scroll_pos[coord] > limit) {
                    file.scroll_pos[coord] = limit;
                }
            } else {
                file.scroll_pos[coord] -|= @intCast(-offset[coord]);
            }
        }

        if (file.scroll_pos[0] == o[0] and file.scroll_pos[1] == o[0]) return;

        self.changed.file_view = true;
    }

    fn moveCursor(self: *App, offset: [2]i32) void {
        if (offset[0] == 0 and offset[1] == 0) return;

        var scroll_y: i32 = 0;

        const file = &self.files.items[self.file_index];
        for (0..file.cursors.items.len) |i| {
            const cursor = &file.cursors.items[i];

            // move
            if (offset[1] > 0) {
                cursor.pos[1] +|= @intCast(offset[1]);
            } else if (offset[1] < 0) {
                cursor.pos[1] -|= @intCast(-offset[1]);
            }

            if (offset[0] > 0) {
                cursor.x +|= @as(u32, @intCast(offset[0]));
                cursor.pos[0] = cursor.x;
            } else if (offset[0] < 0) {
                cursor.x -|= @as(u32, @intCast(-offset[0]));
                cursor.pos[0] = cursor.x;
            }

            // validate
            const lines_count: u32 = @intCast(file.lines.len);
            if (cursor.pos[1] > lines_count) {
                cursor.pos[1] = lines_count;
            }

            const line_len: u32 = @intCast(file.lines.items(.data)[cursor.pos[1]].items.len);
            if (cursor.pos[0] > line_len) {
                cursor.x = line_len;
                // always change the desired location on horizontal moviment
                if (offset[0] != 0) cursor.pos[0] = cursor.x;
            } else {
                cursor.x = cursor.pos[0];
            }

            // scroll
            // todo: horizontal scroll
            // if (offset[1] != 0) {
            //     if (cursor.x < file.scroll_pos[0]) {
            //         scroll_offset[0] = @min(scroll_offset[0], -@as(i32, @intCast(cursor.x + 8)));
            //     } else {
            //         const x = cursor.x - file.scroll_pos[0];
            //         if (x >= self.console_size[0]) {
            //             scroll_offset[0] = @max(scroll_offset[0], @as(i32, @intCast(x + 8)));
            //         }
            //     }
            // }

            if (offset[1] != 0) {
                if (cursor.pos[1] < file.scroll_pos[1]) {
                    scroll_y = @min(scroll_y, -@as(i32, @intCast(file.scroll_pos[1] - cursor.pos[1])));
                } else {
                    const y = cursor.pos[1] - file.scroll_pos[1];
                    if (y >= self.console_size[1]) {
                        scroll_y = @max(scroll_y, @as(i32, @intCast(y - self.console_size[1] + 1)));
                    }
                }
            }
        }

        self.scrollRelative(.{ 0, scroll_y });

        // todo: better delta
        self.changed.file_view = true;
    }

    fn putCursor(self: *App, loc: enum(u32) { start = 0, end = std.math.maxInt(u32) }) void {
        const file = &self.files.items[self.file_index];
        for (0..file.cursors.items.len) |i| {
            const cursor = &file.cursors.items[i];
            const x: u32 = @intFromEnum(loc);
            cursor.pos[0] = x;

            const line_len: u32 = @intCast(file.lines.items(.data)[cursor.pos[1]].items.len);
            if (cursor.pos[0] > line_len) {
                cursor.x = line_len;
            } else {
                cursor.x = cursor.pos[0];
            }
        }

        // todo: better delta
        self.changed.file_view = true;

        // todo: horizontal scroll
    }

    fn pageDown(self: *App) void {
        self.moveCursor(.{ 0, @as(i32, @intCast(self.console_size[1] / 2)) });
    }

    fn pageUp(self: *App) void {
        self.moveCursor(.{ 0, -@as(i32, @intCast(self.console_size[1] / 2)) });
    }
};

fn windowProc(
    hwnd: wf.HWND,
    umsg: u32,
    wparam: wf.WPARAM,
    lparam: wf.LPARAM,
) callconv(@import("std").os.windows.WINAPI) wf.LRESULT {
    if (umsg == wm.WM_CREATE) {
        // save the App* passed in to CreateWindow
        const create_struct: *wm.CREATESTRUCT = @ptrFromInt(@as(usize, @bitCast(lparam)));
        _ = wm.SetWindowLongPtr(hwnd, .P_USERDATA, @bitCast(@intFromPtr(create_struct.lpCreateParams)));
        return 0;
    }

    const app: ?*App = @ptrFromInt(@as(usize, @bitCast(wm.GetWindowLongPtr(hwnd, .P_USERDATA))));

    if (umsg == wm.WM_PAINT) {
        // todo: better error handling
        app.?.tick() catch unreachable;
    } else if (umsg == wm.WM_KEYDOWN) {
        app.?.keyDown(@enumFromInt(wparam));
    } else if (umsg == wm.WM_KEYUP) {
        app.?.keyUp(@enumFromInt(wparam));
    } else if (umsg == wm.WM_MOUSEMOVE) {
        // const x: i16 = @truncate((lparam) & 0xffff);
        // const y: i16 = @truncate((lparam >> 16) & 0xffff);
    } else if (umsg == wm.WM_MOUSEWHEEL) {
        var y: i32 = @intCast(@as(i16, @bitCast(@as(u16, @truncate((wparam >> 16) & 0xffff)))));
        y = @divTrunc(y, 120);
        app.?.scrollRelative(.{ 0, -y });
        // } else if (umsg == wm.WM_MOUSEACTIVATE) {
    } else if (umsg == wm.WM_DESTROY) {
        wm.PostQuitMessage(0);
    }

    if (app != null and @as(u32, @bitCast(app.?.changed)) != 0) {
        _ = gdi.InvalidateRect(hwnd, null, 0);
    }

    return wm.DefWindowProc(hwnd, umsg, wparam, lparam);
}

pub fn main() !void {
    const hinstance = win.system.library_loader.GetModuleHandle(null).?;

    const class_name = L("WindowClass");
    const wc = wm.WNDCLASS{
        .style = wm.WNDCLASS_STYLES.initFlags(.{ .VREDRAW = 1, .HREDRAW = 1 }),
        .lpfnWndProc = &windowProc,
        .cbClsExtra = 0,
        .cbWndExtra = 0,
        .hInstance = hinstance,
        .hIcon = null,
        .hCursor = null,
        .hbrBackground = null,
        .lpszMenuName = null,
        .lpszClassName = class_name,
    };
    _ = wm.RegisterClass(&wc);

    var app: App = undefined;

    const w = 1024;
    const h = 512;
    var window_rect = wf.RECT{
        .left = 0,
        .top = 0,
        .right = @intCast(w),
        .bottom = @intCast(h),
    };
    _ = wm.AdjustWindowRect(&window_rect, wm.WS_OVERLAPPEDWINDOW, 0);

    const CW_USEDEFAULT = wm.CW_USEDEFAULT;
    const hwnd = wm.CreateWindowEx(
        .LEFT, // optional window styles.
        class_name,
        L("txed"), // window text
        wm.WS_OVERLAPPEDWINDOW, // window style
        // size and position
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        window_rect.right - window_rect.left,
        window_rect.bottom - window_rect.top,
        null, // parent window
        null, // menu
        hinstance, // instance handle
        &app, // additional application data
    ).?;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer if (gpa.deinit() == .leak) @panic("memory leak");

    app = try App.init(gpa.allocator(), hwnd, .{ w, h });
    defer app.deinit();

    _ = wm.ShowWindow(hwnd, .SHOWDEFAULT);

    // run the message loop.
    var msg: wm.MSG = undefined;
    while (wm.GetMessage(&msg, null, 0, 0) > 0) {
        _ = wm.TranslateMessage(&msg);
        _ = wm.DispatchMessage(&msg);
    }
}
