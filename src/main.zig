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

const Font = struct {
    info: tt.FontInfo,
    data: ?[]u8 = null,

    fn init(data: []const u8) !Font {
        return .{ .info = try tt.FontInfo.init(data, 0) };
    }

    fn initSystemFont(hwnd: wf.HWND, font_name: [:0]const u8) !Font {
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

        const data = try std.heap.page_allocator.alloc(u8, fond_data_size);
        _ = gdi.GetFontData(hdc, 0, 0, @ptrCast(data.ptr), fond_data_size);
        _ = gdi.EndPaint(hwnd, &ps);

        return Font{
            .info = try tt.FontInfo.init(data, 0),
            .data = data,
        };
    }

    fn deinit(self: *Font) void {
        if (self.data) |data| std.heap.page_allocator.free(data);
    }
};

const App = struct {
    const frame_count = 2;

    const Vert = extern struct {
        pos: [3]f32,
        uv: [2]f32,
    };

    hwnd: wf.HWND,

    aspect_ratio: f32,
    viewport: dx12.D3D12_VIEWPORT,
    scissor_rect: wf.RECT,

    device: *dx12.ID3D12Device,
    command_queue: *dx12.ID3D12CommandQueue,
    swap_chain: *dxgi.IDXGISwapChain3,
    srv_heap: *dx12.ID3D12DescriptorHeap,
    rtv_heap: *dx12.ID3D12DescriptorHeap,
    rtv_desc_size: u32,
    render_targets: [frame_count]*dx12.ID3D12Resource,
    command_allocator: *dx12.ID3D12CommandAllocator,
    command_list: *dx12.ID3D12GraphicsCommandList,
    root_sig: *dx12.ID3D12RootSignature,

    frame_index: u32,
    fence_event: wf.HANDLE,
    fence: *dx12.ID3D12Fence,
    fence_value: u64,

    staging_buffers: [frame_count]GPUStagingBuffer,
    vb: GPUBuffer,

    pipeline_state: *dx12.ID3D12PipelineState,
    mesh_vbv: dx12.D3D12_VERTEX_BUFFER_VIEW,

    font: Font,
    font_size: f32,
    font_scale: f32,
    slot_size: [2]u32,
    font_atlas: *dx12.ID3D12Resource,
    font_atlas_size: [2]u32,
    font_atlas_slot_pos: [2]u32,
    font_atlas_slot_count_per_dim: [2]u32,
    font_atlas_slot_count: u32,
    unicode_map: std.ArrayListUnmanaged(u32),

    fn init(hwnd: wf.HWND, size: [2]u32) !App {
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
        var srv_heap: *dx12.ID3D12DescriptorHeap = undefined;
        try hrErrorOnFail(device.createDescriptorHeap(
            &dx12.D3D12_DESCRIPTOR_HEAP_DESC{
                .Type = .CBV_SRV_UAV,
                .NumDescriptors = frame_count,
                .Flags = .SHADER_VISIBLE,
                .NodeMask = 0,
            },
            dx12.IID_ID3D12DescriptorHeap,
            @ptrCast(&srv_heap),
        ));
        errdefer _ = srv_heap.release();
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
            const sampler = dx12.D3D12_STATIC_SAMPLER_DESC{
                .Filter = .MIN_MAG_MIP_POINT,
                .AddressU = .BORDER,
                .AddressV = .BORDER,
                .AddressW = .BORDER,
                .MipLODBias = 0.0,
                .MaxAnisotropy = 0,
                .ComparisonFunc = .NEVER,
                .BorderColor = .TRANSPARENT_BLACK,
                .MinLOD = 0.0,
                .MaxLOD = std.math.floatMax(f32),
                .ShaderRegister = 0,
                .RegisterSpace = 0,
                .ShaderVisibility = .PIXEL,
            };

            const param = dx12.D3D12_ROOT_PARAMETER{
                .ParameterType = .DESCRIPTOR_TABLE,
                .Anonymous = .{
                    .DescriptorTable = dx12.D3D12_ROOT_DESCRIPTOR_TABLE{
                        .NumDescriptorRanges = 1,
                        .pDescriptorRanges = @ptrCast(&dx12.D3D12_DESCRIPTOR_RANGE{
                            .RangeType = .SRV,
                            .NumDescriptors = 1,
                            .BaseShaderRegister = 0,
                            .RegisterSpace = 0,
                            // .Flags = .DATA_STATIC,
                            .OffsetInDescriptorsFromTableStart = 0xffffffff,
                        }),
                    },
                },
                .ShaderVisibility = .PIXEL,
            };

            var signature: *dx.ID3DBlob = undefined;
            var err: *dx.ID3DBlob = undefined;
            errdefer _ = err.release(); // otherwise err is null
            try hrErrorOnFail(dx12.D3D12SerializeRootSignature(
                &dx12.D3D12_ROOT_SIGNATURE_DESC{
                    .NumParameters = 1,
                    .pParameters = @ptrCast(&param),
                    .NumStaticSamplers = 1,
                    .pStaticSamplers = @ptrCast(&sampler),
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
            staging_buffers[i] = try GPUStagingBuffer.init(device, 4 * 1024 * 1024);
            staging_buffers_len += 1;
        }

        // create buffers
        var vb = try GPUBuffer.init(
            device,
            256,
            .DEFAULT,
            .COPY_DEST,
        );
        errdefer vb.deinit();

        var self = App{
            .hwnd = hwnd,

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
            .srv_heap = srv_heap,
            .rtv_heap = rtv_heap,
            .rtv_desc_size = rtv_desc_size,
            .render_targets = render_targets,
            .command_allocator = command_allocator,
            .command_list = command_list,
            .root_sig = root_sig,

            .frame_index = swap_chain3.getCurrentBackBufferIndex(),
            .fence_event = fence_event,
            .fence = fence,
            .fence_value = fence_value,

            .staging_buffers = staging_buffers,
            .vb = vb,

            .pipeline_state = undefined,
            .mesh_vbv = undefined,

            .font = undefined,
            .font_size = undefined,
            .font_scale = undefined,
            .slot_size = undefined,
            .font_atlas = undefined,
            .font_atlas_size = undefined,
            .font_atlas_slot_pos = undefined,
            .font_atlas_slot_count_per_dim = undefined,
            .font_atlas_slot_count = undefined,
            .unicode_map = .{},
        };

        // load shaders
        {
            const vs_cso = @embedFile("shader.vs.cso");
            const ps_cso = @embedFile("shader.ps.cso");

            var pso_desc = std.mem.zeroes(dx12.D3D12_GRAPHICS_PIPELINE_STATE_DESC);
            // define the vertex input layout.
            const inputElementDescs = [_]dx12.D3D12_INPUT_ELEMENT_DESC{
                .{
                    .SemanticName = "POSITION",
                    .SemanticIndex = 0,
                    .Format = .R32G32B32_FLOAT,
                    .InputSlot = 0,
                    .AlignedByteOffset = @offsetOf(Vert, "pos"),
                    .InputSlotClass = .VERTEX_DATA,
                    .InstanceDataStepRate = 0,
                },
                .{
                    .SemanticName = "TEXCOORD",
                    .SemanticIndex = 0,
                    .Format = .R32G32_FLOAT,
                    .InputSlot = 0,
                    .AlignedByteOffset = @offsetOf(Vert, "uv"),
                    .InputSlotClass = .VERTEX_DATA,
                    .InstanceDataStepRate = 0,
                },
            };
            // describe and create the graphics pipeline state object (PSO).
            pso_desc.InputLayout.NumElements = @intCast(inputElementDescs.len);
            pso_desc.InputLayout.pInputElementDescs = @constCast(@ptrCast(&inputElementDescs));
            pso_desc.pRootSignature = self.root_sig;
            pso_desc.VS = .{ .pShaderBytecode = vs_cso, .BytecodeLength = vs_cso.len };
            pso_desc.PS = .{ .pShaderBytecode = ps_cso, .BytecodeLength = ps_cso.len };
            pso_desc.RasterizerState = .{};
            pso_desc.BlendState = .{};
            pso_desc.DepthStencilState.DepthEnable = wz.FALSE;
            pso_desc.DepthStencilState.StencilEnable = wz.FALSE;
            pso_desc.SampleMask = std.math.maxInt(u32);
            pso_desc.PrimitiveTopologyType = .TRIANGLE;
            pso_desc.NumRenderTargets = 1;
            pso_desc.RTVFormats[0] = .R8G8B8A8_UNORM;
            pso_desc.SampleDesc.Count = 1;
            try hrErrorOnFail(self.device.createGraphicsPipelineState(
                &pso_desc,
                dx12.IID_ID3D12PipelineState,
                @ptrCast(&self.pipeline_state),
            ));
        }
        errdefer _ = self.pipeline_state.release();

        const staging_buffer = &self.staging_buffers[self.frame_index];

        // create buffers
        {
            // define the geometry for a triangle.
            const triangle = [_]Vert{
                .{ .pos = .{ -1.0, 1.0, 0.0 }, .uv = .{ 0.0, 0.0 } },
                .{ .pos = .{ 3.0, 1.0, 0.0 }, .uv = .{ 2.0, 0.0 } },
                .{ .pos = .{ -1.0, -3.0, 0.0 }, .uv = .{ 0.0, 2.0 } },
            };
            const triangle_bytes = std.mem.sliceAsBytes(&triangle);

            // copy the triangle data to the vertex buffer.
            const result = try staging_buffer.alloc(@intCast(triangle_bytes.len));
            @memcpy(result.cpu_slice, triangle_bytes);
            self.command_list.copyBufferRegion(
                self.vb.heap,
                0,
                staging_buffer.buffer.heap,
                result.offset,
                @intCast(triangle_bytes.len),
            );

            // initialize the vertex buffer view
            self.mesh_vbv = dx12.D3D12_VERTEX_BUFFER_VIEW{
                .BufferLocation = staging_buffer.buffer.heap.getGPUVirtualAddress() + result.offset,
                .StrideInBytes = @sizeOf(Vert),
                .SizeInBytes = @intCast(triangle_bytes.len),
            };
        }

        // load font
        self.font = try Font.initSystemFont(hwnd, "Consolas");
        //self.font = try Font.init(@embedFile("fonts/intro.otf"));
        errdefer self.font.deinit();

        self.font_size = 18.0;
        self.font_atlas_size = .{ 1024, 512 };

        const font_format = dxgi.common.DXGI_FORMAT.R8_UNORM;
        const font_stride = self.font_atlas_size[0];

        // assuming monospaced font, figure out the render metrics
        self.font_scale = self.font.info.scaleForPixelHeight(self.font_size);
        const default_codepoint = self.font.info.findCodepointIndex(@intCast('A')); // sometime the codepoint 0 isnt available
        const v_metrics = self.font.info.getFontVMetrics();
        const h_metrics = self.font.info.getHMetrics(default_codepoint);
        const ascent: u32 = @intFromFloat(@ceil(@as(f32, @floatFromInt(v_metrics.ascent)) * self.font_scale) + 1.0); // account for rounding error
        self.slot_size[0] = @intFromFloat(@ceil(@as(f32, @floatFromInt(h_metrics.advance_width)) * self.font_scale));
        self.slot_size[1] = @intFromFloat(@ceil(@as(f32, @floatFromInt(v_metrics.ascent - v_metrics.descent)) * self.font_scale));
        self.font_atlas_slot_pos = .{ 0, 0 };
        self.font_atlas_slot_count_per_dim[0] = self.font_atlas_size[0] / self.slot_size[0];
        self.font_atlas_slot_count_per_dim[1] = self.font_atlas_size[1] / self.slot_size[1];
        self.font_atlas_slot_count = self.font_atlas_slot_count_per_dim[0] * self.font_atlas_slot_count_per_dim[1];

        //self.font.info.getFontVMetrics()

        // pre cache font_strip characteres
        const strip_height = (128 / self.font_atlas_slot_count_per_dim[0] + 1) * self.slot_size[1];
        var font_strip: []u8 = &.{};
        {
            font_strip = try std.heap.page_allocator.alloc(u8, self.font_atlas_size[0] * strip_height);
            @memset(font_strip, 0);

            try self.unicode_map.ensureTotalCapacity(std.heap.page_allocator, 128);

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
        defer std.heap.page_allocator.free(font_strip);

        // create font atlas texture
        {
            try hrErrorOnFail(self.device.createCommittedResource(
                &dx12.D3D12_HEAP_PROPERTIES{
                    .Type = .DEFAULT,
                    .CPUPageProperty = .UNKNOWN,
                    .MemoryPoolPreference = .UNKNOWN,
                    .CreationNodeMask = 1,
                    .VisibleNodeMask = 1,
                },
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
                self.srv_heap.getCPUDescriptorHandleForHeapStart(),
            );
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

        self.unicode_map.deinit(std.heap.page_allocator);
        _ = self.font_atlas.release();
        self.font.deinit();

        _ = self.pipeline_state.release();

        self.vb.deinit();
        for (0..frame_count) |i| {
            self.staging_buffers[i].deinit();
        }

        _ = self.fence.release();
        _ = wf.CloseHandle(self.fence_event);

        _ = self.command_allocator.release();
        _ = self.command_list.release();

        _ = self.root_sig.release();

        for (0..frame_count) |i| {
            _ = self.render_targets[i].release();
        }

        _ = self.rtv_heap.release();
        _ = self.srv_heap.release();
        _ = self.swap_chain.release();
        _ = self.command_queue.release();
        _ = self.device.release();
    }

    fn tick(self: *App) !void {
        try hrErrorOnFail(self.command_allocator.reset());
        try hrErrorOnFail(self.command_list.reset(self.command_allocator, self.pipeline_state));

        // set necessary state.
        self.command_list.setGraphicsRootSignature(self.root_sig);
        self.command_list.setDescriptorHeaps(1, @ptrCast(&self.srv_heap));
        self.command_list.setGraphicsRootDescriptorTable(0, self.srv_heap.getGPUDescriptorHandleForHeapStart());
        self.command_list.rsSetViewports(1, @ptrCast(&self.viewport));
        self.command_list.rsSetScissorRects(1, @ptrCast(&self.scissor_rect));

        // indicate that the back buffer will be used as a render target.
        self.command_list.resourceBarrier(
            1,
            @ptrCast(&dx12.D3D12_RESOURCE_BARRIER{
                .Type = .TRANSITION,
                .Flags = .NONE,
                .Anonymous = .{ .Transition = .{
                    .pResource = self.render_targets[self.frame_index],
                    .StateBefore = .COMMON,
                    .StateAfter = .RENDER_TARGET,
                    .Subresource = 0xffffffff,
                } },
            }),
        );

        var rtv_handle = self.rtv_heap.getCPUDescriptorHandleForHeapStart();
        rtv_handle.ptr += self.frame_index * self.rtv_desc_size;
        self.command_list.omSetRenderTargets(1, &rtv_handle, wz.FALSE, null);

        // record commands.
        const clearColor = [4]f32{ 0.0, 0.2, 0.4, 1.0 };
        self.command_list.clearRenderTargetView(rtv_handle, @ptrCast(&clearColor), 0, null);
        self.command_list.iaSetPrimitiveTopology(.PT_TRIANGLELIST);
        self.command_list.iaSetVertexBuffers(0, 1, @ptrCast(&self.mesh_vbv));
        self.command_list.drawInstanced(3, 1, 0, 0);

        // indicate that the back buffer will now be used to present.
        self.command_list.resourceBarrier(
            1,
            @ptrCast(&dx12.D3D12_RESOURCE_BARRIER{
                .Type = .TRANSITION,
                .Flags = .NONE,
                .Anonymous = .{ .Transition = .{
                    .pResource = self.render_targets[self.frame_index],
                    .StateBefore = .RENDER_TARGET,
                    .StateAfter = .COMMON,
                    .Subresource = 0xffffffff,
                } },
            }),
        );

        try hrErrorOnFail(self.command_list.close());

        // execute the command list.
        const command_lists = [_]*dx12.ID3D12GraphicsCommandList{self.command_list};
        self.command_queue.executeCommandLists(@intCast(command_lists.len), @constCast(@ptrCast(&command_lists)));

        // Present the frame.
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
        }
    }

    fn keyUp(self: *App, key: kbm.VIRTUAL_KEY) void {
        _ = self;
        log.info("up: {s}", .{@tagName(key)});
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
        // var ps: gdi.PAINTSTRUCT = undefined;
        // const hdc = gdi.BeginPaint(hwnd, &ps);
        // // all painting occurs here, between BeginPaint and EndPaint.
        // _ = gdi.FillRect(hdc, &ps.rcPaint, @as(gdi.HBRUSH, @ptrFromInt(@intFromEnum(wm.COLOR_WINDOWFRAME))));
        // _ = gdi.EndPaint(hwnd, &ps);
        app.?.tick() catch {
            // todo: better error handling
            _ = wm.DestroyWindow(hwnd);
        };
        return 0;
    } else if (umsg == wm.WM_KEYDOWN) {
        app.?.keyDown(@enumFromInt(wparam));
        return 0;
    } else if (umsg == wm.WM_KEYUP) {
        app.?.keyUp(@enumFromInt(wparam));
        return 0;
    } else if (umsg == wm.WM_MOUSEMOVE) {
        return 0;
    } else if (umsg == wm.WM_MOUSEWHEEL) {
        return 0;
    } else if (umsg == wm.WM_MOUSEACTIVATE) {
        return 0;
    } else if (umsg == wm.WM_DESTROY) {
        wm.PostQuitMessage(0);
        return 0;
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

    app = try App.init(hwnd, .{ w, h });
    defer app.deinit();

    _ = wm.ShowWindow(hwnd, .SHOWDEFAULT);

    // run the message loop.
    var msg: wm.MSG = undefined;
    while (wm.GetMessage(&msg, null, 0, 0) > 0) {
        _ = wm.TranslateMessage(&msg);
        _ = wm.DispatchMessage(&msg);
    }
}
