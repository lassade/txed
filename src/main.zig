const std = @import("std");
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

// const I32x2 = @Vector(2, i32);
// const U32x2 = @Vector(2, u32);

fn hrErr(hr: wf.HRESULT) void {
    std.log.err(
        "HRESULT 0x{x}",
        .{@as(c_ulong, @bitCast(hr))},
    );
}

inline fn hrErrorOnFail(hr: wf.HRESULT) !void {
    if (hr < 0) {
        hrErr(hr);
        return error.HResultError;
    }
}

inline fn hrPanicOnFail(hr: wf.HRESULT) void {
    if (hr < 0) unreachable;
}

const App = struct {
    const frame_count = 2;

    const Vert = extern struct {
        pos: [3]f32,
        color: [4]f32,
    };

    hwnd: wf.HWND,

    aspect_ratio: f32,
    viewport: dx12.D3D12_VIEWPORT,
    scissor_rect: wf.RECT,

    device: *dx12.ID3D12Device,
    command_queue: *dx12.ID3D12CommandQueue,
    swap_chain: *dxgi.IDXGISwapChain3,
    frame: u32,
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

    pipeline_state: *dx12.ID3D12PipelineState,
    vertex_buffer: *dx12.ID3D12Resource,
    vertex_buffer_view: dx12.D3D12_VERTEX_BUFFER_VIEW,

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
                    _ = adapter.getDesc1(&desc);

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

        const swap_chain_desc = dxgi.DXGI_SWAP_CHAIN_DESC1{
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
        };
        var swap_chain: *dxgi.IDXGISwapChain1 = undefined;
        try hrErrorOnFail(factory.createSwapChainForHwnd(
            @ptrCast(command_queue), // Swap chain needs the queue so that it can force a flush on it.
            hwnd,
            &swap_chain_desc,
            null,
            null,
            @ptrCast(&swap_chain),
        ));
        errdefer _ = swap_chain.release();

        var swap_chain3: *dxgi.IDXGISwapChain3 = undefined;
        try hrErrorOnFail(swap_chain.queryInterface(dxgi.IID_IDXGISwapChain3, @ptrCast(&swap_chain3)));

        // this sample does not support fullscreen transitions.
        _ = factory.makeWindowAssociation(hwnd, dxgi.DXGI_MWA_NO_ALT_ENTER);

        // create descriptor heaps.
        // describe and create a render target view (RTV) descriptor heap.
        var rtv_heap_desc = dx12.D3D12_DESCRIPTOR_HEAP_DESC{
            .Type = .RTV,
            .NumDescriptors = frame_count,
            .Flags = .NONE,
            .NodeMask = 0,
        };
        var rtv_heap: *dx12.ID3D12DescriptorHeap = undefined;
        try hrErrorOnFail(device.createDescriptorHeap(
            &rtv_heap_desc,
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
            try hrErrorOnFail(swap_chain.getBuffer(
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

        // load assets

        // create an empty root signature.
        var root_sig: *dx12.ID3D12RootSignature = undefined;
        {
            var root_signature_desc = dx12.D3D12_ROOT_SIGNATURE_DESC{
                .NumParameters = 0,
                .pParameters = null,
                .NumStaticSamplers = 0,
                .pStaticSamplers = null,
                .Flags = .ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            };

            var signature: *dx.ID3DBlob = undefined;
            var err: *dx.ID3DBlob = undefined;
            errdefer _ = err.release(); // otherwise err is null
            try hrErrorOnFail(dx12.D3D12SerializeRootSignature(
                &root_signature_desc,
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

        // command lists are created in the recording state, but there is nothing
        // to record yet. The main loop expects it to be closed, so close it now.
        _ = command_list.close();

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
            .frame = swap_chain3.getCurrentBackBufferIndex(),
            .rtv_heap = rtv_heap,
            .rtv_desc_size = rtv_desc_size,
            .render_targets = render_targets,
            .command_allocator = command_allocator,
            .command_list = command_list,
            .root_sig = root_sig,

            .frame_index = 0,
            .fence_event = fence_event,
            .fence = fence,
            .fence_value = fence_value,

            .pipeline_state = undefined,
            .vertex_buffer = undefined,
            .vertex_buffer_view = undefined,
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
                    .SemanticName = "COLOR",
                    .SemanticIndex = 0,
                    .Format = .R32G32B32A32_FLOAT,
                    .InputSlot = 0,
                    .AlignedByteOffset = @offsetOf(Vert, "color"),
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

        // create buffers
        {
            // define the geometry for a triangle.
            const triangle = [_]Vert{
                .{ .pos = .{ 0.0, 0.25 * self.aspect_ratio, 0.0 }, .color = .{ 1.0, 0.0, 0.0, 1.0 } },
                .{ .pos = .{ 0.25, -0.25 * self.aspect_ratio, 0.0 }, .color = .{ 0.0, 1.0, 0.0, 1.0 } },
                .{ .pos = .{ -0.25, -0.25 * self.aspect_ratio, 0.0 }, .color = .{ 0.0, 0.0, 1.0, 1.0 } },
            };

            // Note: using upload heaps to transfer static data like vert buffers is not
            // recommended. Every time the GPU needs it, the upload heap will be marshalled
            // over. Please read up on Default Heap usage. An upload heap is used here for
            // code simplicity and because there are very few verts to actually transfer.,
            const heap_props = dx12.D3D12_HEAP_PROPERTIES{
                .Type = .UPLOAD,
                .CPUPageProperty = .UNKNOWN,
                .MemoryPoolPreference = .UNKNOWN,
                .CreationNodeMask = 1,
                .VisibleNodeMask = 1,
            };
            const vertex_buffer_desc = dx12.D3D12_RESOURCE_DESC{
                .Dimension = .BUFFER,
                .Alignment = 0,
                .Width = @intCast(triangle.len),
                .Height = 1,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = .UNKNOWN,
                .SampleDesc = .{
                    .Count = 1,
                    .Quality = 0,
                },
                .Layout = .ROW_MAJOR,
                .Flags = .NONE,
            };
            try hrErrorOnFail(device.createCommittedResource(
                &heap_props,
                .NONE,
                &vertex_buffer_desc,
                .GENERIC_READ,
                null,
                dx12.IID_ID3D12Resource,
                @ptrCast(&self.vertex_buffer),
            ));
            errdefer _ = self.vertex_buffer.release();

            // copy the triangle data to the vertex buffer.
            var vertex_data: [*]u8 = undefined;
            const read_range = dx12.D3D12_RANGE{ .Begin = 0, .End = 0 }; // we do not intend to read from this resource on the CPU.
            try hrErrorOnFail(self.vertex_buffer.map(0, &read_range, @ptrCast(&vertex_data)));
            const triangle_bytes = std.mem.sliceAsBytes(&triangle);
            @memcpy(vertex_data, triangle_bytes);
            self.vertex_buffer.unmap(0, null);

            // initialize the vertex buffer view.
            self.vertex_buffer_view.BufferLocation = self.vertex_buffer.getGPUVirtualAddress();
            self.vertex_buffer_view.StrideInBytes = @sizeOf(Vert);
            self.vertex_buffer_view.SizeInBytes = @intCast(triangle_bytes.len);
        }

        try self.waitForPreviousFrame();

        return self;
    }

    fn deinit(self: *App) void {
        self.waitForPreviousFrame() catch {};

        _ = self.vertex_buffer.release();
        _ = self.pipeline_state.release();

        _ = self.fence.release();
        _ = wf.CloseHandle(self.fence_event);

        _ = self.command_allocator.release();
        _ = self.command_list.release();

        _ = self.root_sig.release();

        for (self.render_targets) |rt| {
            _ = rt.release();
        }

        _ = self.rtv_heap.release();
        _ = self.swap_chain.release();
        _ = self.command_queue.release();
        _ = self.device.release();
    }

    fn tick(self: *App) !void {
        try hrErrorOnFail(self.command_allocator.reset());
        try hrErrorOnFail(self.command_list.reset(self.command_allocator, self.pipeline_state));

        // set necessary state.
        self.command_list.setGraphicsRootSignature(self.root_sig);
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
        self.command_list.iaSetVertexBuffers(0, 1, @ptrCast(&self.vertex_buffer_view));
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

    const w = 1280;
    const h = 720;
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
