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

const L = win.zig.L;
const SUCCEEDED = win.zig.SUCCEEDED;
const FAILED = win.zig.FAILED;
const FALSE = win.zig.FALSE;

// const I32x2 = @Vector(2, i32);
// const U32x2 = @Vector(2, u32);

fn panicOnFail(hresult: wf.HRESULT) void {
    if (hresult < 0) unreachable;
}

const App = struct {
    const frame_count = 2;

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

    fn init(hwnd: wf.HWND, size: [2]u32) !App {
        var dxgi_factory_flags: u32 = 0;

        if (builtin.mode == .Debug) {
            var debug_ctrl: *dx12.ID3D12Debug = undefined;
            if (SUCCEEDED(dx12.D3D12GetDebugInterface(dx12.IID_ID3D12Debug, @ptrCast(&debug_ctrl)))) {
                debug_ctrl.enableDebugLayer();

                // enable additional debug layers.
                dxgi_factory_flags |= dxgi.DXGI_CREATE_FACTORY_DEBUG;
                _ = debug_ctrl.release();
            }
        }

        var factory: *dxgi.IDXGIFactory4 = undefined;
        if (FAILED(dxgi.CreateDXGIFactory2(
            dxgi_factory_flags,
            dxgi.IID_IDXGIFactory4,
            @ptrCast(&factory),
        ))) {
            return error.CreateDXGIFactoryFailed;
        }
        defer _ = factory.release();

        var hardwareAdapter: *dxgi.IDXGIAdapter1 = undefined;
        getHardwareAdapter: {
            var adapterIndex: u32 = undefined;
            var adapter: *dxgi.IDXGIAdapter1 = undefined;

            var factory6: *dxgi.IDXGIFactory6 = undefined;
            if (SUCCEEDED(factory.queryInterface(dxgi.IID_IDXGIFactory6, @ptrCast(&factory6)))) {
                defer _ = factory6.release();

                adapterIndex = 0;
                while (SUCCEEDED(factory6.enumAdapterByGpuPreference(
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
                        if (SUCCEEDED(dx12.D3D12CreateDevice(@ptrCast(adapter), dx.D3D_FEATURE_LEVEL_11_0, dx12.IID_ID3D12Device, null))) {
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
            while (SUCCEEDED(factory.enumAdapters1(adapterIndex, @ptrCast(&adapter)))) : (adapterIndex += 1) {
                var desc: dxgi.DXGI_ADAPTER_DESC1 = undefined;
                _ = adapter.getDesc1(&desc);

                if ((desc.Flags & @as(u32, @intFromEnum(dxgi.DXGI_ADAPTER_FLAG_SOFTWARE))) != 0) {
                    if (SUCCEEDED(dx12.D3D12CreateDevice(@ptrCast(adapter), dx.D3D_FEATURE_LEVEL_11_0, dx12.IID_ID3D12Device, null))) {
                        hardwareAdapter = adapter;
                        break :getHardwareAdapter;
                    }
                }

                _ = adapter.release();
            }
        }
        defer _ = hardwareAdapter.release();

        var device: *dx12.ID3D12Device = undefined;
        if (FAILED(dx12.D3D12CreateDevice(
            @ptrCast(hardwareAdapter),
            dx.D3D_FEATURE_LEVEL_11_0,
            dx12.IID_ID3D12Device,
            @ptrCast(&device),
        ))) {
            return error.CreateDeviceFailed;
        }
        errdefer _ = device.release();

        const queue_desc = dx12.D3D12_COMMAND_QUEUE_DESC{
            .Type = .DIRECT,
            .Priority = 0,
            .Flags = .NONE,
            .NodeMask = 0,
        };
        var command_queue: *dx12.ID3D12CommandQueue = undefined;
        if (FAILED(device.createCommandQueue(
            &queue_desc,
            dx12.IID_ID3D12CommandQueue,
            @ptrCast(&command_queue),
        ))) {
            return error.CreateCommandQueueFailed;
        }
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
        if (FAILED(factory.createSwapChainForHwnd(
            @ptrCast(command_queue), // Swap chain needs the queue so that it can force a flush on it.
            hwnd,
            &swap_chain_desc,
            null,
            null,
            @ptrCast(&swap_chain),
        ))) {
            return error.CreateSwapChainFailed;
        }
        errdefer _ = swap_chain.release();

        var swap_chain3: *dxgi.IDXGISwapChain3 = undefined;
        if (FAILED(swap_chain.queryInterface(dxgi.IID_IDXGISwapChain3, @ptrCast(&swap_chain3)))) {
            return error.SwapChainVersionMissmatch;
        }

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
        if (FAILED(device.createDescriptorHeap(
            &rtv_heap_desc,
            dx12.IID_ID3D12DescriptorHeap,
            @ptrCast(&rtv_heap),
        ))) {
            return error.CreateDescriptorHeapFailed;
        }
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
            if (FAILED(swap_chain.getBuffer(
                @intCast(n),
                dx12.IID_ID3D12Resource,
                @ptrCast(&render_targets[n]),
            ))) {
                return error.SwapChainGetBufferFailed;
            }
            device.createRenderTargetView(@ptrCast(render_targets[n]), null, rtv_handle);
            rtv_handle.ptr += rtv_desc_size;
            render_targets_len += 1;
        }

        var command_allocator: *dx12.ID3D12CommandAllocator = undefined;
        if (FAILED(device.createCommandAllocator(
            .DIRECT,
            dx12.IID_ID3D12CommandAllocator,
            @ptrCast(&command_allocator),
        ))) {
            return error.CreateCommandAllocatorFailed;
        }
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
            if (FAILED(dx12.D3D12SerializeRootSignature(
                &root_signature_desc,
                .v1_0,
                @ptrCast(&signature),
                @ptrCast(&err),
            ))) {
                // otherwise err is null
                _ = err.release();
                return error.SerializeRootSignatureFailed;
            }
            defer _ = signature.release();
            if (FAILED(device.createRootSignature(
                0,
                @ptrCast(signature.getBufferPointer()),
                signature.getBufferSize(),
                dx12.IID_ID3D12RootSignature,
                @ptrCast(&root_sig),
            ))) {
                return error.CreateRootSignatureFailed;
            }
        }
        errdefer _ = root_sig.release();

        // todo: load shaders and buffers here

        // create the command list.
        var command_list: *dx12.ID3D12GraphicsCommandList = undefined;
        if (FAILED(device.createCommandList(
            0,
            .DIRECT,
            command_allocator,
            null,
            dx12.IID_ID3D12GraphicsCommandList,
            @ptrCast(&command_list),
        ))) {
            return error.CreateCommandListFailed;
        }
        errdefer _ = command_list.release();

        // command lists are created in the recording state, but there is nothing
        // to record yet. The main loop expects it to be closed, so close it now.
        _ = command_list.close();

        // create synchronization objects and wait until assets have been uploaded to the GPU.
        var fence: *dx12.ID3D12Fence = undefined;
        if (FAILED(device.createFence(0, .NONE, dx12.IID_ID3D12Fence, @ptrCast(&fence)))) {
            return error.CreateFenceFailed;
        }
        errdefer _ = fence.release();

        const fence_value: u64 = 1;

        // create an event handle to use for frame synchronization.
        const fence_event = win.system.threading.CreateEventA(null, FALSE, FALSE, null) orelse {
            //if (wf.GetLastError() != .NO_ERROR) {}
            return error.CreateEventFailed;
        };
        errdefer _ = wf.CloseHandle(fence_event);

        var self = App{
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
        };

        self.waitForPreviousFrame();

        return self;
    }

    fn deinit(self: *App) void {
        self.waitForPreviousFrame();

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

    fn tick(self: *App) void {
        panicOnFail(self.command_allocator.reset());
        panicOnFail(self.command_list.reset(self.command_allocator, null));

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
        self.command_list.omSetRenderTargets(1, &rtv_handle, FALSE, null);

        // record commands.
        const clearColor = [4]f32{ 0.0, 0.2, 0.4, 1.0 };
        self.command_list.clearRenderTargetView(rtv_handle, @ptrCast(&clearColor), 0, null);

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

        panicOnFail(self.command_list.close());

        // execute the command list.
        const command_lists = [_]*dx12.ID3D12GraphicsCommandList{self.command_list};
        self.command_queue.executeCommandLists(@intCast(command_lists.len), @constCast(@ptrCast(&command_lists)));

        // Present the frame.
        panicOnFail(self.swap_chain.present(1, 0));

        self.waitForPreviousFrame();
    }

    fn waitForPreviousFrame(self: *App) void {
        // WAITING FOR THE FRAME TO COMPLETE BEFORE CONTINUING IS NOT BEST PRACTICE.
        // this is code implemented as such for simplicity. The D3D12HelloFrameBuffering
        // sample illustrates how to use fences for efficient resource usage and to
        // maximize GPU utilization.

        // signal and increment the fence value.
        const fence: u64 = self.fence_value;
        panicOnFail(self.command_queue.signal(self.fence, fence));
        self.fence_value += 1;

        // Wait until the previous frame is finished.
        if (self.fence.getCompletedValue() < fence) {
            panicOnFail(self.fence.setEventOnCompletion(fence, self.fence_event));
            _ = win.system.threading.WaitForSingleObject(self.fence_event, 0xffffffff);
        }

        self.frame_index = self.swap_chain.getCurrentBackBufferIndex();
    }

    fn keyDown(self: *App, key: kbm.VIRTUAL_KEY) void {
        _ = self;
        log.info("down: {s}", .{@tagName(key)});
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
        app.?.tick();
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