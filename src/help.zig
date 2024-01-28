const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const win = @import("win32.zig");
const wf = win.foundation;
const dx12 = win.graphics.direct3d12;

pub fn hrErr(hr: wf.HRESULT) void {
    std.log.err(
        "HRESULT 0x{x}",
        .{@as(c_ulong, @bitCast(hr))},
    );
}

pub inline fn hrErrorOnFail(hr: wf.HRESULT) !void {
    if (hr < 0) {
        hrErr(hr);
        return error.HResultError;
    }
}

pub inline fn hrPanicOnFail(hr: wf.HRESULT) void {
    if (hr < 0) unreachable;
}

pub const GPUBuffer = struct {
    heap: *dx12.ID3D12Resource,
    capacity: u64,
    desc_offset: u64,

    pub fn init(
        device: *dx12.ID3D12Device,
        capacity: u64,
        heap_type: dx12.D3D12_HEAP_TYPE,
        states: dx12.D3D12_RESOURCE_STATES,
    ) !GPUBuffer {
        var resource: *dx12.ID3D12Resource = undefined;
        try hrErrorOnFail(device.createCommittedResource(
            &dx12.D3D12_HEAP_PROPERTIES{
                .Type = heap_type,
                .CPUPageProperty = .UNKNOWN,
                .MemoryPoolPreference = .UNKNOWN,
                .CreationNodeMask = 1,
                .VisibleNodeMask = 1,
            },
            .NONE,
            &dx12.D3D12_RESOURCE_DESC{
                .Dimension = .BUFFER,
                .Alignment = 0,
                .Width = capacity,
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
            },
            states,
            null,
            dx12.IID_ID3D12Resource,
            @ptrCast(&resource),
        ));
        errdefer _ = resource.release();

        return GPUBuffer{
            .heap = resource,
            .capacity = capacity,
            .desc_offset = undefined,
        };
    }

    pub fn deinit(self: *GPUBuffer) void {
        assert(self.heap.release() == 0);
    }
};

pub const GPUStagingBuffer = struct {
    // todo: where this aligment comes from?
    pub const alloc_alignment: u32 = 512;

    buffer: GPUBuffer,
    cpu_slice: []u8,
    gpu_addr: u64,
    len: u64,

    pub const AllocResult = struct {
        cpu_slice: []u8,
        offset: u64,
    };

    pub fn init(device: *dx12.ID3D12Device, capacity: u64) !GPUStagingBuffer {
        var buffer = try GPUBuffer.init(
            device,
            capacity,
            .UPLOAD,
            .GENERIC_READ,
        );
        errdefer _ = buffer.deinit();

        var cpu_base: [*]u8 = undefined;
        try hrErrorOnFail(buffer.heap.map(
            0,
            &dx12.D3D12_RANGE{ .Begin = 0, .End = 0 },
            @ptrCast(&cpu_base),
        ));

        return GPUStagingBuffer{
            .buffer = buffer,
            .cpu_slice = cpu_base[0..capacity],
            .gpu_addr = buffer.heap.getGPUVirtualAddress(),
            .len = 0,
        };
    }

    pub fn deinit(self: *GPUStagingBuffer) void {
        self.buffer.deinit();
    }

    pub fn alloc(self: *GPUStagingBuffer, size: u64) Allocator.Error!AllocResult {
        const aligned_size = (size + (alloc_alignment - 1)) & ~(alloc_alignment - 1);
        if ((self.len + aligned_size) > self.buffer.capacity) {
            return Allocator.Error.OutOfMemory;
        }

        const result = AllocResult{
            .cpu_slice = (self.cpu_slice.ptr + self.len)[0..size],
            .offset = self.len,
        };

        self.len += aligned_size;

        return result;
    }
};

pub const GPUDescHeap = struct {
    heap: *dx12.ID3D12DescriptorHeap,
    offset: u64,
    size: u32,
    capacity: u32,

    pub fn init(
        device: *dx12.ID3D12Device,
        type_: dx12.D3D12_DESCRIPTOR_HEAP_TYPE,
        capacity: u32,
        flags: dx12.D3D12_DESCRIPTOR_HEAP_FLAGS,
    ) !@This() {
        var heap: *dx12.ID3D12DescriptorHeap = undefined;
        try hrErrorOnFail(device.createDescriptorHeap(
            &dx12.D3D12_DESCRIPTOR_HEAP_DESC{
                .Type = type_,
                .NumDescriptors = capacity,
                .Flags = flags,
                .NodeMask = 0,
            },
            dx12.IID_ID3D12DescriptorHeap,
            @ptrCast(&heap),
        ));
        const size = device.getDescriptorHandleIncrementSize(type_);
        return @This(){
            .heap = heap,
            .offset = 0,
            .size = size,
            .capacity = capacity * size,
        };
    }

    pub fn deinit(self: *@This()) void {
        _ = self.heap.release();
    }

    pub fn alloc(self: *@This()) Allocator.Error!u64 {
        const offset = self.offset;
        if (offset >= self.capacity) {
            return Allocator.Error.OutOfMemory;
        }
        self.offset += self.size;
        return offset;
    }
};
