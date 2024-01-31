const std = @import("std");
// const builtin = @import("builtin");
const math = std.math;
const assert = std.debug.assert;

// const endianess = builtin.target.cpu.arch.endian();

pub const pi = math.pi;
pub const deg_to_rad = math.pi / 180.0;
pub const rad_to_deg = 180.0 / math.pi;

pub const f32_min: f32 = std.math.floatMin(f32);
pub const f32_max: f32 = std.math.floatMax(f32);

// fundamental types
pub const F32x2 = @Vector(2, f32);
pub const F32x3 = @Vector(3, f32);
pub const F32x4 = @Vector(4, f32);
pub const U32x2 = @Vector(2, u32);
pub const U32x4 = @Vector(4, u32);
pub const I32x2 = @Vector(2, i32);

pub const lerp = std.math.lerp;

pub const Mat2 = extern struct {
    rows: [2]F32x2 = .{
        .{ 1.0, 0.0 },
        .{ 0.0, 1.0 },
    },

    pub const ZERO = Mat2{
        .{ 0.0, 0.0 },
        .{ 0.0, 0.0 },
    };

    // pub fn mul(self: Mat2, r: Mat2) Mat2
    // pub fn mulScalar(self: Mat2, r: f32) Mat2
    // pub fn mulVector(self: Mat2, r: F32x2) F32x2
    // pub fn inverse(self: Mat2) ?Mat2
    // pub fn det(self: Mat2) f32

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !Mat2 {
        return @as(Mat2, @bitCast(try std.json.innerParse([2]F32x4, allocator, source, options)));
    }
};

pub const Mat3 = extern struct {
    rows: [3]F32x4 = .{
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 1.0, 0.0 },
    },

    pub const ZERO = Mat3{
        .rows = .{
            .{ 0.0, 0.0, 0.0, 0.0 },
            .{ 0.0, 0.0, 0.0, 0.0 },
            .{ 0.0, 0.0, 0.0, 0.0 },
        },
    };

    /// Mat3 * F32x3
    pub fn mulVector(self: Mat3, v: F32x4) F32x4 {
        const vx = @shuffle(f32, v, f32x4s(0.0), [4]i32{ 0, 0, 0, ~@as(i32, 0) });
        const vy = @shuffle(f32, v, f32x4s(0.0), [4]i32{ 1, 1, 1, ~@as(i32, 0) });
        const vz = @shuffle(f32, v, f32x4s(0.0), [4]i32{ 2, 2, 2, ~@as(i32, 0) });
        return (self.rows[0] * vx) + (self.rows[1] * vy) + (self.rows[2] * vz);
    }
};

pub const Mat4 = extern struct {
    rows: [4]F32x4 = .{
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    },

    pub const ZERO = Mat4{
        .rows = .{
            .{ 0.0, 0.0, 0.0, 0.0 },
            .{ 0.0, 0.0, 0.0, 0.0 },
            .{ 0.0, 0.0, 0.0, 0.0 },
            .{ 0.0, 0.0, 0.0, 0.0 },
        },
    };

    pub inline fn pos(self: Mat4) [3]f32 {
        return .{ self.rows[3][0], self.rows[3][1], self.rows[3][2] };
    }

    pub fn mul(self: Mat4, r: Mat4) Mat4 {
        var result: Mat4 = undefined;
        comptime var row: u32 = 0;
        inline while (row < 4) : (row += 1) {
            const vx = swizzle(self.rows[row], .x, .x, .x, .x);
            const vy = swizzle(self.rows[row], .y, .y, .y, .y);
            const vz = swizzle(self.rows[row], .z, .z, .z, .z);
            const vw = swizzle(self.rows[row], .w, .w, .w, .w);
            result.rows[row] = @mulAdd(F32x4, vx, r.rows[0], vz * r.rows[2]) + @mulAdd(F32x4, vy, r.rows[1], vw * r.rows[3]);
        }
        return result;
    }

    // pub fn mulScalar(self: Mat4, r: f32) Mat4

    /// Mat4 * F32x4
    pub fn mulVector(self: Mat4, v: F32x4) F32x4 {
        const vx = swizzle(v, .x, .x, .x, .x);
        const vy = swizzle(v, .y, .y, .y, .y);
        const vz = swizzle(v, .z, .z, .z, .z);
        const vw = swizzle(v, .w, .w, .w, .w);
        return (self.rows[0] * vx) + (self.rows[1] * vy) + (self.rows[2] * vz) + (self.rows[3] * vw);
    }

    pub fn transpose(m: Mat4) Mat4 {
        const tmp1 = @shuffle(f32, m.rows[0], m.rows[1], [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1) });
        const tmp3 = @shuffle(f32, m.rows[0], m.rows[1], [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3) });
        const tmp2 = @shuffle(f32, m.rows[2], m.rows[3], [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1) });
        const tmp4 = @shuffle(f32, m.rows[2], m.rows[3], [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3) });
        return Mat4{
            .rows = .{
                @shuffle(f32, tmp1, tmp2, [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) }),
                @shuffle(f32, tmp1, tmp2, [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) }),
                @shuffle(f32, tmp3, tmp4, [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) }),
                @shuffle(f32, tmp3, tmp4, [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) }),
            },
        };
    }

    /// Creates a column-major affine matrix to be used inside the shaders
    pub fn affine(m: Mat4) [12]f32 {
        const tmp1 = @shuffle(f32, m.rows[0], m.rows[1], [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1) });
        const tmp3 = @shuffle(f32, m.rows[0], m.rows[1], [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3) });
        const tmp2 = @shuffle(f32, m.rows[2], m.rows[3], [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1) });
        const tmp4 = @shuffle(f32, m.rows[2], m.rows[3], [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3) });
        return @bitCast([3]F32x4{
            @shuffle(f32, tmp1, tmp2, [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) }),
            @shuffle(f32, tmp1, tmp2, [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) }),
            @shuffle(f32, tmp3, tmp4, [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) }),
            // simplified transpose ignore the last row
        });

        // // row-major impl
        // var output: [12]f32 = undefined;
        // comptime var i: usize = 0;
        // inline for (0..4) |j| {
        //     inline for (0..3) |k| {
        //         output[i] = m.rows[j][k];
        //         i += 1;
        //     }
        // }
        // return output;
    }

    pub fn inverse(m: Mat4) struct { mat: Mat4, det: f32 } {
        const mt = transpose(m).rows;
        var v0: [4]F32x4 = undefined;
        var v1: [4]F32x4 = undefined;

        v0[0] = swizzle(mt[2], .x, .x, .y, .y);
        v1[0] = swizzle(mt[3], .z, .w, .z, .w);
        v0[1] = swizzle(mt[0], .x, .x, .y, .y);
        v1[1] = swizzle(mt[1], .z, .w, .z, .w);
        v0[2] = @shuffle(f32, mt[2], mt[0], [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) });
        v1[2] = @shuffle(f32, mt[3], mt[1], [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) });

        var d0 = v0[0] * v1[0];
        var d1 = v0[1] * v1[1];
        var d2 = v0[2] * v1[2];

        v0[0] = swizzle(mt[2], .z, .w, .z, .w);
        v1[0] = swizzle(mt[3], .x, .x, .y, .y);
        v0[1] = swizzle(mt[0], .z, .w, .z, .w);
        v1[1] = swizzle(mt[1], .x, .x, .y, .y);
        v0[2] = @shuffle(f32, mt[2], mt[0], [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) });
        v1[2] = @shuffle(f32, mt[3], mt[1], [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) });

        d0 = @mulAdd(F32x4, -v0[0], v1[0], d0);
        d1 = @mulAdd(F32x4, -v0[1], v1[1], d1);
        d2 = @mulAdd(F32x4, -v0[2], v1[2], d2);

        v0[0] = swizzle(mt[1], .y, .z, .x, .y);
        v1[0] = @shuffle(f32, d0, d2, [4]i32{ ~@as(i32, 1), 1, 3, 0 });
        v0[1] = swizzle(mt[0], .z, .x, .y, .x);
        v1[1] = @shuffle(f32, d0, d2, [4]i32{ 3, ~@as(i32, 1), 1, 2 });
        v0[2] = swizzle(mt[3], .y, .z, .x, .y);
        v1[2] = @shuffle(f32, d1, d2, [4]i32{ ~@as(i32, 3), 1, 3, 0 });
        v0[3] = swizzle(mt[2], .z, .x, .y, .x);
        v1[3] = @shuffle(f32, d1, d2, [4]i32{ 3, ~@as(i32, 3), 1, 2 });

        var c0 = v0[0] * v1[0];
        var c2 = v0[1] * v1[1];
        var c4 = v0[2] * v1[2];
        var c6 = v0[3] * v1[3];

        v0[0] = swizzle(mt[1], .z, .w, .y, .z);
        v1[0] = @shuffle(f32, d0, d2, [4]i32{ 3, 0, 1, ~@as(i32, 0) });
        v0[1] = swizzle(mt[0], .w, .z, .w, .y);
        v1[1] = @shuffle(f32, d0, d2, [4]i32{ 2, 1, ~@as(i32, 0), 0 });
        v0[2] = swizzle(mt[3], .z, .w, .y, .z);
        v1[2] = @shuffle(f32, d1, d2, [4]i32{ 3, 0, 1, ~@as(i32, 2) });
        v0[3] = swizzle(mt[2], .w, .z, .w, .y);
        v1[3] = @shuffle(f32, d1, d2, [4]i32{ 2, 1, ~@as(i32, 2), 0 });

        c0 = @mulAdd(F32x4, -v0[0], v1[0], c0);
        c2 = @mulAdd(F32x4, -v0[1], v1[1], c2);
        c4 = @mulAdd(F32x4, -v0[2], v1[2], c4);
        c6 = @mulAdd(F32x4, -v0[3], v1[3], c6);

        v0[0] = swizzle(mt[1], .w, .x, .w, .x);
        v1[0] = @shuffle(f32, d0, d2, [4]i32{ 2, ~@as(i32, 1), ~@as(i32, 0), 2 });
        v0[1] = swizzle(mt[0], .y, .w, .x, .z);
        v1[1] = @shuffle(f32, d0, d2, [4]i32{ ~@as(i32, 1), 0, 3, ~@as(i32, 0) });
        v0[2] = swizzle(mt[3], .w, .x, .w, .x);
        v1[2] = @shuffle(f32, d1, d2, [4]i32{ 2, ~@as(i32, 3), ~@as(i32, 2), 2 });
        v0[3] = swizzle(mt[2], .y, .w, .x, .z);
        v1[3] = @shuffle(f32, d1, d2, [4]i32{ ~@as(i32, 3), 0, 3, ~@as(i32, 2) });

        const c1 = @mulAdd(F32x4, -v0[0], v1[0], c0);
        const c3 = @mulAdd(F32x4, v0[1], v1[1], c2);
        const c5 = @mulAdd(F32x4, -v0[2], v1[2], c4);
        const c7 = @mulAdd(F32x4, v0[3], v1[3], c6);

        c0 = @mulAdd(F32x4, v0[0], v1[0], c0);
        c2 = @mulAdd(F32x4, -v0[1], v1[1], c2);
        c4 = @mulAdd(F32x4, v0[2], v1[2], c4);
        c6 = @mulAdd(F32x4, -v0[3], v1[3], c6);

        var mr = Mat4{
            .rows = .{
                f32x4(c0[0], c1[1], c0[2], c1[3]),
                f32x4(c2[0], c3[1], c2[2], c3[3]),
                f32x4(c4[0], c5[1], c4[2], c5[3]),
                f32x4(c6[0], c7[1], c6[2], c7[3]),
            },
        };

        const det = dotSplat(mr.rows[0], mt[0]);
        if (math.approxEqAbs(f32, det[0], 0.0, math.floatEps(f32))) {
            return .{ .mat = Mat4.ZERO, .det = det[0] };
        }

        const scale = splat(F32x4, 1.0) / det;
        mr.rows[0] *= scale;
        mr.rows[1] *= scale;
        mr.rows[2] *= scale;
        mr.rows[3] *= scale;
        return .{ .mat = mr, .det = det[0] };
    }

    // pub fn det(self: Mat4) f32

    /// x is right, y is up, z goes into the screen, clip space is [-1, 1]
    pub fn orthographicLh(ortho_size: f32, aspect: f32, near: f32, far: f32) Mat4 {
        const w = ortho_size * aspect;

        assert(!math.approxEqAbs(f32, w, 0.0, 0.001));
        assert(!math.approxEqAbs(f32, ortho_size, 0.0, 0.001));
        assert(!math.approxEqAbs(f32, far, near, 0.001));

        const a = 1 / w;
        const b = -1 / ortho_size; // z is up
        const c = 1 / (far - near);
        //const offset = bitMask(pos, .{ on, on, on, off }) * f32x4(a, b, c, 0.0);
        return Mat4{
            .rows = .{
                // zig fmt: off
                f32x4(  a, 0.0,        0.0, 0.0),
                f32x4(0.0,   b,        0.0, 0.0),
                f32x4(0.0, 0.0,          c, 0.0),
                f32x4(0.0, 0.0, -c * near , 1.0),// - offset,
                // zig fmt: on
            },
        };
    }

    pub fn perspectiveLh(fovy: f32, aspect: f32, near: f32, far: f32) Mat4 {
        assert(near > 0.0 and far > 0.0);
        assert(!math.approxEqAbs(f32, far, near, 0.001));
        assert(!math.approxEqAbs(f32, aspect, 0.0, 0.01));

        const scfov = sincos(0.5 * fovy);
        assert(!math.approxEqAbs(f32, scfov[0], 0.0, 0.001));

        const h = scfov[1] / scfov[0];
        const w = h / aspect;
        const r = far / (far - near);
        //const offset = bitMask(pos, .{ on, on, on, off }) * f32x4(w, -h, r, 0.0) + swizzle(pos, .zero, .zero, .zero, .z);
        return Mat4{
            .rows = .{
                // zig fmt: off
                f32x4(  w, 0.0,       0.0, 0.0),
                f32x4(0.0,  -h,       0.0, 0.0),
                f32x4(0.0, 0.0,         r, 1.0),
                f32x4(0.0, 0.0, -r * near, 0.0),// - offset,
                // zig fmt: on
            },
        };
    }

    // // todo: untested
    // pub fn frustum(self: Mat4) [6]Plane {
    //     var planes: [6]Plane = undefined;
    //     // zig fmt: off
    //     inline for (0..4) |i| planes[0].xyzd[i] = self.rows[i][3] + self.rows[i][0];
    //     inline for (0..4) |i| planes[1].xyzd[i] = self.rows[i][3] - self.rows[i][0];
    //     inline for (0..4) |i| planes[2].xyzd[i] = self.rows[i][3] + self.rows[i][1];
    //     inline for (0..4) |i| planes[3].xyzd[i] = self.rows[i][3] - self.rows[i][1];
    //     inline for (0..4) |i| planes[4].xyzd[i] = self.rows[i][3] + self.rows[i][2];
    //     inline for (0..4) |i| planes[5].xyzd[i] = self.rows[i][3] - self.rows[i][2];
    //     // zig fmt: on

    //     // normalize planes
    //     inline for (0..6) |i| planes[i] = planes[i].normalize();

    //     return planes;
    // }

    pub inline fn translate(self: Mat4, p: F32x4) Mat4 {
        return Mat4{
            .rows = .{
                self.rows[0],
                self.rows[1],
                self.rows[2],
                self.mulVector(swizzle(p, .x, .y, .z, .one)),
            },
        };
    }

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !Mat4 {
        return @as(Mat4, @bitCast(try std.json.innerParse([4]F32x4, allocator, source, options)));
    }
};

pub fn frustumCorners(pos: F32x4, fovy: f32, aspect: f32, near: f32, far: f32) [8]F32x4 {
    const dy = @tan(0.5 * fovy * deg_to_rad);
    const dx = dy * aspect;
    const t = f32x4(-dx, -dy, dx, dy);

    const p_r = swizzle(pos, .x, .y, .x, .y);
    const near_r = t * f32x4s(near) + p_r;
    const far_r = t * f32x4s(far) + p_r;
    const z = f32x4(near, far, 0, 0) + swizzle(pos, .z, .z, .zero, .zero);

    return .{
        // zig fmt: off
        shuffle( far_r, z, .x0, .y0, .y1, .w1),
        shuffle( far_r, z, .x0, .w0, .y1, .w1),
        shuffle( far_r, z, .z0, .w0, .y1, .w1),
        shuffle( far_r, z, .z0, .y0, .y1, .w1),
        shuffle(near_r, z, .x0, .y0, .x1, .w1),
        shuffle(near_r, z, .x0, .w0, .x1, .w1),
        shuffle(near_r, z, .z0, .w0, .x1, .w1),
        shuffle(near_r, z, .z0, .y0, .x1, .w1),
        // zig fmt: on
    };
}

pub const TigthOrthographicProjectionResult = struct {
    pos: F32x4,
    ortho_size: f32,
    far: f32,
};

// https://learn.microsoft.com/pt-br/windows/win32/dxtecharts/common-techniques-to-improve-shadow-depth-maps?redirectedfrom=MSDN
pub fn tigthOrthographicProjection(frustum_corners: [8]F32x4, axises: [3]F32x4, step: f32) TigthOrthographicProjectionResult {
    var min: F32x4 = f32x4s(f32_max);
    var max: F32x4 = f32x4s(f32_min);
    for (0..8) |i| {
        const z = dot(frustum_corners[i], axises[2]);
        const v = frustum_corners[i] - (f32x4s(z) * axises[2]);
        const x = dot(v, axises[0]);
        const y = dot(v, axises[1]);
        const u = f32x4(x, y, z, 0);
        min = @min(min, u);
        max = @max(max, u);
    }

    // fix shimmering edge effect byt moving in steps
    const step_v = f32x4s(step);
    min = @floor((min / step_v) - f32x4s(1)) * step_v;
    max = @ceil((max / step_v) + f32x4s(1)) * step_v;

    const mid = @floor((max + min) * f32x4s(0.5 / step)) * step_v;
    const size = @abs(max - min);
    const pos = f32x4s(mid[0]) * axises[0] + f32x4s(mid[1]) * axises[1] + f32x4s(min[2]) * axises[2];

    return TigthOrthographicProjectionResult {
        .pos = pos,
        .ortho_size = @max(size[0], size[1]) * 0.5,
        .far = size[2],
    };
}

pub const Quat = extern struct {
    xyzw: F32x4 = .{ 0.0, 0.0, 0.0, 1.0 },

    pub const ZERO = Quat{ .xyzw = .{ 0.0, 0.0, 0.0, 0.0 } };

    // todo: lerp slerp

    // todo: pub fn fromAxisAngle(v: F32x4, angle: f32) Quat {}

    /// pitch, yaw, roll
    pub fn fromEulerXYZ(x: f32, y: f32, z: f32) Quat {
        return fromEuler(f32x4(x, y, z, 0));
    }

    /// (pitch, yaw, roll, 0 )
    pub fn fromEuler(angles: F32x4) Quat {
        const sc = sincos(f32x4s(0.5) * angles);
        const p0 = @shuffle(f32, sc[1], sc[0], [4]i32{ ~@as(i32, 0), 0, 0, 0 });
        const p1 = @shuffle(f32, sc[0], sc[1], [4]i32{ ~@as(i32, 0), 0, 0, 0 });
        const y0 = @shuffle(f32, sc[1], sc[0], [4]i32{ 1, ~@as(i32, 1), 1, 1 });
        const y1 = @shuffle(f32, sc[0], sc[1], [4]i32{ 1, ~@as(i32, 1), 1, 1 });
        const r0 = @shuffle(f32, sc[1], sc[0], [4]i32{ 2, 2, ~@as(i32, 2), 2 });
        const r1 = @shuffle(f32, sc[0], sc[1], [4]i32{ 2, 2, ~@as(i32, 2), 2 });
        const q1 = p1 * f32x4(1.0, -1.0, -1.0, 1.0) * y1;
        const q0 = p0 * y0 * r0;
        return .{ .xyzw = @mulAdd(F32x4, q1, r1, q0) };
    }

    // Algorithm from: https://github.com/g-truc/glm/blob/master/glm/detail/type_quat.inl
    pub fn mulVector(self: Quat, v: F32x4) F32x4 {
        const w = f32x4s(self.xyzw[3]);
        const axis = f32x4(self.xyzw[0], self.xyzw[1], self.xyzw[2], 0.0);
        const uv = cross(axis, v);
        return v + ((uv * w) + cross(axis, uv)) * f32x4s(2.0);
    }

    pub inline fn conjugate(self: Quat) Quat {
        return .{ .xyzw = self.xyzw * f32x4(-1.0, -1.0, -1.0, 1.0) };
    }

    pub fn inverse(self: Quat) Quat {
        const l = f32x4s(lenghtSq(self.xyzw));
        const conj = self.conjugate();
        return .{ .xyzw = @select(f32, l <= f32x4s(math.floatEps(f32)), f32x4s(0), conj.xyzw / l) };
    }

    pub fn mat(self: Quat) Mat4 {
        const q0 = self.xyzw + self.xyzw;
        var q1 = self.xyzw * q0;

        var v0 = swizzle(q1, .y, .x, .x, .w);
        v0 = bitMask(v0, .on, .on, .on, .off);

        var v1 = swizzle(q1, .z, .z, .y, .w);
        v1 = bitMask(v1, .on, .on, .on, .off);

        const r0 = (f32x4(1.0, 1.0, 1.0, 0.0) - v0) - v1;

        v0 = swizzle(self.xyzw, .x, .x, .y, .w);
        v1 = swizzle(q0, .z, .y, .z, .w);
        v0 = v0 * v1;

        v1 = swizzle(self.xyzw, .w, .w, .w, .w);
        const v2 = swizzle(q0, .y, .z, .x, .w);
        v1 = v1 * v2;

        const r1 = v0 + v1;
        const r2 = v0 - v1;

        v0 = @shuffle(f32, r1, r2, [4]i32{ 1, 2, ~@as(i32, 0), ~@as(i32, 1) });
        v0 = swizzle(v0, .x, .z, .w, .y);
        v1 = @shuffle(f32, r1, r2, [4]i32{ 0, 0, ~@as(i32, 2), ~@as(i32, 2) });
        v1 = swizzle(v1, .x, .z, .x, .z);

        q1 = @shuffle(f32, r0, v0, [4]i32{ 0, 3, ~@as(i32, 0), ~@as(i32, 1) });
        q1 = swizzle(q1, .x, .z, .w, .y);

        var m: Mat4 = undefined;
        m.rows[0] = q1;

        q1 = @shuffle(f32, r0, v0, [4]i32{ 1, 3, ~@as(i32, 2), ~@as(i32, 3) });
        q1 = swizzle(q1, .z, .x, .w, .y);
        m.rows[1] = q1;

        q1 = @shuffle(f32, v1, r0, [4]i32{ 0, 1, ~@as(i32, 2), ~@as(i32, 3) });
        m.rows[2] = q1;
        m.rows[3] = f32x4(0, 0, 0, 1);
        return m;
    }

    pub fn smoothDamp(s0: Quat, s1: Quat, vel_ptr: *F32x4, damping: f32, max_speed: f32, dt: f32) Quat {
        // const SMOL: f32 = 0.00000001;
        // if (dt < SMOL) {
        //     return s0;
        // }

        // account for double-cover
        const k0 = s1.xyzw * splat(F32x4, Scalar.copysign(1.0, dot(s0.xyzw, s1.xyzw)));

        // smooth damp (nlerp approx)
        const s = normalize(Vector(4).smoothDamp(s0.xyzw, k0, vel_ptr, damping, max_speed, dt));

        // ensure deriv is tangent
        vel_ptr.* -= s * splat(F32x4, (dot(s, vel_ptr.*) / dot(s, s)));

        return .{ .xyzw = s };
    }

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !Quat {
        return @as(Quat, @bitCast(try std.json.innerParse(F32x4, allocator, source, options)));
    }
};

// initialization functions
pub inline fn f32x2(e0: f32, e1: f32) F32x2 {
    return .{ e0, e1 };
}

pub inline fn f32x3(e0: f32, e1: f32, e2: f32) F32x3 {
    return .{ e0, e1, e2 };
}

pub inline fn f32x4(e0: f32, e1: f32, e2: f32, e3: f32) F32x4 {
    return .{ e0, e1, e2, e3 };
}

pub inline fn i32x2(e0: i32, e1: i32) I32x2 {
    return .{ e0, e1 };
}

pub inline fn u32x4(e0: u32, e1: u32, e2: u32, e3: u32) F32x4 {
    return .{ e0, e1, e2, e3 };
}

pub inline fn quat(e0: f32, e1: f32, e2: f32, e3: f32) Quat {
    return Quat{ .xyzw = .{ e0, e1, e2, e3 } };
}

pub inline fn f32x2s(e0: f32) F32x2 {
    return @splat(e0);
}

pub inline fn f32x3s(e0: f32) F32x3 {
    return @splat(e0);
}

pub inline fn f32x4s(e0: f32) F32x4 {
    return @splat(e0);
}

pub inline fn i32x2s(e0: i32) I32x2 {
    return @splat(e0);
}

pub inline fn splat(comptime T: type, v: f32) T {
    return if (T == f32) v else @as(T, @splat(v));
}

pub inline fn clamp(v: anytype, min: anytype, max: anytype) @TypeOf(v, min, max) {
    return @min(max, @max(min, v));
}

pub inline fn dotSplat(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    if (vecLenght(@TypeOf(v0, v1)) == 4) {
        // fast path for sse2
        var xmm0 = v0 * v1; // | x0*x1 | y0*y1 | z0*z1 | w0*w1 |
        var xmm1 = swizzle(xmm0, .y, .x, .w, .x); // | y0*y1 | -- | w0*w1 | -- |
        xmm1 = xmm0 + xmm1; // | x0*x1 + y0*y1 | -- | z0*z1 + w0*w1 | -- |
        xmm0 = swizzle(xmm1, .z, .x, .x, .x); // | z0*z1 + w0*w1 | -- | -- | -- |
        xmm0 = F32x4{ xmm0[0] + xmm1[0], xmm0[1], xmm0[2], xmm0[2] }; // addss
        return swizzle(xmm0, .x, .x, .x, .x);
    } else {
        return @splat(@reduce(.Add, v0 * v1));
    }
}

pub inline fn dot(v0: anytype, v1: anytype) f32 {
    return dotSplat(v0, v1)[0];
}

pub inline fn lenghtSq(v: anytype) f32 {
    return dot(v, v);
}

pub inline fn lenght(v: anytype) f32 {
    return @sqrt(lenghtSq(v));
}

pub inline fn normalize(v: anytype) @TypeOf(v) {
    return v * @as(@TypeOf(v), @splat(1.0 / lenght(v)));
}

pub inline fn cross(v0: F32x4, v1: F32x4) F32x4 {
    var xmm0 = swizzle(v0, .y, .z, .x, .zero);
    var xmm1 = swizzle(v1, .z, .x, .y, .zero);
    const result = xmm0 * xmm1;
    xmm0 = swizzle(xmm0, .y, .z, .x, .zero);
    xmm1 = swizzle(xmm1, .z, .x, .y, .zero);
    return result - xmm0 * xmm1;
}

pub inline fn recip(v: anytype) @TypeOf(v) {
    return @as(@TypeOf(v), @splat(1.0)) / v;
}

pub fn sincos(v: anytype) [2]@TypeOf(v) {
    const T = @TypeOf(v);

    var x = modAngle(v);
    var sign = bitAnd(x, negativeZero(T));
    const c = bitOr(sign, splat(T, math.pi));
    const absx = bitNotAnd(sign, x);
    const rflx = c - x;
    const comp = absx <= splat(T, 0.5 * math.pi);
    x = select(comp, x, rflx);
    sign = select(comp, splat(T, 1.0), splat(T, -1.0));
    const x2 = x * x;

    var sresult = @mulAdd(T, splat(T, -2.3889859e-08), x2, splat(T, 2.7525562e-06));
    sresult = @mulAdd(T, sresult, x2, splat(T, -0.00019840874));
    sresult = @mulAdd(T, sresult, x2, splat(T, 0.0083333310));
    sresult = @mulAdd(T, sresult, x2, splat(T, -0.16666667));
    sresult = x * @mulAdd(T, sresult, x2, splat(T, 1.0));

    var cresult = @mulAdd(T, splat(T, -2.6051615e-07), x2, splat(T, 2.4760495e-05));
    cresult = @mulAdd(T, cresult, x2, splat(T, -0.0013888378));
    cresult = @mulAdd(T, cresult, x2, splat(T, 0.041666638));
    cresult = @mulAdd(T, cresult, x2, splat(T, -0.5));
    cresult = sign * @mulAdd(T, cresult, x2, splat(T, 1.0));

    return .{ sresult, cresult };
}

pub inline fn select(mask: anytype, v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    if (@TypeOf(mask) == bool) {
        return if (mask) v0 else v1;
    } else {
        return @select(f32, mask, v0, v1);
    }
}

pub const SwizzleMask = enum(i32) {
    x = 0,
    y = 1,
    z = 2,
    w = 3,
    zero = ~@as(i32, 0),
    one = ~@as(i32, 1),
};

pub inline fn swizzle(
    v: F32x4,
    comptime x: SwizzleMask,
    comptime y: SwizzleMask,
    comptime z: SwizzleMask,
    comptime w: SwizzleMask,
) F32x4 {
    return @shuffle(f32, v, f32x4(0.0, 1.0, 0.0, 0.0), [4]i32{ @intFromEnum(x), @intFromEnum(y), @intFromEnum(z), @intFromEnum(w) });
}

pub const ShuffleXYZW = enum(i32) {
    x0 = 0,
    y0 = 1,
    z0 = 2,
    w0 = 3,
    x1 = ~@as(i32, 0),
    y1 = ~@as(i32, 1),
    z1 = ~@as(i32, 2),
    w1 = ~@as(i32, 3),
};

pub inline fn shuffle(
    v0: F32x4,
    v1: F32x4,
    comptime x: ShuffleXYZW,
    comptime y: ShuffleXYZW,
    comptime z: ShuffleXYZW,
    comptime w: ShuffleXYZW,
) F32x4 {
    return @shuffle(f32, v0, v1, [4]i32{ @intFromEnum(x), @intFromEnum(y), @intFromEnum(z), @intFromEnum(w) });
}

pub const Mask = enum(u32) {
    on = ~@as(u32, 0),
    off = 0,
};

pub inline fn bitMask(
    v: F32x4,
    comptime x: Mask,
    comptime y: Mask,
    comptime z: Mask,
    comptime w: Mask,
) F32x4 {
    // todo: generates a vblendps witch is the same as the swizzle
    const mask = comptime U32x4{ @intFromEnum(x), @intFromEnum(y), @intFromEnum(z), @intFromEnum(w) };
    return @bitCast(@as(U32x4, @bitCast(v)) & mask); // andps
}

pub inline fn bitAnd(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    const T = @TypeOf(v0, v1);
    const Tu = if (T == f32) u32 else @Vector(vecLenght(T), u32);
    const v0u = @as(Tu, @bitCast(v0));
    const v1u = @as(Tu, @bitCast(v1));
    return @as(T, @bitCast(v0u & v1u)); // andps
}

pub inline fn bitOr(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    const T = @TypeOf(v0, v1);
    const Tu = if (T == f32) u32 else @Vector(vecLenght(T), u32);
    const v0u = @as(Tu, @bitCast(v0));
    const v1u = @as(Tu, @bitCast(v1));
    return @as(T, @bitCast(v0u | v1u)); // orps
}

// ~a & b
pub inline fn bitNotAnd(a: anytype, b: anytype) @TypeOf(a, b) {
    const T = @TypeOf(a, b);
    const Tu = if (T == f32) u32 else @Vector(vecLenght(T), u32);
    const au = @as(Tu, @bitCast(a));
    const bu = @as(Tu, @bitCast(b));
    return @as(T, @bitCast(~au & bu)); // andnps
}

pub inline fn xy(v: anytype) @Vector(2, childType(@TypeOf(v))) {
    return .{
        v[0],
        v[1],
    };
}

pub inline fn xyz(v: anytype) @Vector(3, childType(@TypeOf(v))) {
    return .{
        v[0],
        v[1],
        v[2],
    };
}

pub inline fn concatScalar(v: anytype, scalar: anytype) @Vector(vecLenght(@TypeOf(v)) + 1, childType(@TypeOf(v))) {
    const len = vecLenght(@TypeOf(v));
    var b: @Vector(len + 1, childType(@TypeOf(v))) = undefined;
    inline for (0..len) |i| b[i] = v[i];
    b[len] = scalar;
    return b;
}

pub inline fn color(r: u8, g: u8, b: u8, a: u8) Color {
    return .{ .rgba = .{ r, g, b, a } };
}

pub const Color align(@alignOf(u32)) = extern struct {
    rgba: [4]u8 = .{ 255, 255, 255, 255 },

    // todo: other methods

    pub inline fn init(r: u8, g: u8, b: u8, a: u8) Color {
        return .{ .rgba = .{ r, g, b, a } };
    }

    pub inline fn normalize(self: Color) F32x4 {
        var u: U32x4 = @splat(@as(u32, @bitCast(self.rgba)));
        // if (endianess == .Big) {
        //     u &= U32x4{ 0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff };
        //     u >>= U32x4{ 24, 16, 8, 0 };
        // } else {
        u &= U32x4{ 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000 };
        u >>= U32x4{ 0, 8, 16, 24 };
        // }
        var val: F32x4 = @floatFromInt(u);
        val *= @splat(1.0 / 255.0);
        return val;
    }

    /// RMSE of 0.002387
    pub inline fn linearFast(self: Color) F32x4 {
        const val = self.normalize();
        return @shuffle(f32, val * val, val, [4]i32{ 0, 1, 2, ~@as(i32, 3) });
    }

    /// RMSE of 0.001064, see https://www.desmos.com/calculator/rbjqw6re8i
    pub inline fn linearAprox(self: Color) F32x4 {
        const val = self.normalize();
        const a = val * val;
        var b = a * @sqrt(val);
        b = f32x4s(0.567) * a + f32x4s(1 - 0.567) * b;
        return @shuffle(f32, b, val, [4]i32{ 0, 1, 2, ~@as(i32, 3) });
    }

    pub inline fn pack(self: Color) u32 {
        var v: u32 = 0;
        inline for (0..4) |i| v |= @as(u32, @intCast(self.rgba[i])) << (i * 8);
        return v;
    }

    pub inline fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !Color {
        return @as(Color, @bitCast(try std.json.innerParse([4]u8, allocator, source, options)));
    }
};

// todo: moveTowards
// todo: rotateTowards

pub const Scalar = struct {
    pub inline fn copysign(v: f32, sign: f32) f32 {
        return std.math.copysign(v, sign);
    }

    pub fn smoothDamp(s0: f32, s1: f32, vel_ptr: *f32, damping: f32, max_speed: f32, dt: f32) f32 {
        // based on Game Programming Gems 4 Chapter 1.10
        const d = @max(damping, 0.0001);

        var ds = s0 - s1;

        // clamp maximum speed
        const ds_max = max_speed * d;
        ds = clamp(ds, -ds_max, ds_max);

        const omega = 2.0 / d;
        const x = omega * dt;
        const exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x);
        const target = s0 - ds;
        const k3 = (vel_ptr.* + omega * ds) * dt;
        var s = target + (ds + k3) * exp;
        vel_ptr.* = (vel_ptr.* - omega * k3) * exp;

        // prevent overshooting
        if ((s1 - s0 > 0.0) == (s > s1)) {
            s = target;
            vel_ptr.* = (s - s1) / dt;
        }

        return s;
    }
};

pub fn Vector(comptime N: comptime_int) type {
    return struct {
        pub const T = @Vector(N, f32);

        pub fn smoothDamp(s0: T, s1: T, vel_ptr: *T, damping: f32, max_speed: f32, dt: f32) T {
            // based on Game Programming Gems 4 Chapter 1.10
            const d = @max(damping, 0.0001);

            var ds = s0 - s1;

            // clamp maximum speed
            const ds_max = max_speed * d;
            const m_sq = lenghtSq(ds);
            if (m_sq > ds_max * ds_max) {
                ds = ds * splat(T, ds_max / @sqrt(m_sq));
            }

            const omega = 2.0 / d;
            const x = omega * dt;
            const exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x);
            const target = s0 - ds;
            const k0 = splat(T, omega);
            const k1 = splat(T, dt);
            const k2 = splat(T, exp);
            const k3 = (vel_ptr.* + k0 * ds) * k1;
            var s = target + (ds + k3) * k2;
            vel_ptr.* = (vel_ptr.* - k0 * k3) * k2;

            // prevent overshooting
            if (@reduce(.Add, (s1 - s0) * (s - s1)) > 0.0) {
                s = target;
                vel_ptr.* = (s - s1) / k1;
            }

            return s;
        }
    };
}

pub const Box = extern struct {
    min: F32x4 = @splat(0.0),
    max: F32x4 = @splat(0.0),

    pub inline fn init(min: F32x4, max: F32x4) Box {
        return .{ .min = min, .max = max };
    }

    pub inline fn fromAABB(aabb: AABB) Box {
        return Box.init(aabb.center - aabb.extents, aabb.center + aabb.extents);
    }

    pub inline fn encapsulatePoint(self: Box, point: F32x4) Box {
        return .{ .min = @min(self.min, point), .max = @max(self.max, point) };
    }

    pub inline fn rectXY(box: Box) Rect {
        return .{ .xyxy = @shuffle(f32, box.min, box.max, [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1) }) };
    }
};

pub const AABB = extern struct {
    center: F32x4 = @splat(0.0),
    extents: F32x4 = @splat(0.0),

    pub inline fn init(center: F32x4, extents: F32x4) AABB {
        return .{ .center = center, .extents = extents };
    }

    pub inline fn fromBox(box: Box) AABB {
        const center = (box.max + box.min) * f32x4s(0.5);
        return AABB.init(center, box.max - center);
    }

    pub inline fn encapsulatePoint(self: Box, point: F32x4) AABB {
        return .{ .min = @min(self.min, point), .max = @max(self.max, point) };
    }

    // from: https://gist.github.com/cmf028/81e8d3907035640ee0e3fdd69ada543f
    /// transform a AABB it surppots both `Mat4` and `Mat4A`
    pub fn transform(self: AABB, mat: anytype) AABB {
        // transform center
        const t_center = mat.mulVector(swizzle(self.center, .x, .y, .z, .one));

        // transform extents (take maximum)
        const vx = swizzle(self.extents, .x, .x, .x, .zero);
        const vy = swizzle(self.extents, .y, .y, .y, .zero);
        const vz = swizzle(self.extents, .z, .z, .z, .zero);
        const t_extents = @abs(mat.rows[0] * vx) + @abs(mat.rows[1] * vy) + @abs(mat.rows[2] * vz);

        return .{ .center = t_center, .extents = t_extents };
    }

    pub inline fn rectXY(self: AABB) Rect {
        return Box.fromAABB(self).rectXY();
    }

    // todo: pub fn cull(self: AABB, frustum: [6]Plane) bool;
};

pub const Rect = extern struct {
    xyxy: F32x4 = @splat(0.0),

    // todo: pub inline fn containsPoint(self: Rect, point: F32x2) bool;

    pub inline fn overlaps(r0: Rect, r1: Rect) bool {
        // zig fmt: off
        const a = @shuffle(f32, r0.xyxy, r1.xyxy, [4]i32{            0,            1, ~@as(i32, 0), ~@as(i32, 1) });
        const b = @shuffle(f32, r0.xyxy, r1.xyxy, [4]i32{ ~@as(i32, 2), ~@as(i32, 3),            2,            3 });
        // zig fmt: on
        return @reduce(.And, a <= b);
    }
};

pub const Plane = extern struct {
    xyzd: F32x4 = .{ 0, 1, 0, 0 },

    pub inline fn fromOriginNormal(o: F32x4, n: F32x4) Plane {
        return .{ .xyzd = shuffle(n, dotSplat(o, n), .x0, .y0, .z0, .w1) };
    }

    pub inline fn origin(self: Plane) F32x4 {
        return swizzle(self.xyzd * f32x4s(self.xyzd[3]), .x, .y, .z, .zero);
    }

    pub inline fn normalize(self: Plane) Plane {
        var v = swizzle(self.xyzd, .x, .y, .z, .zero);
        const n = 1.0 / lenght(v);
        v[3] = self.xyzd[3];
        v *= f32x4s(n);
        return .{ .xyzd = v };
    }
};

inline fn vecLenght(comptime T: type) comptime_int {
    return @typeInfo(T).Vector.len;
}

inline fn childType(comptime T: type) type {
    return @typeInfo(T).Vector.child;
}

/// converts a vector `v` into an array of lenght `len`
pub inline fn arrTrunc(comptime len: usize, v: anytype) [len]childType(@TypeOf(v)) {
    var a: [len]childType(@TypeOf(v)) = undefined;
    inline for (0..len) |i| a[i] = v[i];
    return a;
}

/// converts a vector `v` into an array
pub inline fn arr(v: anytype) [vecLenght(@TypeOf(v))]childType(@TypeOf(v)) {
    return arrTrunc(vecLenght(@TypeOf(v)), v);
}

pub inline fn modAngle(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    return v - splat(T, math.tau) * @round(v * splat(T, 1.0 / math.tau)); // 2 x vmulps, 2 x load, vroundps, vaddps
}

pub inline fn negativeZero(comptime T: type) T {
    return splat(T, @as(f32, @bitCast(@as(u32, 0x8000_0000))));
}

/// encodes an id into the packed color RGB channels alpha is always equal to 255
pub inline fn encodeId(id: u32) u32 {
    return Color.init(@intCast(id & 255), @intCast((id >> 8) & 255), @intCast((id >> 16) & 255), 255).pack();
}
