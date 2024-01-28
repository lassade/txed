
struct Char {
    uint index;
    uint color;
};

cbuffer Config : register(b0) {
    uint2 slot_size;
    uint2 font_atlas_size;
    uint2 console_size;
    uint2 screen_size;
    float4 bg_color;
};
StructuredBuffer<Char> console: register(t0);
Texture2D<float> font_atlas : register(t1);
RWTexture2D<float4> screen : register(u0);

[numthreads(16, 16, 1)]
void csMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= console_size.x) return;
    if (id.y >= console_size.y) return;

    // current console char
    uint i = id.y * console_size.x + id.x;

    Char c = console[i];
    if (c.index >= font_atlas_size.x * font_atlas_size.y) return;

    // atlas pos
    uint font_y = c.index / font_atlas_size.x;
    uint font_x = c.index - (font_y * font_atlas_size.x);

    // todo: int flags = c.color & 0x000000ff;
    // for things like invert bg/fg

    float4 fg_color = float4(
        float(c.color & 0xff000000 >> 24) / 255.0,
        float(c.color & 0x00ff0000 >> 16) / 255.0,
        float(c.color & 0x0000ff00 >> 8)  / 255.0,
        1.0
    );

    for (uint x = 0; x < slot_size.x; x++) {
        for (uint y = 0; y < slot_size.y; y++) {
            uint2 p1 = uint2(slot_size.x * id.x + x, slot_size.y * id.y + y);
            if (p1.x >= screen_size.x) continue;
            if (p1.y >= screen_size.y) continue;

            uint2 p0 = uint2(slot_size.x * font_x + x, slot_size.y * font_y + y);
            float alpha = font_atlas[p0];
            screen[p1] = lerp(bg_color, fg_color, alpha);
        }
    }
}