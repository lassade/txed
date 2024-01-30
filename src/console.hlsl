
struct Char {
    uint index;
    uint flags;
};

cbuffer Config : register(b0) {
    uint2 slot_size;
    uint2 font_atlas_size;
    uint2 console_size;
    float4 colors[16];
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

    bool cursor_line = (c.flags & 0x00000100) != 0;
    bool cursor_block = (c.flags & 0x00000200) != 0;
    bool box = (c.flags & 0x00000400) != 0;
    // todo: bool selected = (c.flags & 0x02000000) != 0;
    // todo: bool line_over = (c.flags & 0x04000000) != 0;

    float4 fg_color = colors[(c.flags & 0x0000000f)];
    float4 bg_color = colors[(c.flags & 0x000000f0) >> 4];

    uint2 box_max = slot_size - uint2(2, 2);

    for (uint x = 0; x < slot_size.x; x++) {
        for (uint y = 0; y < slot_size.y; y++) {
            uint2 p1 = uint2(slot_size.x * id.x + x, slot_size.y * id.y + y);

            uint2 p0 = uint2(slot_size.x * font_x + x, slot_size.y * font_y + y);
            float alpha = font_atlas[p0];

            if (cursor_block || (cursor_line && x < 2)) {
                alpha = 1.0 - alpha;
            } else if (box && (x < 1 || y < 1 || x > box_max.x || y > box_max.y)) {
                alpha = 1.0 - alpha;
            }
            
            screen[p1] = lerp(bg_color, fg_color, alpha);
        }
    }
}