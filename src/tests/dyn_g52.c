/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * dyn_g52: many draws in one pass on the Bifrost (v7) path, the way an
 * emulator records them: a pipeline with dynamic stencil reference, one
 * descriptor set with two dynamic uniform buffers (one read by the vertex
 * shader, one by the fragment shader) rebound before every draw, most of
 * the time with new offsets for one stage only, sometimes with the same
 * offsets, sometimes through a second set holding the same buffers; two
 * pipeline objects of the same state bound in turns. It checks the draw
 * state the driver keeps between draws instead of building it again (the
 * dynamic-state bake, the descriptor tables and FAU blocks of a stage whose
 * offsets did not change) and the order of the jobs in the chain (Tiler
 * jobs linked behind later draws' Vertex jobs).
 *
 * One 64x64 pass into a linear RGBA8 image (cleared to black) with a
 * depth/stencil image (stencil cleared to 0), in three parts:
 *
 *   1. 56 draws, one 8x8 cell each of rows 0 to 6 (cell i at column i % 8,
 *      row i / 8): the vertex shader places a unit quad by its uniform
 *      buffer (slot i), the fragment shader writes the colour of its
 *      uniform buffer (slot i / 4: the fragment offsets change every
 *      fourth draw, the vertex ones every draw), stencil ALWAYS / REPLACE
 *      with the dynamic reference (i / 2) % 3 + 1 (set before every draw,
 *      changing every second one), red and green written;
 *   2. row 7, three overlapping draws of one pipeline: the whole row in
 *      colour X, its left three quarters in Y, its left half in Z (last
 *      one wins);
 *   3. three full-screen draws with stencil EQUAL reference 1, 2, 3, blue
 *      only, each writing blue r * 64: the blue of a cell of rows 0 to 6
 *      tells the stencil reference its draw used.
 *
 * Expected: cell i of rows 0 to 6 has the red and green of fragment slot
 * i / 4 and blue ((i / 2) % 3 + 1) * 64; row 7 is Z over x < 32, Y over
 * 32 <= x < 48, X over x >= 48 (blue 0). The command buffer is submitted
 * twice (it is not one-time-submit): before the second run the host
 * rewrites the fragment colours, which the second readback must show.
 *
 * Then the set switch cases, each its own pass into the same image (cleared
 * to black): a pipeline layout whose set 0 holds one binding, a dynamic
 * uniform or storage buffer, so the set has no descriptor slots of its own,
 * only the dynamic buffer. Set A points at buffer X holding colour 1, set B
 * at buffer Y holding colour 2 (both 16 bytes at the bound offset). Set A is
 * bound and a quad drawn over the left half, then set B at the same dynamic
 * offset and a quad over the right half, one pipeline throughout. Expected:
 * the left half in colour 1, the right half in colour 2. A driver that
 * matches the bound sets by their descriptor memory and slot count (address
 * 0 and no slots for every such set, so A and B look alike) and keeps a
 * stage's descriptor tables and FAU uniform words when those sets come
 * with unchanged offsets draws the right half in colour 1. The cases are
 * every combination of
 *
 *   - UNIFORM_BUFFER_DYNAMIC (read at a constant offset, which the driver
 *     may push into FAU) or STORAGE_BUFFER_DYNAMIC;
 *   - visible to and read by the fragment shader, or by the vertex shader,
 *     which hands the colour on as a flat varying;
 *   - both sets bound at offset 0, or both at the same non-zero offset (the
 *     largest of 256 and the device's uniform and storage buffer offset
 *     alignments; offset 0 of X and Y then holds a decoy colour of its own,
 *     which a dropped offset would show).
 *
 * Shaders (glslangValidator 16.6.0, -V --target-env vulkan1.1):
 *
 *   dyn.vert:  #version 450
 *              layout(location = 0) in vec2 pos;
 *              layout(set = 0, binding = 0) uniform V { vec4 rect; } v;
 *              void main() { gl_Position = vec4(v.rect.xy + pos * v.rect.zw, 0.5, 1.0); }
 *
 *   dyn.frag:  #version 450
 *              layout(location = 0) out vec4 color;
 *              layout(set = 0, binding = 1) uniform F { vec4 c; } f;
 *              void main() { color = f.c; }
 *
 *   The set switch cases:
 *
 *   sw.vert:       #version 450
 *                  layout(location = 0) in vec2 pos;
 *                  void main() { gl_Position = vec4(pos, 0.5, 1.0); }
 *
 *   sw_ubo.frag:   #version 450
 *                  layout(location = 0) out vec4 color;
 *                  layout(set = 0, binding = 0) uniform C { vec4 c; } u;
 *                  void main() { color = u.c; }
 *
 *   sw_ssbo.frag:  #version 450
 *                  layout(location = 0) out vec4 color;
 *                  layout(set = 0, binding = 0, std430) readonly buffer C { vec4 c; } b;
 *                  void main() { color = b.c; }
 *
 *   sw_ubo.vert:   #version 450
 *                  layout(location = 0) in vec2 pos;
 *                  layout(location = 0) flat out vec4 v_color;
 *                  layout(set = 0, binding = 0) uniform C { vec4 c; } u;
 *                  void main() { gl_Position = vec4(pos, 0.5, 1.0); v_color = u.c; }
 *
 *   sw_ssbo.vert:  #version 450
 *                  layout(location = 0) in vec2 pos;
 *                  layout(location = 0) flat out vec4 v_color;
 *                  layout(set = 0, binding = 0, std430) readonly buffer C { vec4 c; } b;
 *                  void main() { gl_Position = vec4(pos, 0.5, 1.0); v_color = b.c; }
 *
 *   sw_flat.frag:  #version 450
 *                  layout(location = 0) flat in vec4 v_color;
 *                  layout(location = 0) out vec4 color;
 *                  void main() { color = v_color; }
 *
 * Usage: dyn_g52 [path/to/libvulkan_malisx2.so]
 *   The path defaults to $MALISX2_ICD, then to DYN_G52_DEFAULT_ICD.
 *
 * Exit status: 0 PASS; 1 FAIL (wrong pixels, a timeout or a Vulkan error
 * once the device exists); 2 the program did not get as far as a
 * VkDevice.
 */

#include "g52_harness.h"

#ifndef DYN_G52_DEFAULT_ICD
#define DYN_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define IMG_W 64
#define IMG_H 64
#define CELL 8
#define CELLS 56            /* rows 0 to 6 */
#define SENTINEL 0xa5
#define VS_SLOTS (CELLS + 3 + 1)   /* the cells, the three row-7 rects, full screen */
#define FS_SLOTS (CELLS / 4 + 3 + 3)
#define SW_HALF (IMG_W / 2)        /* set switch cases: the left quad's width */
#define SW_CASES 8u

/* ---------------------------------------------------------------------- */
/* Shaders (see the file comment)                                          */

static const uint32_t dyn_vert_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000027, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000000,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000019, 0x00030003,
   0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
   0x00060005, 0x0000000b, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000,
   0x00060006, 0x0000000b, 0x00000000, 0x505f6c67, 0x7469736f, 0x006e6f69,
   0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953,
   0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c,
   0x61747369, 0x0065636e, 0x00070006, 0x0000000b, 0x00000003, 0x435f6c67,
   0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x0000000d, 0x00000000,
   0x00030005, 0x00000010, 0x00000056, 0x00050006, 0x00000010, 0x00000000,
   0x74636572, 0x00000000, 0x00030005, 0x00000012, 0x00000076, 0x00030005,
   0x00000019, 0x00736f70, 0x00030047, 0x0000000b, 0x00000002, 0x00050048,
   0x0000000b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000000b,
   0x00000001, 0x0000000b, 0x00000001, 0x00050048, 0x0000000b, 0x00000002,
   0x0000000b, 0x00000003, 0x00050048, 0x0000000b, 0x00000003, 0x0000000b,
   0x00000004, 0x00030047, 0x00000010, 0x00000002, 0x00050048, 0x00000010,
   0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000012, 0x00000021,
   0x00000000, 0x00040047, 0x00000012, 0x00000022, 0x00000000, 0x00040047,
   0x00000019, 0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021,
   0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017,
   0x00000007, 0x00000006, 0x00000004, 0x00040015, 0x00000008, 0x00000020,
   0x00000000, 0x0004002b, 0x00000008, 0x00000009, 0x00000001, 0x0004001c,
   0x0000000a, 0x00000006, 0x00000009, 0x0006001e, 0x0000000b, 0x00000007,
   0x00000006, 0x0000000a, 0x0000000a, 0x00040020, 0x0000000c, 0x00000003,
   0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000003, 0x00040015,
   0x0000000e, 0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f,
   0x00000000, 0x0003001e, 0x00000010, 0x00000007, 0x00040020, 0x00000011,
   0x00000002, 0x00000010, 0x0004003b, 0x00000011, 0x00000012, 0x00000002,
   0x00040017, 0x00000013, 0x00000006, 0x00000002, 0x00040020, 0x00000014,
   0x00000002, 0x00000007, 0x00040020, 0x00000018, 0x00000001, 0x00000013,
   0x0004003b, 0x00000018, 0x00000019, 0x00000001, 0x0004002b, 0x00000006,
   0x00000020, 0x3f000000, 0x0004002b, 0x00000006, 0x00000021, 0x3f800000,
   0x00040020, 0x00000025, 0x00000003, 0x00000007, 0x00050036, 0x00000002,
   0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041,
   0x00000014, 0x00000015, 0x00000012, 0x0000000f, 0x0004003d, 0x00000007,
   0x00000016, 0x00000015, 0x0007004f, 0x00000013, 0x00000017, 0x00000016,
   0x00000016, 0x00000000, 0x00000001, 0x0004003d, 0x00000013, 0x0000001a,
   0x00000019, 0x00050041, 0x00000014, 0x0000001b, 0x00000012, 0x0000000f,
   0x0004003d, 0x00000007, 0x0000001c, 0x0000001b, 0x0007004f, 0x00000013,
   0x0000001d, 0x0000001c, 0x0000001c, 0x00000002, 0x00000003, 0x00050085,
   0x00000013, 0x0000001e, 0x0000001a, 0x0000001d, 0x00050081, 0x00000013,
   0x0000001f, 0x00000017, 0x0000001e, 0x00050051, 0x00000006, 0x00000022,
   0x0000001f, 0x00000000, 0x00050051, 0x00000006, 0x00000023, 0x0000001f,
   0x00000001, 0x00070050, 0x00000007, 0x00000024, 0x00000022, 0x00000023,
   0x00000020, 0x00000021, 0x00050041, 0x00000025, 0x00000026, 0x0000000d,
   0x0000000f, 0x0003003e, 0x00000026, 0x00000024, 0x000100fd, 0x00010038,
};

static const uint32_t dyn_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000012, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
   0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
   0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63, 0x00000072,
   0x00030005, 0x0000000a, 0x00000046, 0x00040006, 0x0000000a, 0x00000000,
   0x00000063, 0x00030005, 0x0000000c, 0x00000066, 0x00040047, 0x00000009,
   0x0000001e, 0x00000000, 0x00030047, 0x0000000a, 0x00000002, 0x00050048,
   0x0000000a, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000000c,
   0x00000021, 0x00000001, 0x00040047, 0x0000000c, 0x00000022, 0x00000000,
   0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
   0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
   0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008,
   0x00000009, 0x00000003, 0x0003001e, 0x0000000a, 0x00000007, 0x00040020,
   0x0000000b, 0x00000002, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c,
   0x00000002, 0x00040015, 0x0000000d, 0x00000020, 0x00000001, 0x0004002b,
   0x0000000d, 0x0000000e, 0x00000000, 0x00040020, 0x0000000f, 0x00000002,
   0x00000007, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
   0x000200f8, 0x00000005, 0x00050041, 0x0000000f, 0x00000010, 0x0000000c,
   0x0000000e, 0x0004003d, 0x00000007, 0x00000011, 0x00000010, 0x0003003e,
   0x00000009, 0x00000011, 0x000100fd, 0x00010038,
};

/* The set switch cases. */

static const uint32_t sw_vert_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x0000001b, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000000,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000012, 0x00030003,
   0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
   0x00060005, 0x0000000b, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000,
   0x00060006, 0x0000000b, 0x00000000, 0x505f6c67, 0x7469736f, 0x006e6f69,
   0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953,
   0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c,
   0x61747369, 0x0065636e, 0x00070006, 0x0000000b, 0x00000003, 0x435f6c67,
   0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x0000000d, 0x00000000,
   0x00030005, 0x00000012, 0x00736f70, 0x00030047, 0x0000000b, 0x00000002,
   0x00050048, 0x0000000b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048,
   0x0000000b, 0x00000001, 0x0000000b, 0x00000001, 0x00050048, 0x0000000b,
   0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x0000000b, 0x00000003,
   0x0000000b, 0x00000004, 0x00040047, 0x00000012, 0x0000001e, 0x00000000,
   0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
   0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
   0x00040015, 0x00000008, 0x00000020, 0x00000000, 0x0004002b, 0x00000008,
   0x00000009, 0x00000001, 0x0004001c, 0x0000000a, 0x00000006, 0x00000009,
   0x0006001e, 0x0000000b, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a,
   0x00040020, 0x0000000c, 0x00000003, 0x0000000b, 0x0004003b, 0x0000000c,
   0x0000000d, 0x00000003, 0x00040015, 0x0000000e, 0x00000020, 0x00000001,
   0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040017, 0x00000010,
   0x00000006, 0x00000002, 0x00040020, 0x00000011, 0x00000001, 0x00000010,
   0x0004003b, 0x00000011, 0x00000012, 0x00000001, 0x0004002b, 0x00000006,
   0x00000014, 0x3f000000, 0x0004002b, 0x00000006, 0x00000015, 0x3f800000,
   0x00040020, 0x00000019, 0x00000003, 0x00000007, 0x00050036, 0x00000002,
   0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d,
   0x00000010, 0x00000013, 0x00000012, 0x00050051, 0x00000006, 0x00000016,
   0x00000013, 0x00000000, 0x00050051, 0x00000006, 0x00000017, 0x00000013,
   0x00000001, 0x00070050, 0x00000007, 0x00000018, 0x00000016, 0x00000017,
   0x00000014, 0x00000015, 0x00050041, 0x00000019, 0x0000001a, 0x0000000d,
   0x0000000f, 0x0003003e, 0x0000001a, 0x00000018, 0x000100fd, 0x00010038,
};

static const uint32_t sw_ubo_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000012, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
   0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
   0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63, 0x00000072,
   0x00030005, 0x0000000a, 0x00000043, 0x00040006, 0x0000000a, 0x00000000,
   0x00000063, 0x00030005, 0x0000000c, 0x00000075, 0x00040047, 0x00000009,
   0x0000001e, 0x00000000, 0x00030047, 0x0000000a, 0x00000002, 0x00050048,
   0x0000000a, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000000c,
   0x00000021, 0x00000000, 0x00040047, 0x0000000c, 0x00000022, 0x00000000,
   0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
   0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
   0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008,
   0x00000009, 0x00000003, 0x0003001e, 0x0000000a, 0x00000007, 0x00040020,
   0x0000000b, 0x00000002, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c,
   0x00000002, 0x00040015, 0x0000000d, 0x00000020, 0x00000001, 0x0004002b,
   0x0000000d, 0x0000000e, 0x00000000, 0x00040020, 0x0000000f, 0x00000002,
   0x00000007, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
   0x000200f8, 0x00000005, 0x00050041, 0x0000000f, 0x00000010, 0x0000000c,
   0x0000000e, 0x0004003d, 0x00000007, 0x00000011, 0x00000010, 0x0003003e,
   0x00000009, 0x00000011, 0x000100fd, 0x00010038,
};

static const uint32_t sw_ssbo_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000012, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
   0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
   0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63, 0x00000072,
   0x00030005, 0x0000000a, 0x00000043, 0x00040006, 0x0000000a, 0x00000000,
   0x00000063, 0x00030005, 0x0000000c, 0x00000062, 0x00040047, 0x00000009,
   0x0000001e, 0x00000000, 0x00030047, 0x0000000a, 0x00000002, 0x00040048,
   0x0000000a, 0x00000000, 0x00000018, 0x00050048, 0x0000000a, 0x00000000,
   0x00000023, 0x00000000, 0x00030047, 0x0000000c, 0x00000018, 0x00040047,
   0x0000000c, 0x00000021, 0x00000000, 0x00040047, 0x0000000c, 0x00000022,
   0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
   0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006,
   0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b,
   0x00000008, 0x00000009, 0x00000003, 0x0003001e, 0x0000000a, 0x00000007,
   0x00040020, 0x0000000b, 0x0000000c, 0x0000000a, 0x0004003b, 0x0000000b,
   0x0000000c, 0x0000000c, 0x00040015, 0x0000000d, 0x00000020, 0x00000001,
   0x0004002b, 0x0000000d, 0x0000000e, 0x00000000, 0x00040020, 0x0000000f,
   0x0000000c, 0x00000007, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
   0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x0000000f, 0x00000010,
   0x0000000c, 0x0000000e, 0x0004003d, 0x00000007, 0x00000011, 0x00000010,
   0x0003003e, 0x00000009, 0x00000011, 0x000100fd, 0x00010038,
};

static const uint32_t sw_ubo_vert_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000022, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0008000f, 0x00000000,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000012, 0x0000001b,
   0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
   0x00000000, 0x00060005, 0x0000000b, 0x505f6c67, 0x65567265, 0x78657472,
   0x00000000, 0x00060006, 0x0000000b, 0x00000000, 0x505f6c67, 0x7469736f,
   0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f,
   0x657a6953, 0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67,
   0x4470696c, 0x61747369, 0x0065636e, 0x00070006, 0x0000000b, 0x00000003,
   0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x0000000d,
   0x00000000, 0x00030005, 0x00000012, 0x00736f70, 0x00040005, 0x0000001b,
   0x6f635f76, 0x00726f6c, 0x00030005, 0x0000001c, 0x00000043, 0x00040006,
   0x0000001c, 0x00000000, 0x00000063, 0x00030005, 0x0000001e, 0x00000075,
   0x00030047, 0x0000000b, 0x00000002, 0x00050048, 0x0000000b, 0x00000000,
   0x0000000b, 0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x0000000b,
   0x00000001, 0x00050048, 0x0000000b, 0x00000002, 0x0000000b, 0x00000003,
   0x00050048, 0x0000000b, 0x00000003, 0x0000000b, 0x00000004, 0x00040047,
   0x00000012, 0x0000001e, 0x00000000, 0x00030047, 0x0000001b, 0x0000000e,
   0x00040047, 0x0000001b, 0x0000001e, 0x00000000, 0x00030047, 0x0000001c,
   0x00000002, 0x00050048, 0x0000001c, 0x00000000, 0x00000023, 0x00000000,
   0x00040047, 0x0000001e, 0x00000021, 0x00000000, 0x00040047, 0x0000001e,
   0x00000022, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
   0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007,
   0x00000006, 0x00000004, 0x00040015, 0x00000008, 0x00000020, 0x00000000,
   0x0004002b, 0x00000008, 0x00000009, 0x00000001, 0x0004001c, 0x0000000a,
   0x00000006, 0x00000009, 0x0006001e, 0x0000000b, 0x00000007, 0x00000006,
   0x0000000a, 0x0000000a, 0x00040020, 0x0000000c, 0x00000003, 0x0000000b,
   0x0004003b, 0x0000000c, 0x0000000d, 0x00000003, 0x00040015, 0x0000000e,
   0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000,
   0x00040017, 0x00000010, 0x00000006, 0x00000002, 0x00040020, 0x00000011,
   0x00000001, 0x00000010, 0x0004003b, 0x00000011, 0x00000012, 0x00000001,
   0x0004002b, 0x00000006, 0x00000014, 0x3f000000, 0x0004002b, 0x00000006,
   0x00000015, 0x3f800000, 0x00040020, 0x00000019, 0x00000003, 0x00000007,
   0x0004003b, 0x00000019, 0x0000001b, 0x00000003, 0x0003001e, 0x0000001c,
   0x00000007, 0x00040020, 0x0000001d, 0x00000002, 0x0000001c, 0x0004003b,
   0x0000001d, 0x0000001e, 0x00000002, 0x00040020, 0x0000001f, 0x00000002,
   0x00000007, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
   0x000200f8, 0x00000005, 0x0004003d, 0x00000010, 0x00000013, 0x00000012,
   0x00050051, 0x00000006, 0x00000016, 0x00000013, 0x00000000, 0x00050051,
   0x00000006, 0x00000017, 0x00000013, 0x00000001, 0x00070050, 0x00000007,
   0x00000018, 0x00000016, 0x00000017, 0x00000014, 0x00000015, 0x00050041,
   0x00000019, 0x0000001a, 0x0000000d, 0x0000000f, 0x0003003e, 0x0000001a,
   0x00000018, 0x00050041, 0x0000001f, 0x00000020, 0x0000001e, 0x0000000f,
   0x0004003d, 0x00000007, 0x00000021, 0x00000020, 0x0003003e, 0x0000001b,
   0x00000021, 0x000100fd, 0x00010038,
};

static const uint32_t sw_ssbo_vert_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000022, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0008000f, 0x00000000,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000012, 0x0000001b,
   0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
   0x00000000, 0x00060005, 0x0000000b, 0x505f6c67, 0x65567265, 0x78657472,
   0x00000000, 0x00060006, 0x0000000b, 0x00000000, 0x505f6c67, 0x7469736f,
   0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f,
   0x657a6953, 0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67,
   0x4470696c, 0x61747369, 0x0065636e, 0x00070006, 0x0000000b, 0x00000003,
   0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x0000000d,
   0x00000000, 0x00030005, 0x00000012, 0x00736f70, 0x00040005, 0x0000001b,
   0x6f635f76, 0x00726f6c, 0x00030005, 0x0000001c, 0x00000043, 0x00040006,
   0x0000001c, 0x00000000, 0x00000063, 0x00030005, 0x0000001e, 0x00000062,
   0x00030047, 0x0000000b, 0x00000002, 0x00050048, 0x0000000b, 0x00000000,
   0x0000000b, 0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x0000000b,
   0x00000001, 0x00050048, 0x0000000b, 0x00000002, 0x0000000b, 0x00000003,
   0x00050048, 0x0000000b, 0x00000003, 0x0000000b, 0x00000004, 0x00040047,
   0x00000012, 0x0000001e, 0x00000000, 0x00030047, 0x0000001b, 0x0000000e,
   0x00040047, 0x0000001b, 0x0000001e, 0x00000000, 0x00030047, 0x0000001c,
   0x00000002, 0x00040048, 0x0000001c, 0x00000000, 0x00000018, 0x00050048,
   0x0000001c, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000001e,
   0x00000018, 0x00040047, 0x0000001e, 0x00000021, 0x00000000, 0x00040047,
   0x0000001e, 0x00000022, 0x00000000, 0x00020013, 0x00000002, 0x00030021,
   0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017,
   0x00000007, 0x00000006, 0x00000004, 0x00040015, 0x00000008, 0x00000020,
   0x00000000, 0x0004002b, 0x00000008, 0x00000009, 0x00000001, 0x0004001c,
   0x0000000a, 0x00000006, 0x00000009, 0x0006001e, 0x0000000b, 0x00000007,
   0x00000006, 0x0000000a, 0x0000000a, 0x00040020, 0x0000000c, 0x00000003,
   0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000003, 0x00040015,
   0x0000000e, 0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f,
   0x00000000, 0x00040017, 0x00000010, 0x00000006, 0x00000002, 0x00040020,
   0x00000011, 0x00000001, 0x00000010, 0x0004003b, 0x00000011, 0x00000012,
   0x00000001, 0x0004002b, 0x00000006, 0x00000014, 0x3f000000, 0x0004002b,
   0x00000006, 0x00000015, 0x3f800000, 0x00040020, 0x00000019, 0x00000003,
   0x00000007, 0x0004003b, 0x00000019, 0x0000001b, 0x00000003, 0x0003001e,
   0x0000001c, 0x00000007, 0x00040020, 0x0000001d, 0x0000000c, 0x0000001c,
   0x0004003b, 0x0000001d, 0x0000001e, 0x0000000c, 0x00040020, 0x0000001f,
   0x0000000c, 0x00000007, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
   0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000010, 0x00000013,
   0x00000012, 0x00050051, 0x00000006, 0x00000016, 0x00000013, 0x00000000,
   0x00050051, 0x00000006, 0x00000017, 0x00000013, 0x00000001, 0x00070050,
   0x00000007, 0x00000018, 0x00000016, 0x00000017, 0x00000014, 0x00000015,
   0x00050041, 0x00000019, 0x0000001a, 0x0000000d, 0x0000000f, 0x0003003e,
   0x0000001a, 0x00000018, 0x00050041, 0x0000001f, 0x00000020, 0x0000001e,
   0x0000000f, 0x0004003d, 0x00000007, 0x00000021, 0x00000020, 0x0003003e,
   0x0000001b, 0x00000021, 0x000100fd, 0x00010038,
};

static const uint32_t sw_flat_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x0000000d, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000b, 0x00030010,
   0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
   0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63,
   0x00000072, 0x00040005, 0x0000000b, 0x6f635f76, 0x00726f6c, 0x00040047,
   0x00000009, 0x0000001e, 0x00000000, 0x00030047, 0x0000000b, 0x0000000e,
   0x00040047, 0x0000000b, 0x0000001e, 0x00000000, 0x00020013, 0x00000002,
   0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
   0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008,
   0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003,
   0x00040020, 0x0000000a, 0x00000001, 0x00000007, 0x0004003b, 0x0000000a,
   0x0000000b, 0x00000001, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
   0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000007, 0x0000000c,
   0x0000000b, 0x0003003e, 0x00000009, 0x0000000c, 0x000100fd, 0x00010038,
};

/* ---------------------------------------------------------------------- */
/* What the harness does not load                                          */

struct dyn_fns {
   PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
   PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
   PFN_vkCreateDescriptorPool CreateDescriptorPool;
   PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
   PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
   PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
   PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
   PFN_vkCmdSetStencilReference CmdSetStencilReference;
};

/* ---------------------------------------------------------------------- */
/* Colours                                                                 */

/* Red and green of fragment slot k in run 0 or 1, blue 0, alpha 255. */
static void
cell_rgba(unsigned k, unsigned run, uint8_t out[4])
{
   out[0] = run ? (uint8_t)(k * 13 + 100) : (uint8_t)(k * 17 + 20);
   out[1] = run ? (uint8_t)(k * 11 + 7) : (uint8_t)(k * 9 + 40);
   out[2] = 0;
   out[3] = 255;
}

static const uint8_t row7_rgba[3][4] = {
   {250, 30, 0, 255},   /* X: the whole row */
   {30, 250, 0, 255},   /* Y: the left three quarters */
   {120, 120, 0, 255},  /* Z: the left half */
};

static uint8_t
stencil_ref(unsigned i)
{
   return (uint8_t)((i / 2) % 3 + 1);
}

/* ---------------------------------------------------------------------- */
/* Pipelines                                                               */

static VkPipeline
make_pipeline(struct g52 *g, VkPipelineLayout layout, VkFormat color_format, VkFormat zs_format,
              const VkPipelineDepthStencilStateCreateInfo *ds, VkColorComponentFlags write_mask,
              const char *what)
{
   const struct g52_fns *f = &g->f;
   const VkShaderModuleCreateInfo vsmci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(dyn_vert_spv),
      .pCode = dyn_vert_spv,
   };
   const VkShaderModuleCreateInfo fsmci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(dyn_frag_spv),
      .pCode = dyn_frag_spv,
   };
   VkShaderModule vsm = VK_NULL_HANDLE, fsm = VK_NULL_HANDLE;
   VkResult r = f->CreateShaderModule(g->device, &vsmci, NULL, &vsm);
   if (r == VK_SUCCESS)
      r = f->CreateShaderModule(g->device, &fsmci, NULL, &fsm);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateShaderModule failed: %s\n", g->prog, g52_result_str(r));
      return VK_NULL_HANDLE;
   }
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vsm,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fsm,
         .pName = "main",
      },
   };
   const VkVertexInputBindingDescription vb_desc = {
      .binding = 0,
      .stride = 2 * sizeof(float),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription va_desc = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
   };
   const VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vb_desc,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &va_desc,
   };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {0.0f, 0.0f, (float)IMG_W, (float)IMG_H, 0.0f, 1.0f};
   const VkRect2D scissor = {{0, 0}, {IMG_W, IMG_H}};
   const VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkPipelineColorBlendAttachmentState blend = {.colorWriteMask = write_mask};
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend,
   };
   const VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_STENCIL_REFERENCE};
   const VkPipelineDynamicStateCreateInfo dyn = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 1,
      .pDynamicStates = dyn_states,
   };
   const VkPipelineRenderingCreateInfo prci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &color_format,
      .depthAttachmentFormat = zs_format == VK_FORMAT_S8_UINT ? VK_FORMAT_UNDEFINED : zs_format,
      .stencilAttachmentFormat = zs_format,
   };
   const VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &prci,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pDepthStencilState = ds,
      .pColorBlendState = &cb,
      .pDynamicState = &dyn,
      .layout = layout,
   };
   VkPipeline pipeline = VK_NULL_HANDLE;
   r = f->CreateGraphicsPipelines(g->device, VK_NULL_HANDLE, 1, &gpci, NULL, &pipeline);
   f->DestroyShaderModule(g->device, vsm, NULL);
   f->DestroyShaderModule(g->device, fsm, NULL);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateGraphicsPipelines (%s) failed: %s\n", g->prog, what,
              g52_result_str(r));
      return VK_NULL_HANDLE;
   }
   printf("  pipeline %s created\n", what);
   return pipeline;
}

/* ---------------------------------------------------------------------- */
/* The uniform buffer                                                      */

struct ubo {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint8_t *map;
   bool coherent;
   VkDeviceSize align, fs_base, size;
};

static bool
ubo_create(struct g52 *g, struct ubo *u, VkDeviceSize align)
{
   const struct g52_fns *f = &g->f;
   memset(u, 0, sizeof(*u));
   u->align = align;
   u->fs_base = VS_SLOTS * align;
   u->size = u->fs_base + FS_SLOTS * align;
   const VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = u->size,
      .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkResult r = f->CreateBuffer(g->device, &bci, NULL, &u->buffer);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateBuffer (uniforms) failed: %s\n", g->prog, g52_result_str(r));
      return false;
   }
   VkMemoryRequirements reqs;
   f->GetBufferMemoryRequirements(g->device, u->buffer, &reqs);
   const int type = g52_pick_memory_type(g, reqs.memoryTypeBits, true, &u->coherent);
   if (type < 0) {
      fprintf(stderr, "%s: no host-visible memory type for the uniforms\n", g->prog);
      return false;
   }
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)type,
   };
   r = f->AllocateMemory(g->device, &mai, NULL, &u->memory);
   if (r == VK_SUCCESS)
      r = f->BindBufferMemory(g->device, u->buffer, u->memory, 0);
   if (r == VK_SUCCESS)
      r = f->MapMemory(g->device, u->memory, 0, VK_WHOLE_SIZE, 0, (void **)&u->map);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: uniform buffer memory: %s\n", g->prog, g52_result_str(r));
      return false;
   }
   memset(u->map, 0, (size_t)u->size);
   return true;
}

static void
ubo_vs_rect(struct ubo *u, unsigned slot, float x, float y, float w, float h)
{
   const float v[4] = {x, y, w, h};
   memcpy(u->map + slot * u->align, v, sizeof(v));
}

static void
ubo_fs_rgba(struct ubo *u, unsigned slot, const uint8_t rgba[4])
{
   const float v[4] = {rgba[0] / 255.0f, rgba[1] / 255.0f, rgba[2] / 255.0f, rgba[3] / 255.0f};
   memcpy(u->map + u->fs_base + slot * u->align, v, sizeof(v));
}

static void
ubo_flush(struct g52 *g, struct ubo *u)
{
   if (u->coherent)
      return;
   const VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = u->memory,
      .size = VK_WHOLE_SIZE,
   };
   g->f.FlushMappedMemoryRanges(g->device, 1, &whole);
}

/* Fragment slots: 0 .. CELLS/4 - 1 the cells' colours of the run, then
 * X, Y, Z, then the three stencil probes (blue r * 64). */
static void
ubo_fill_fs(struct ubo *u, unsigned run)
{
   for (unsigned k = 0; k < CELLS / 4; k++) {
      uint8_t c[4];
      cell_rgba(k, run, c);
      ubo_fs_rgba(u, k, c);
   }
   for (unsigned k = 0; k < 3; k++)
      ubo_fs_rgba(u, CELLS / 4 + k, row7_rgba[k]);
   for (unsigned r = 1; r <= 3; r++) {
      const uint8_t probe[4] = {0, 0, (uint8_t)(r * 64), 255};
      ubo_fs_rgba(u, CELLS / 4 + 3 + (r - 1), probe);
   }
}

/* ---------------------------------------------------------------------- */
/* The check                                                               */

static void
expected(uint32_t x, uint32_t y, unsigned run, uint8_t want[4])
{
   if (y < CELLS / 8 * CELL) {
      const unsigned i = (y / CELL) * 8 + x / CELL;
      cell_rgba(i / 4, run, want);
      want[2] = (uint8_t)(stencil_ref(i) * 64);
      return;
   }
   const unsigned k = x < 32 ? 2 : x < 48 ? 1 : 0;
   memcpy(want, row7_rgba[k], 4);
}

static bool
check(struct g52 *g, struct g52_image *color, unsigned run)
{
   g52_image_sync_for_read(g, color);
   uint32_t bad = 0, sent = 0;
   int fx = -1, fy = -1;
   uint8_t fgot[4] = {0}, fwant[4] = {0};
   for (uint32_t y = 0; y < IMG_H; y++) {
      for (uint32_t x = 0; x < IMG_W; x++) {
         const uint8_t *px = g52_image_pixel(color, x, y);
         uint8_t want[4];
         expected(x, y, run, want);
         bool ok = true;
         for (unsigned c = 0; c < 4; c++) {
            const int d = (int)px[c] - (int)want[c];
            ok &= d >= -1 && d <= 1;
         }
         if (ok)
            continue;
         if (px[0] == SENTINEL && px[1] == SENTINEL && px[2] == SENTINEL && px[3] == SENTINEL)
            sent++;
         if (!bad) {
            fx = (int)x;
            fy = (int)y;
            memcpy(fgot, px, 4);
            memcpy(fwant, want, 4);
         }
         bad++;
      }
   }
   if (!bad) {
      printf("  PASS run %u: all %u pixels as expected\n", run, IMG_W * IMG_H);
      return true;
   }
   printf("  FAIL run %u: %u pixels wrong (%u sentinel); first at (%d,%d) = %u %u %u %u, "
          "expected %u %u %u %u", run, bad, sent, fx, fy, fgot[0], fgot[1], fgot[2], fgot[3],
          fwant[0], fwant[1], fwant[2], fwant[3]);
   if (fy >= 0 && fy < CELLS / 8 * CELL) {
      const unsigned i = (unsigned)(fy / CELL) * 8 + (unsigned)fx / CELL;
      printf(" (cell %u: fragment slot %u, stencil reference %u)", i, i / 4, stencil_ref(i));
   }
   printf("\n");
   return false;
}

/* ---------------------------------------------------------------------- */
/* The set switch cases                                                    */

/* Colour 1 (buffer X, set A) and colour 2 (buffer Y, set B), at the bound
 * offset of their buffer. */
static const uint8_t sw_rgba[2][4] = {
   {220, 40, 20, 255},
   {20, 160, 240, 255},
};

/* What X and Y hold at offset 0 when the sets are bound at a non-zero one. */
static const uint8_t sw_decoy_rgba[2][4] = {
   {90, 220, 60, 255},
   {240, 200, 30, 255},
};

/* The left half (vertices 0 to 5) and the right half (6 to 11), two
 * triangles each. */
static const float sw_halves[24] = {
   -1.0f, -1.0f, 0.0f, -1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, -1.0f, 1.0f,
   0.0f, -1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
};

struct sw_case {
   VkDescriptorType type;     /* UNIFORM_BUFFER_DYNAMIC or STORAGE_BUFFER_DYNAMIC */
   VkShaderStageFlags stage;  /* the one stage the binding is visible to, which reads it */
   uint32_t offset;           /* the dynamic offset both sets are bound at */
};

/* What every case uses: the image, the command buffer, the quads. */
struct sw_ctx {
   struct g52 *g;
   const struct dyn_fns *d;
   struct g52_image *color;
   VkCommandPool pool;
   VkCommandBuffer cmd;
   VkBuffer halves;
};

/* What one case creates. */
struct sw_objs {
   struct g52_buffer x, y;
   VkDescriptorSetLayout dsl;
   VkDescriptorPool dpool;
   VkDescriptorSet sets[2];   /* A over X, B over Y */
   VkPipelineLayout layout;
   VkPipeline pipeline;
};

/* Every channel within one LSB. */
static bool
rgba_near(const uint8_t *a, const uint8_t *b)
{
   for (unsigned c = 0; c < 4; c++) {
      const int d = (int)a[c] - (int)b[c];
      if (d < -1 || d > 1)
         return false;
   }
   return true;
}

static void
rgba_to_float(const uint8_t rgba[4], float out[4])
{
   for (unsigned c = 0; c < 4; c++)
      out[c] = rgba[c] / 255.0f;
}

/* Buffer X (which 0) or Y (which 1): its colour at byte `offset` (a
 * multiple of 256), its decoy at offset 0 when `offset` is not 0. */
static bool
sw_buffer_create(struct g52 *g, struct g52_buffer *b, VkBufferUsageFlags usage, uint32_t offset,
                 unsigned which)
{
   const size_t size = (size_t)offset + 4 * sizeof(float);
   float *data = calloc(1, size);
   if (!data) {
      fprintf(stderr, "%s: out of host memory for a %zu-byte buffer\n", g->prog, size);
      return false;
   }
   rgba_to_float(sw_rgba[which], data + offset / sizeof(float));
   if (offset)
      rgba_to_float(sw_decoy_rgba[which], data);
   const bool ok = g52_buffer_create(g, b, data, size, usage);
   free(data);
   return ok;
}

static void
sw_objs_destroy(const struct sw_ctx *c, struct sw_objs *o)
{
   struct g52 *g = c->g;
   if (o->pipeline)
      g->f.DestroyPipeline(g->device, o->pipeline, NULL);
   if (o->layout)
      g->f.DestroyPipelineLayout(g->device, o->layout, NULL);
   if (o->dpool)
      c->d->DestroyDescriptorPool(g->device, o->dpool, NULL);
   if (o->dsl)
      c->d->DestroyDescriptorSetLayout(g->device, o->dsl, NULL);
   g52_buffer_destroy(g, &o->y);
   g52_buffer_destroy(g, &o->x);
   memset(o, 0, sizeof(*o));
}

/* Buffers X and Y, the set layout, sets A and B, the pipeline layout and
 * the pipeline of a case. False (after saying why) at the first failure,
 * with what was created so far in *o for sw_objs_destroy. */
static bool
sw_objs_create(const struct sw_ctx *c, const struct sw_case *sc, const char *name,
               struct sw_objs *o)
{
   struct g52 *g = c->g;
   const struct g52_fns *f = &g->f;
   const struct dyn_fns *d = c->d;
   const bool ubo = sc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
   const bool vs = sc->stage == VK_SHADER_STAGE_VERTEX_BIT;
   memset(o, 0, sizeof(*o));

   const VkBufferUsageFlags usage =
      ubo ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
   if (!sw_buffer_create(g, &o->x, usage, sc->offset, 0) ||
       !sw_buffer_create(g, &o->y, usage, sc->offset, 1))
      return false;

   /* Set 0: the one dynamic buffer binding, so no descriptor slots. */
   const VkDescriptorSetLayoutBinding binding = {0, sc->type, 1, sc->stage, NULL};
   const VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   VkResult r = d->CreateDescriptorSetLayout(g->device, &dslci, NULL, &o->dsl);
   const VkDescriptorPoolSize pool_size = {sc->type, 2};
   const VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
   };
   if (r == VK_SUCCESS)
      r = d->CreateDescriptorPool(g->device, &dpci, NULL, &o->dpool);
   const VkDescriptorSetLayout dsls[2] = {o->dsl, o->dsl};
   const VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = o->dpool,
      .descriptorSetCount = 2,
      .pSetLayouts = dsls,
   };
   if (r == VK_SUCCESS)
      r = d->AllocateDescriptorSets(g->device, &dsai, o->sets);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: %s: descriptor sets: %s\n", g->prog, name, g52_result_str(r));
      return false;
   }
   const VkDescriptorBufferInfo dbi[2] = {{o->x.buffer, 0, 16}, {o->y.buffer, 0, 16}};
   VkWriteDescriptorSet writes[2];
   for (unsigned s = 0; s < 2; s++) {
      writes[s] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = o->sets[s],
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = sc->type,
         .pBufferInfo = &dbi[s],
      };
   }
   d->UpdateDescriptorSets(g->device, 2, writes, 0, NULL);

   const VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &o->dsl,
   };
   r = f->CreatePipelineLayout(g->device, &plci, NULL, &o->layout);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: %s: vkCreatePipelineLayout failed: %s\n", g->prog, name,
              g52_result_str(r));
      return false;
   }

   /* The reading stage gets the buffer's shader, the other one the plain
    * one (sw.vert, or sw_flat.frag taking the vertex shader's colour). */
   const struct g52_pipeline_desc pd = {
      .vs_spv = !vs ? sw_vert_spv : ubo ? sw_ubo_vert_spv : sw_ssbo_vert_spv,
      .vs_size = !vs ? sizeof(sw_vert_spv) : ubo ? sizeof(sw_ubo_vert_spv)
                                                  : sizeof(sw_ssbo_vert_spv),
      .fs_spv = vs ? sw_flat_frag_spv : ubo ? sw_ubo_frag_spv : sw_ssbo_frag_spv,
      .fs_size = vs ? sizeof(sw_flat_frag_spv) : ubo ? sizeof(sw_ubo_frag_spv)
                                                    : sizeof(sw_ssbo_frag_spv),
      .attrib_format = VK_FORMAT_R32G32_SFLOAT,
      .vertex_stride = 2 * sizeof(float),
      .width = IMG_W,
      .height = IMG_H,
      .color_format = c->color->format,
      .depth_format = VK_FORMAT_UNDEFINED,
      .stencil_format = VK_FORMAT_UNDEFINED,
      .blend = {.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT},
   };
   o->pipeline = g52_pipeline_create(g, &pd, o->layout, name);
   return o->pipeline != VK_NULL_HANDLE;
}

/* The left half in colour 1, the right half in colour 2. */
static bool
sw_check(const struct sw_ctx *c, const struct sw_case *sc, const char *name)
{
   static const uint8_t black[4] = {0, 0, 0, 255};
   struct g52_image *color = c->color;
   g52_image_sync_for_read(c->g, color);
   uint32_t bad = 0, sent = 0, cleared = 0, stale = 0, unoffset = 0;
   int fx = -1, fy = -1;
   uint8_t fgot[4] = {0};
   for (uint32_t y = 0; y < IMG_H; y++) {
      for (uint32_t x = 0; x < IMG_W; x++) {
         const unsigned half = x >= SW_HALF;
         const uint8_t *px = g52_image_pixel(color, x, y);
         if (rgba_near(px, sw_rgba[half]))
            continue;
         if (px[0] == SENTINEL && px[1] == SENTINEL && px[2] == SENTINEL && px[3] == SENTINEL)
            sent++;
         else if (rgba_near(px, black))
            cleared++;
         else if (half && rgba_near(px, sw_rgba[0]))
            stale++;
         else if (sc->offset &&
                  (rgba_near(px, sw_decoy_rgba[0]) || rgba_near(px, sw_decoy_rgba[1])))
            unoffset++;
         if (!bad) {
            fx = (int)x;
            fy = (int)y;
            memcpy(fgot, px, 4);
         }
         bad++;
      }
   }
   if (!bad) {
      printf("  PASS %s: left half colour 1 (set A), right half colour 2 (set B)\n", name);
      return true;
   }
   const uint8_t *fwant = sw_rgba[fx >= SW_HALF];
   printf("  FAIL %s: %u pixels wrong (%u sentinel, %u black, %u of the right half in set A's "
          "colour 1, %u in a buffer's offset 0 decoy); first at (%d,%d) = %u %u %u %u, "
          "expected %u %u %u %u\n", name, bad, sent, cleared, stale, unoffset, fx, fy, fgot[0],
          fgot[1], fgot[2], fgot[3], fwant[0], fwant[1], fwant[2], fwant[3]);
   if (stale == (uint32_t)(IMG_W - SW_HALF) * IMG_H)
      printf("    (the whole right half shows colour 1: the draw after binding set B kept set "
             "A's descriptor tables or FAU uniform words, as when sets with no descriptor slots "
             "are matched by their address)\n");
   if (unoffset)
      printf("    (a buffer's offset 0 decoy: the dynamic offset was not applied)\n");
   return false;
}

/*
 * One case: bind set A, draw the left half, bind set B at the same offset,
 * draw the right half; submit and check. *stop is set when the submit did
 * not complete (a timeout or a lost device): no further case runs, and this
 * one's objects are left to the process exit since the GPU may still hold
 * them.
 */
static bool
sw_run(const struct sw_ctx *c, const struct sw_case *sc, bool *stop)
{
   struct g52 *g = c->g;
   const struct g52_fns *f = &g->f;
   const bool ubo = sc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
   const bool vs = sc->stage == VK_SHADER_STAGE_VERTEX_BIT;
   char name[112];
   snprintf(name, sizeof(name), "set switch, %s buffer read by the %s shader, both sets at "
            "offset %u", ubo ? "uniform" : "storage", vs ? "vertex" : "fragment", sc->offset);

   struct sw_objs o;
   if (!sw_objs_create(c, sc, name, &o)) {
      printf("  FAIL %s: setup failed (see stderr)\n", name);
      sw_objs_destroy(c, &o);
      return false;
   }

   VkResult r = f->ResetCommandPool(g->device, c->pool, 0);
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   if (r == VK_SUCCESS)
      r = f->BeginCommandBuffer(c->cmd, &bi);
   if (r != VK_SUCCESS) {
      printf("  FAIL %s: command buffer: %s\n", name, g52_result_str(r));
      sw_objs_destroy(c, &o);
      return false;
   }
   g52_image_refill(g, c->color, SENTINEL);
   g52_cmd_transition(g, c->cmd, c->color->image, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
   const VkRenderingAttachmentInfo color_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = c->color->view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = {0.0f, 0.0f, 0.0f, 1.0f},
   };
   const VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {IMG_W, IMG_H}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_att,
   };
   f->CmdBeginRendering(c->cmd, &ri);
   f->CmdBindPipeline(c->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, o.pipeline);
   const VkDeviceSize vb_offset = 0;
   f->CmdBindVertexBuffers(c->cmd, 0, 1, &c->halves, &vb_offset);
   const uint32_t offset = sc->offset;
   c->d->CmdBindDescriptorSets(c->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, o.layout, 0, 1,
                               &o.sets[0], 1, &offset);
   f->CmdDraw(c->cmd, 6, 1, 0, 0);
   c->d->CmdBindDescriptorSets(c->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, o.layout, 0, 1,
                               &o.sets[1], 1, &offset);
   f->CmdDraw(c->cmd, 6, 1, 6, 0);
   f->CmdEndRendering(c->cmd);
   g52_cmd_transition(g, c->cmd, c->color->image, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                      VK_ACCESS_HOST_READ_BIT);
   r = f->EndCommandBuffer(c->cmd);
   if (r != VK_SUCCESS) {
      printf("  FAIL %s: vkEndCommandBuffer: %s (see the log)\n", name, g52_result_str(r));
      sw_objs_destroy(c, &o);
      return false;
   }
   r = g52_submit_wait(g, c->cmd, name);
   if (r != VK_SUCCESS) {
      printf("  FAIL %s: the submit did not complete (%s); no further case runs\n", name,
             g52_result_str(r));
      *stop = true;
      return false;
   }
   const bool pass = sw_check(c, sc, name);
   sw_objs_destroy(c, &o);
   return pass;
}

/* ---------------------------------------------------------------------- */

int
main(int argc, char **argv)
{
   struct g52 g;
   int rc = g52_open(&g, "dyn_g52", argc, argv, DYN_G52_DEFAULT_ICD);
   if (rc)
      return rc;
   const struct g52_fns *f = &g.f;
   struct dyn_fns d;
   bool ok = true;
#define DYN(name) d.name = (PFN_vk##name)g52_load_device_fn(&g, "vk" #name, NULL, true, &ok)
   DYN(CreateDescriptorSetLayout);
   DYN(DestroyDescriptorSetLayout);
   DYN(CreateDescriptorPool);
   DYN(DestroyDescriptorPool);
   DYN(AllocateDescriptorSets);
   DYN(UpdateDescriptorSets);
   DYN(CmdBindDescriptorSets);
   DYN(CmdSetStencilReference);
#undef DYN
   if (!ok)
      return 1;

   VkPhysicalDeviceProperties2 props2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
   f->GetPhysicalDeviceProperties2(g.pdev, &props2);
   VkDeviceSize align = props2.properties.limits.minUniformBufferOffsetAlignment;
   if (align < 256)
      align = 256;
   /* The set switch cases' non-zero offset, valid for both buffer types. */
   VkDeviceSize sw_offset = align;
   if (props2.properties.limits.minStorageBufferOffsetAlignment > sw_offset)
      sw_offset = props2.properties.limits.minStorageBufferOffsetAlignment;

   static const VkFormat zs_candidates[] = {VK_FORMAT_D24_UNORM_S8_UINT,
                                            VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_S8_UINT};
   VkFormat zs_format = VK_FORMAT_UNDEFINED;
   for (unsigned i = 0; i < sizeof(zs_candidates) / sizeof(zs_candidates[0]); i++) {
      VkFormatProperties fp;
      f->GetPhysicalDeviceFormatProperties(g.pdev, zs_candidates[i], &fp);
      if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
         zs_format = zs_candidates[i];
         break;
      }
   }
   if (zs_format == VK_FORMAT_UNDEFINED) {
      printf("FAIL: no stencil format is offered as a depth/stencil attachment\n");
      return 1;
   }
   const VkImageAspectFlags zs_aspects =
      zs_format == VK_FORMAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT
                                     : VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
   const VkFormat zs_depth = zs_format == VK_FORMAT_S8_UINT ? VK_FORMAT_UNDEFINED : zs_format;

   struct g52_image color, zs;
   if (!g52_image_create(&g, &color, VK_FORMAT_R8G8B8A8_UNORM, IMG_W, IMG_H,
                         VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, true, SENTINEL) ||
       !g52_image_create(&g, &zs, zs_format, IMG_W, IMG_H, VK_IMAGE_TILING_OPTIMAL,
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, zs_aspects, false, 0))
      return 1;

   static const float quad[12] = {0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
   struct g52_buffer vbuf;
   if (!g52_buffer_create(&g, &vbuf, quad, sizeof(quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT))
      return 1;
   struct ubo u;
   if (!ubo_create(&g, &u, align))
      return 1;
   const float cell = 2.0f * CELL / IMG_W;
   for (unsigned i = 0; i < CELLS; i++)
      ubo_vs_rect(&u, i, -1.0f + cell * (i % 8), -1.0f + cell * (i / 8), cell, cell);
   const float row7 = -1.0f + cell * 7;
   ubo_vs_rect(&u, CELLS + 0, -1.0f, row7, 2.0f, cell);
   ubo_vs_rect(&u, CELLS + 1, -1.0f, row7, 1.5f, cell);
   ubo_vs_rect(&u, CELLS + 2, -1.0f, row7, 1.0f, cell);
   ubo_vs_rect(&u, CELLS + 3, -1.0f, -1.0f, 2.0f, 2.0f);
   ubo_fill_fs(&u, 0);
   ubo_flush(&g, &u);

   /* One set layout, two sets over the same buffer. */
   const VkDescriptorSetLayoutBinding bindings[2] = {
      {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL},
   };
   const VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
   VkResult r = d.CreateDescriptorSetLayout(g.device, &dslci, NULL, &dsl);
   const VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4};
   const VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
   };
   VkDescriptorPool dpool = VK_NULL_HANDLE;
   if (r == VK_SUCCESS)
      r = d.CreateDescriptorPool(g.device, &dpci, NULL, &dpool);
   VkDescriptorSet sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
   const VkDescriptorSetLayout dsls[2] = {dsl, dsl};
   const VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool,
      .descriptorSetCount = 2,
      .pSetLayouts = dsls,
   };
   if (r == VK_SUCCESS)
      r = d.AllocateDescriptorSets(g.device, &dsai, sets);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "dyn_g52: descriptor sets: %s\n", g52_result_str(r));
      return 1;
   }
   const VkDescriptorBufferInfo dbi = {u.buffer, 0, 16};
   VkWriteDescriptorSet writes[4];
   for (unsigned s = 0; s < 2; s++) {
      for (unsigned b = 0; b < 2; b++) {
         writes[s * 2 + b] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sets[s],
            .dstBinding = b,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pBufferInfo = &dbi,
         };
      }
   }
   d.UpdateDescriptorSets(g.device, 4, writes, 0, NULL);

   const VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &dsl,
   };
   VkPipelineLayout layout = VK_NULL_HANDLE;
   r = f->CreatePipelineLayout(g.device, &plci, NULL, &layout);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "dyn_g52: vkCreatePipelineLayout failed: %s\n", g52_result_str(r));
      return 1;
   }

   /* Pipelines: the cells (two objects, same state), row 7, the probes. */
   const VkStencilOpState st_write = {
      .failOp = VK_STENCIL_OP_KEEP,
      .passOp = VK_STENCIL_OP_REPLACE,
      .depthFailOp = VK_STENCIL_OP_KEEP,
      .compareOp = VK_COMPARE_OP_ALWAYS,
      .compareMask = 0xff,
      .writeMask = 0xff,
   };
   const VkPipelineDepthStencilStateCreateInfo ds_write = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .stencilTestEnable = VK_TRUE,
      .front = st_write,
      .back = st_write,
   };
   const VkPipelineDepthStencilStateCreateInfo ds_off = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
   };
   const VkStencilOpState st_test = {
      .failOp = VK_STENCIL_OP_KEEP,
      .passOp = VK_STENCIL_OP_KEEP,
      .depthFailOp = VK_STENCIL_OP_KEEP,
      .compareOp = VK_COMPARE_OP_EQUAL,
      .compareMask = 0xff,
      .writeMask = 0,
   };
   const VkPipelineDepthStencilStateCreateInfo ds_test = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .stencilTestEnable = VK_TRUE,
      .front = st_test,
      .back = st_test,
   };
   const VkColorComponentFlags rg = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
   const VkColorComponentFlags rgba = rg | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
   const VkFormat cf = color.format;
   VkPipeline p_cell_a = make_pipeline(&g, layout, cf, zs_format, &ds_write, rg,
                                       "cells A (stencil REPLACE, dynamic reference)");
   VkPipeline p_cell_b = make_pipeline(&g, layout, cf, zs_format, &ds_write, rg,
                                       "cells B (the same state)");
   VkPipeline p_row = make_pipeline(&g, layout, cf, zs_format, &ds_off, rgba, "row 7 (no stencil)");
   VkPipeline p_probe = make_pipeline(&g, layout, cf, zs_format, &ds_test, VK_COLOR_COMPONENT_B_BIT,
                                      "probes (stencil EQUAL, blue)");
   if (!p_cell_a || !p_cell_b || !p_row || !p_probe)
      return 1;

   const VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkCommandPool pool = VK_NULL_HANDLE;
   r = f->CreateCommandPool(g.device, &cpci, NULL, &pool);
   const VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd = VK_NULL_HANDLE;
   if (r == VK_SUCCESS)
      r = f->AllocateCommandBuffers(g.device, &cbai, &cmd);
   const VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   if (r == VK_SUCCESS)
      r = f->BeginCommandBuffer(cmd, &bi);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "dyn_g52: command buffer: %s\n", g52_result_str(r));
      return 1;
   }

   g52_cmd_transition(&g, cmd, color.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
   g52_cmd_transition(&g, cmd, zs.image, zs_aspects, VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
   const VkRenderingAttachmentInfo color_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = color.view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = {0.0f, 0.0f, 0.0f, 1.0f},
   };
   const VkRenderingAttachmentInfo zs_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = zs.view,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .clearValue.depthStencil = {1.0f, 0},
   };
   const VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {IMG_W, IMG_H}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_att,
      .pDepthAttachment = zs_depth != VK_FORMAT_UNDEFINED ? &zs_att : NULL,
      .pStencilAttachment = &zs_att,
   };
   f->CmdBeginRendering(cmd, &ri);
   const VkDeviceSize vb_offset = 0;
   f->CmdBindVertexBuffers(cmd, 0, 1, &vbuf.buffer, &vb_offset);

   /* 1. The cells. */
   for (unsigned i = 0; i < CELLS; i++) {
      f->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, i % 5 == 0 ? p_cell_b : p_cell_a);
      const uint32_t offs[2] = {(uint32_t)(i * align), (uint32_t)(u.fs_base + (i / 4) * align)};
      const VkDescriptorSet set = sets[i % 7 == 3 ? 1 : 0];
      d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 2, offs);
      if (i % 8 == 7)
         d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 2, offs);
      d.CmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_ref(i));
      f->CmdDraw(cmd, 6, 1, 0, 0);
   }
   /* 2. Row 7, last draw wins. */
   f->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p_row);
   d.CmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
   for (unsigned k = 0; k < 3; k++) {
      const uint32_t offs[2] = {(uint32_t)((CELLS + k) * align),
                                (uint32_t)(u.fs_base + (CELLS / 4 + k) * align)};
      d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &sets[0], 2,
                              offs);
      f->CmdDraw(cmd, 6, 1, 0, 0);
   }
   /* 3. The stencil probes. */
   f->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p_probe);
   for (unsigned ref = 1; ref <= 3; ref++) {
      const uint32_t offs[2] = {(uint32_t)((CELLS + 3) * align),
                                (uint32_t)(u.fs_base + (CELLS / 4 + 3 + ref - 1) * align)};
      d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &sets[0], 2,
                              offs);
      d.CmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, ref);
      f->CmdDraw(cmd, 6, 1, 0, 0);
   }
   f->CmdEndRendering(cmd);
   g52_cmd_transition(&g, cmd, color.image, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                      VK_ACCESS_HOST_READ_BIT);
   r = f->EndCommandBuffer(cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "dyn_g52: vkEndCommandBuffer failed: %s\n", g52_result_str(r));
      return 1;
   }
   printf("  command buffer recorded: %u draws in one pass\n", CELLS + 6);

   bool pass = true;
   for (unsigned run = 0; run < 2 && pass; run++) {
      if (run) {
         /* New fragment colours for the second run of the same commands. */
         ubo_fill_fs(&u, 1);
         ubo_flush(&g, &u);
         g52_image_refill(&g, &color, SENTINEL);
      }
      r = g52_submit_wait(&g, cmd, run ? "(second run)" : "(first run)");
      if (r != VK_SUCCESS) {
         pass = false;
         break;
      }
      pass = check(&g, &color, run);
   }

   /* 4. The set switch cases, each its own pass into the same image, as
    * long as the device completes the work. */
   unsigned sw_passed = 0, sw_ran = 0;
   struct g52_buffer halves = {0};
   if (r != VK_SUCCESS) {
      printf("  set switch cases not run: the draws above did not complete\n");
   } else if (!g52_buffer_create(&g, &halves, sw_halves, sizeof(sw_halves),
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
      printf("  set switch cases not run: no vertex buffer\n");
   } else {
      printf("  set switch cases: set 0 holds one dynamic buffer binding and no descriptor "
             "slots; set A (colour 1) then set B (colour 2) at one offset\n");
      const struct sw_ctx swc = {&g, &d, &color, pool, cmd, halves.buffer};
      bool stop = false;
      for (unsigned i = 0; i < SW_CASES && !stop; i++) {
         const struct sw_case sc = {
            .type = (i & 4) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                            : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .stage = (i & 2) ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = (i & 1) ? (uint32_t)sw_offset : 0,
         };
         sw_ran++;
         if (sw_run(&swc, &sc, &stop))
            sw_passed++;
      }
   }

   const bool all = pass && sw_passed == SW_CASES;
   if (all)
      printf("PASS: dynamic stencil reference, rebound uniform offsets and draw order over %u "
             "draws, two runs; %u of %u set switch cases\n", (unsigned)(CELLS + 6), sw_passed,
             SW_CASES);
   else
      printf("FAIL: see above (%u draws: %s; set switch cases: %u of %u passed, %u not run)\n",
             (unsigned)(CELLS + 6), pass ? "PASS" : "FAIL", sw_passed, SW_CASES,
             SW_CASES - sw_ran);

   f->DeviceWaitIdle(g.device);
   f->DestroyCommandPool(g.device, pool, NULL);
   f->DestroyPipeline(g.device, p_cell_a, NULL);
   f->DestroyPipeline(g.device, p_cell_b, NULL);
   f->DestroyPipeline(g.device, p_row, NULL);
   f->DestroyPipeline(g.device, p_probe, NULL);
   f->DestroyPipelineLayout(g.device, layout, NULL);
   d.DestroyDescriptorPool(g.device, dpool, NULL);
   d.DestroyDescriptorSetLayout(g.device, dsl, NULL);
   f->UnmapMemory(g.device, u.memory);
   f->DestroyBuffer(g.device, u.buffer, NULL);
   f->FreeMemory(g.device, u.memory, NULL);
   g52_buffer_destroy(&g, &halves);
   g52_buffer_destroy(&g, &vbuf);
   g52_image_destroy(&g, &zs);
   g52_image_destroy(&g, &color);
   g52_close(&g);
   return all ? 0 : 1;
}
