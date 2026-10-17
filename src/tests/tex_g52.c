/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * tex_g52: the first texture sampled on the Bifrost (v7) path of this
 * driver. After draw_g52 proved the Vertex + Tiler job pair, this one
 * proves the descriptor set model: a 4x4 RGBA8 texture with a known
 * pattern is uploaded through a staging buffer (vkCmdCopyBufferToImage),
 * bound as a combined image sampler next to a uniform buffer, and sampled
 * with a nearest sampler by a full-screen triangle onto a 64x64 linear
 * target, so each texel covers a 16x16 block of pixels; every pixel is
 * read back and checked against the texel it should show, transformed by
 * the uniform buffer's scale and bias. That exercises the Bifrost Texture
 * descriptor and its surface payload, the Sampler descriptor, the Uniform
 * Buffer descriptor, the per-stage tables the Draw points at, the
 * descriptor lowering's indices and the set binding.
 *
 * Texel (x, y) holds (16 + 64x, 16 + 64y, 8 + 16(x + 4y), 255); the
 * fragment shader computes texel * scale + bias with scale (0.5, 0.5,
 * 0.5, 1) and bias (0.25, 0.25, 0.25, 0) from the uniform buffer, so
 * pixel (px, py) must read (t.r / 2 + 64, t.g / 2 + 64, t.b / 2 + 64, 255)
 * for t = texel (px / 16, py / 16). The two failure shapes are told
 * apart: a black pixel means the uniform buffer read as zero (scale 0),
 * a uniform 64 64 64 255 means the texture read as zero.
 *
 * The upload: vkCmdCopyBufferToImage goes through the driver's compute
 * copy (a v7 dispatch since the compute merge). Only on a driver that
 * refuses the copy outright (recording fails with
 * VK_ERROR_FEATURE_NOT_PRESENT, a TODO v7 stub) are the texels written into
 * the linear image's memory from the host instead, and the verdict says
 * so; any other recording failure is a FAIL of the copy path.
 * --upload=copy makes the refusal a failure too, --upload=host skips the
 * copy. --tiled makes the texture optimally tiled (u-interleaved on v7),
 * which needs the copy path.
 *
 * The shaders are SPIR-V embedded as arrays, generated with glslangValidator
 * 16.6.0 (`glslangValidator -V --target-env vulkan1.1`) from:
 *
 *   tex.vert:  #version 450
 *              layout(location = 0) in vec2 pos;
 *              layout(location = 0) out vec2 uv;
 *              void main() {
 *                 gl_Position = vec4(pos, 0.0, 1.0);
 *                 uv = pos * 0.5 + 0.5;
 *              }
 *
 *   tex.frag:  #version 450
 *              layout(set = 0, binding = 0) uniform sampler2D tex;
 *              layout(set = 0, binding = 1) uniform Params {
 *                 vec4 scale;
 *                 vec4 bias;
 *              } params;
 *              layout(location = 0) in vec2 uv;
 *              layout(location = 0) out vec4 color;
 *              void main() {
 *                 color = texture(tex, uv) * params.scale + params.bias;
 *              }
 *
 * The fragment shader also reads a dynamic uniform buffer (binding 2) at
 * offset 1136, past the first kilobyte, the way azahar's fragment shaders
 * read their PICA constants: the image is drawn as two rectangles, the top
 * half with dynamic offset 0 (add = 0) and the bottom half with the same
 * set bound again at offset DYN_STRIDE (add = (0.125, 0, 0.0625, 0)), so
 * the bottom half must read 32 and 16 more in red and blue. That covers
 * the uniform buffer words a Bifrost fragment shader takes from FAU and
 * the descriptor tables kept across a rebind with new dynamic offsets. The
 * vertex shader likewise reads its uv scale and bias from a dynamic uniform
 * buffer (binding 3) at offset 1488, k = (1, 1, 1, 1) in both of its
 * blocks, so a word read wrong moves the texture:
 *
 *   binding 3: layout(set = 0, binding = 3) uniform VDyn {
 *                 vec4 pad[93];
 *                 vec4 k;
 *              } vdyn;
 *              uv = pos * 0.5 * vdyn.k.xy + 0.5 * vdyn.k.zw;
 *
 *
 *   binding 2: layout(set = 0, binding = 2) uniform Dyn {
 *                 vec4 pad[71];
 *                 vec4 add;
 *              } dyn;
 *              color = texture(tex, uv) * params.scale + params.bias + dyn.add;
 *
 * No Vulkan loader: the driver library is dlopen'ed (see clear_g52.c).
 *
 * Usage: tex_g52 [path/to/libvulkan_malisx2.so] [--tiled] [--upload=copy|host|auto]
 *   The path defaults to $MALISX2_ICD, then to TEX_G52_DEFAULT_ICD.
 *
 * Exit status: 0 PASS; 1 FAIL (wrong pixels, a timeout or a Vulkan error
 * once the device exists); 2 the program did not get as far as a
 * VkDevice.
 *
 * Driver log: MESA_LOG=file sends it to stderr, MESA_LOG_LEVEL=info adds
 * the atom layout and build lines, MALISX2_DUMP_DRAW=1 the per draw
 * address dump with the descriptor tables (see MALISX2_G52.md).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* clock_gettime under -std=c11 */
#endif

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef TEX_G52_DEFAULT_ICD
#define TEX_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define IMG_W 64
#define IMG_H 64
#define TEX_W 4
#define TEX_H 4
#define BLOCK (IMG_W / TEX_W)
#define WAIT_NS (5ull * 1000 * 1000 * 1000)
#define SENTINEL 0xa5

/* The uniform buffer: scale, bias. */
static const float params[8] = {0.5f, 0.5f, 0.5f, 1.0f, 0.25f, 0.25f, 0.25f, 0.0f};
/* The clear colour under the triangle: black, so a pixel the triangle
 * missed is told apart from one never written (the sentinel). */
static const float clear_rgba[4] = {0.0f, 0.0f, 0.0f, 1.0f};

/* ---------------------------------------------------------------------- */
/* Shaders (see the file comment)                                          */

static const uint32_t vert_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000030, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0008000f, 0x00000000,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000012, 0x0000001c,
   0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
   0x00000000, 0x00060005, 0x0000000b, 0x505f6c67, 0x65567265, 0x78657472,
   0x00000000, 0x00060006, 0x0000000b, 0x00000000, 0x505f6c67, 0x7469736f,
   0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f,
   0x657a6953, 0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67,
   0x4470696c, 0x61747369, 0x0065636e, 0x00070006, 0x0000000b, 0x00000003,
   0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x0000000d,
   0x00000000, 0x00030005, 0x00000012, 0x00736f70, 0x00030005, 0x0000001c,
   0x00007675, 0x00040005, 0x00000022, 0x6e794456, 0x00000000, 0x00040006,
   0x00000022, 0x00000000, 0x00646170, 0x00040006, 0x00000022, 0x00000001,
   0x0000006b, 0x00040005, 0x00000024, 0x6e796476, 0x00000000, 0x00030047,
   0x0000000b, 0x00000002, 0x00050048, 0x0000000b, 0x00000000, 0x0000000b,
   0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x0000000b, 0x00000001,
   0x00050048, 0x0000000b, 0x00000002, 0x0000000b, 0x00000003, 0x00050048,
   0x0000000b, 0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x00000012,
   0x0000001e, 0x00000000, 0x00040047, 0x0000001c, 0x0000001e, 0x00000000,
   0x00040047, 0x00000021, 0x00000006, 0x00000010, 0x00030047, 0x00000022,
   0x00000002, 0x00050048, 0x00000022, 0x00000000, 0x00000023, 0x00000000,
   0x00050048, 0x00000022, 0x00000001, 0x00000023, 0x000005d0, 0x00040047,
   0x00000024, 0x00000021, 0x00000003, 0x00040047, 0x00000024, 0x00000022,
   0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
   0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006,
   0x00000004, 0x00040015, 0x00000008, 0x00000020, 0x00000000, 0x0004002b,
   0x00000008, 0x00000009, 0x00000001, 0x0004001c, 0x0000000a, 0x00000006,
   0x00000009, 0x0006001e, 0x0000000b, 0x00000007, 0x00000006, 0x0000000a,
   0x0000000a, 0x00040020, 0x0000000c, 0x00000003, 0x0000000b, 0x0004003b,
   0x0000000c, 0x0000000d, 0x00000003, 0x00040015, 0x0000000e, 0x00000020,
   0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040017,
   0x00000010, 0x00000006, 0x00000002, 0x00040020, 0x00000011, 0x00000001,
   0x00000010, 0x0004003b, 0x00000011, 0x00000012, 0x00000001, 0x0004002b,
   0x00000006, 0x00000014, 0x00000000, 0x0004002b, 0x00000006, 0x00000015,
   0x3f800000, 0x00040020, 0x00000019, 0x00000003, 0x00000007, 0x00040020,
   0x0000001b, 0x00000003, 0x00000010, 0x0004003b, 0x0000001b, 0x0000001c,
   0x00000003, 0x0004002b, 0x00000006, 0x0000001e, 0x3f000000, 0x0004002b,
   0x00000008, 0x00000020, 0x0000005d, 0x0004001c, 0x00000021, 0x00000007,
   0x00000020, 0x0004001e, 0x00000022, 0x00000021, 0x00000007, 0x00040020,
   0x00000023, 0x00000002, 0x00000022, 0x0004003b, 0x00000023, 0x00000024,
   0x00000002, 0x0004002b, 0x0000000e, 0x00000025, 0x00000001, 0x00040020,
   0x00000026, 0x00000002, 0x00000007, 0x00050036, 0x00000002, 0x00000004,
   0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000010,
   0x00000013, 0x00000012, 0x00050051, 0x00000006, 0x00000016, 0x00000013,
   0x00000000, 0x00050051, 0x00000006, 0x00000017, 0x00000013, 0x00000001,
   0x00070050, 0x00000007, 0x00000018, 0x00000016, 0x00000017, 0x00000014,
   0x00000015, 0x00050041, 0x00000019, 0x0000001a, 0x0000000d, 0x0000000f,
   0x0003003e, 0x0000001a, 0x00000018, 0x0004003d, 0x00000010, 0x0000001d,
   0x00000012, 0x0005008e, 0x00000010, 0x0000001f, 0x0000001d, 0x0000001e,
   0x00050041, 0x00000026, 0x00000027, 0x00000024, 0x00000025, 0x0004003d,
   0x00000007, 0x00000028, 0x00000027, 0x0007004f, 0x00000010, 0x00000029,
   0x00000028, 0x00000028, 0x00000000, 0x00000001, 0x00050085, 0x00000010,
   0x0000002a, 0x0000001f, 0x00000029, 0x00050041, 0x00000026, 0x0000002b,
   0x00000024, 0x00000025, 0x0004003d, 0x00000007, 0x0000002c, 0x0000002b,
   0x0007004f, 0x00000010, 0x0000002d, 0x0000002c, 0x0000002c, 0x00000002,
   0x00000003, 0x0005008e, 0x00000010, 0x0000002e, 0x0000002d, 0x0000001e,
   0x00050081, 0x00000010, 0x0000002f, 0x0000002a, 0x0000002e, 0x0003003e,
   0x0000001c, 0x0000002f, 0x000100fd, 0x00010038,
};
static const uint32_t frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x0000002a, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00000011, 0x00030010,
   0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
   0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63,
   0x00000072, 0x00030005, 0x0000000d, 0x00786574, 0x00030005, 0x00000011,
   0x00007675, 0x00040005, 0x00000014, 0x61726150, 0x0000736d, 0x00050006,
   0x00000014, 0x00000000, 0x6c616373, 0x00000065, 0x00050006, 0x00000014,
   0x00000001, 0x73616962, 0x00000000, 0x00040005, 0x00000016, 0x61726170,
   0x0000736d, 0x00030005, 0x00000024, 0x006e7944, 0x00040006, 0x00000024,
   0x00000000, 0x00646170, 0x00040006, 0x00000024, 0x00000001, 0x00646461,
   0x00030005, 0x00000026, 0x006e7964, 0x00040047, 0x00000009, 0x0000001e,
   0x00000000, 0x00040047, 0x0000000d, 0x00000021, 0x00000000, 0x00040047,
   0x0000000d, 0x00000022, 0x00000000, 0x00040047, 0x00000011, 0x0000001e,
   0x00000000, 0x00030047, 0x00000014, 0x00000002, 0x00050048, 0x00000014,
   0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000014, 0x00000001,
   0x00000023, 0x00000010, 0x00040047, 0x00000016, 0x00000021, 0x00000001,
   0x00040047, 0x00000016, 0x00000022, 0x00000000, 0x00040047, 0x00000023,
   0x00000006, 0x00000010, 0x00030047, 0x00000024, 0x00000002, 0x00050048,
   0x00000024, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000024,
   0x00000001, 0x00000023, 0x00000470, 0x00040047, 0x00000026, 0x00000021,
   0x00000002, 0x00040047, 0x00000026, 0x00000022, 0x00000000, 0x00020013,
   0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
   0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020,
   0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009,
   0x00000003, 0x00090019, 0x0000000a, 0x00000006, 0x00000001, 0x00000000,
   0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x0003001b, 0x0000000b,
   0x0000000a, 0x00040020, 0x0000000c, 0x00000000, 0x0000000b, 0x0004003b,
   0x0000000c, 0x0000000d, 0x00000000, 0x00040017, 0x0000000f, 0x00000006,
   0x00000002, 0x00040020, 0x00000010, 0x00000001, 0x0000000f, 0x0004003b,
   0x00000010, 0x00000011, 0x00000001, 0x0004001e, 0x00000014, 0x00000007,
   0x00000007, 0x00040020, 0x00000015, 0x00000002, 0x00000014, 0x0004003b,
   0x00000015, 0x00000016, 0x00000002, 0x00040015, 0x00000017, 0x00000020,
   0x00000001, 0x0004002b, 0x00000017, 0x00000018, 0x00000000, 0x00040020,
   0x00000019, 0x00000002, 0x00000007, 0x0004002b, 0x00000017, 0x0000001d,
   0x00000001, 0x00040015, 0x00000021, 0x00000020, 0x00000000, 0x0004002b,
   0x00000021, 0x00000022, 0x00000047, 0x0004001c, 0x00000023, 0x00000007,
   0x00000022, 0x0004001e, 0x00000024, 0x00000023, 0x00000007, 0x00040020,
   0x00000025, 0x00000002, 0x00000024, 0x0004003b, 0x00000025, 0x00000026,
   0x00000002, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
   0x000200f8, 0x00000005, 0x0004003d, 0x0000000b, 0x0000000e, 0x0000000d,
   0x0004003d, 0x0000000f, 0x00000012, 0x00000011, 0x00050057, 0x00000007,
   0x00000013, 0x0000000e, 0x00000012, 0x00050041, 0x00000019, 0x0000001a,
   0x00000016, 0x00000018, 0x0004003d, 0x00000007, 0x0000001b, 0x0000001a,
   0x00050085, 0x00000007, 0x0000001c, 0x00000013, 0x0000001b, 0x00050041,
   0x00000019, 0x0000001e, 0x00000016, 0x0000001d, 0x0004003d, 0x00000007,
   0x0000001f, 0x0000001e, 0x00050081, 0x00000007, 0x00000020, 0x0000001c,
   0x0000001f, 0x00050041, 0x00000019, 0x00000027, 0x00000026, 0x0000001d,
   0x0004003d, 0x00000007, 0x00000028, 0x00000027, 0x00050081, 0x00000007,
   0x00000029, 0x00000020, 0x00000028, 0x0003003e, 0x00000009, 0x00000029,
   0x000100fd, 0x00010038,
};

/* Two rectangles of two triangles each: the top half of clip space (drawn
 * with dynamic offset 0), then the bottom half (dynamic offset
 * DYN_STRIDE). */
static const float vertices[24] = {
   -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 0.0f,  -1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f,
   -1.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  -1.0f, 0.0f,  1.0f, 1.0f, -1.0f, 1.0f,
};

/* The dynamic uniform buffer: two blocks DYN_STRIDE apart, each with its
 * vec4 add at DYN_ADD_OFFSET (the std140 offset of dyn.add). */
#define DYN_STRIDE 2048
#define DYN_ADD_OFFSET 1136
#define DYN_BLOCK_SIZE (DYN_ADD_OFFSET + 16)
static const float dyn_add[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.125f, 0.0f, 0.0625f, 0.0f}};
/* The vertex shader's dynamic uniform buffer: vdyn.k at VDYN_K_OFFSET of
 * each of its two blocks. */
#define VDYN_K_OFFSET 1488
#define VDYN_BLOCK_SIZE (VDYN_K_OFFSET + 16)
static const float vdyn_k[4] = {1.0f, 1.0f, 1.0f, 1.0f};

/* ---------------------------------------------------------------------- */
/* Helpers                                                                 */

static void
texel(unsigned x, unsigned y, uint8_t out[4])
{
   out[0] = 16 + 64 * x;
   out[1] = 16 + 64 * y;
   out[2] = 8 + 16 * (x + 4 * y);
   out[3] = 255;
}

/* What a pixel showing texel t in half `half` (0 top, 1 bottom) must read:
 * t * scale + bias + that half's add, rounded. */
static void
expected_pixel(const uint8_t t[4], unsigned half, uint8_t out[4])
{
   for (unsigned c = 0; c < 4; c++) {
      const float v = (t[c] / 255.0f) * params[c] + params[4 + c] + dyn_add[half][c];
      out[c] = (uint8_t)(v * 255.0f + 0.5f);
   }
}

static const char *
result_str(VkResult r)
{
   switch (r) {
   case VK_SUCCESS: return "VK_SUCCESS";
   case VK_NOT_READY: return "VK_NOT_READY";
   case VK_TIMEOUT: return "VK_TIMEOUT";
   case VK_INCOMPLETE: return "VK_INCOMPLETE";
   case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
   case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
   case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
   case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
   case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
   case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
   case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
   case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
   case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
   case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
   case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
   case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
   case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
   default: return "VkResult(?)";
   }
}

static int64_t
now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

struct fns {
   PFN_vkGetInstanceProcAddr gipa;
   PFN_vkCreateInstance CreateInstance;
   PFN_vkDestroyInstance DestroyInstance;
   PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
   PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
   PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
   PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
   PFN_vkCreateDevice CreateDevice;
   PFN_vkGetDeviceProcAddr GetDeviceProcAddr;

   PFN_vkDestroyDevice DestroyDevice;
   PFN_vkGetDeviceQueue GetDeviceQueue;
   PFN_vkCreateImage CreateImage;
   PFN_vkDestroyImage DestroyImage;
   PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
   PFN_vkGetImageSubresourceLayout GetImageSubresourceLayout;
   PFN_vkCreateBuffer CreateBuffer;
   PFN_vkDestroyBuffer DestroyBuffer;
   PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
   PFN_vkBindBufferMemory BindBufferMemory;
   PFN_vkAllocateMemory AllocateMemory;
   PFN_vkFreeMemory FreeMemory;
   PFN_vkBindImageMemory BindImageMemory;
   PFN_vkMapMemory MapMemory;
   PFN_vkUnmapMemory UnmapMemory;
   PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges;
   PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges;
   PFN_vkCreateImageView CreateImageView;
   PFN_vkDestroyImageView DestroyImageView;
   PFN_vkCreateSampler CreateSampler;
   PFN_vkDestroySampler DestroySampler;
   PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
   PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
   PFN_vkCreateDescriptorPool CreateDescriptorPool;
   PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
   PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
   PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
   PFN_vkCreateShaderModule CreateShaderModule;
   PFN_vkDestroyShaderModule DestroyShaderModule;
   PFN_vkCreatePipelineLayout CreatePipelineLayout;
   PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
   PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines;
   PFN_vkDestroyPipeline DestroyPipeline;
   PFN_vkCreateCommandPool CreateCommandPool;
   PFN_vkDestroyCommandPool DestroyCommandPool;
   PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
   PFN_vkBeginCommandBuffer BeginCommandBuffer;
   PFN_vkEndCommandBuffer EndCommandBuffer;
   PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
   PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;
   PFN_vkCmdBeginRendering CmdBeginRendering;
   PFN_vkCmdEndRendering CmdEndRendering;
   PFN_vkCmdBindPipeline CmdBindPipeline;
   PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
   PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers;
   PFN_vkCmdDraw CmdDraw;
   PFN_vkCreateFence CreateFence;
   PFN_vkDestroyFence DestroyFence;
   PFN_vkResetFences ResetFences;
   PFN_vkQueueSubmit QueueSubmit;
   PFN_vkWaitForFences WaitForFences;
   PFN_vkDeviceWaitIdle DeviceWaitIdle;
   PFN_vkGetDeviceFaultInfoEXT GetDeviceFaultInfoEXT; /* optional */
};

static PFN_vkVoidFunction
load_instance_fn(const struct fns *f, VkInstance instance, const char *name, bool *ok)
{
   PFN_vkVoidFunction p = f->gipa(instance, name);
   if (!p) {
      fprintf(stderr, "tex_g52: vk_icdGetInstanceProcAddr(%s) returned NULL\n", name);
      *ok = false;
   }
   return p;
}

static PFN_vkVoidFunction
load_device_fn(const struct fns *f, VkDevice device, const char *name, const char *alt,
               bool required, bool *ok)
{
   PFN_vkVoidFunction p = f->GetDeviceProcAddr(device, name);
   if (!p && alt)
      p = f->GetDeviceProcAddr(device, alt);
   if (!p && required) {
      fprintf(stderr, "tex_g52: vkGetDeviceProcAddr(%s%s%s) returned NULL\n", name,
              alt ? " / " : "", alt ? alt : "");
      *ok = false;
   }
   return p;
}

static void
print_fault(const struct fns *f, VkDevice device)
{
   if (!f->GetDeviceFaultInfoEXT) {
      printf("  (VK_EXT_device_fault not enabled; no fault record)\n");
      return;
   }
   VkDeviceFaultCountsEXT counts = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
   VkResult r = f->GetDeviceFaultInfoEXT(device, &counts, NULL);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      printf("  vkGetDeviceFaultInfoEXT: %s\n", result_str(r));
      return;
   }
   VkDeviceFaultAddressInfoEXT addr[4];
   VkDeviceFaultVendorInfoEXT vendor[4];
   if (counts.addressInfoCount > 4)
      counts.addressInfoCount = 4;
   if (counts.vendorInfoCount > 4)
      counts.vendorInfoCount = 4;
   counts.vendorBinarySize = 0;
   VkDeviceFaultInfoEXT info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
      .pAddressInfos = addr,
      .pVendorInfos = vendor,
   };
   r = f->GetDeviceFaultInfoEXT(device, &counts, &info);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      printf("  vkGetDeviceFaultInfoEXT: %s\n", result_str(r));
      return;
   }
   printf("  device fault: %s\n", info.description);
   for (uint32_t i = 0; i < counts.vendorInfoCount; i++)
      printf("  vendor record %u: %s (kbase event code 0x%llx, atom %llu)\n", i,
             vendor[i].description, (unsigned long long)vendor[i].vendorFaultCode,
             (unsigned long long)vendor[i].vendorFaultData);
   for (uint32_t i = 0; i < counts.addressInfoCount; i++)
      printf("  address record %u: type %d address 0x%llx\n", i, addr[i].addressType,
             (unsigned long long)addr[i].reportedAddress);
}

/* A host-visible memory type for `type_bits`, host-coherent if there is
 * one; -1 when there is none. */
static int
pick_memory_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t type_bits, bool *coherent)
{
   int type = -1;
   *coherent = false;
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      VkMemoryPropertyFlags fl = mp->memoryTypes[i].propertyFlags;
      if (!(type_bits & (1u << i)) || !(fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         continue;
      const bool c = fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      if (type < 0 || (c && !*coherent)) {
         type = (int)i;
         *coherent = c;
      }
   }
   return type;
}

/* A host-visible buffer with its memory mapped; false on failure (logged). */
struct host_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   void *map;
   bool coherent;
   VkDeviceSize size;
};

static bool
host_buffer_create(const struct fns *f, VkDevice device,
                   const VkPhysicalDeviceMemoryProperties *mp, VkDeviceSize size,
                   VkBufferUsageFlags usage, const char *what, struct host_buffer *out)
{
   memset(out, 0, sizeof(*out));
   const VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkResult r = f->CreateBuffer(device, &bci, NULL, &out->buffer);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateBuffer (%s) failed: %s\n", what, result_str(r));
      return false;
   }
   VkMemoryRequirements reqs;
   f->GetBufferMemoryRequirements(device, out->buffer, &reqs);
   int type = pick_memory_type(mp, reqs.memoryTypeBits, &out->coherent);
   if (type < 0) {
      fprintf(stderr, "tex_g52: no host-visible memory type for the %s\n", what);
      return false;
   }
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)type,
   };
   r = f->AllocateMemory(device, &mai, NULL, &out->memory);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkAllocateMemory (%s) failed: %s\n", what, result_str(r));
      return false;
   }
   r = f->BindBufferMemory(device, out->buffer, out->memory, 0);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkBindBufferMemory (%s) failed: %s\n", what, result_str(r));
      return false;
   }
   r = f->MapMemory(device, out->memory, 0, VK_WHOLE_SIZE, 0, &out->map);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkMapMemory (%s) failed: %s\n", what, result_str(r));
      return false;
   }
   out->size = reqs.size;
   return true;
}

static void
host_buffer_flush(const struct fns *f, VkDevice device, const struct host_buffer *b)
{
   if (b->coherent)
      return;
   const VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = b->memory,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
   };
   f->FlushMappedMemoryRanges(device, 1, &whole);
}

static void
host_buffer_destroy(const struct fns *f, VkDevice device, struct host_buffer *b)
{
   if (b->map)
      f->UnmapMemory(device, b->memory);
   if (b->buffer)
      f->DestroyBuffer(device, b->buffer, NULL);
   if (b->memory)
      f->FreeMemory(device, b->memory, NULL);
   memset(b, 0, sizeof(*b));
}

/* Submit one command buffer and wait for it. 0 ok, else an exit status. */
static int
submit_and_wait(const struct fns *f, VkDevice device, VkQueue queue, VkFence fence,
                VkCommandBuffer cmd, const char *what)
{
   f->ResetFences(device, 1, &fence);
   const VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   };
   const int64_t t0 = now_ns();
   VkResult r = f->QueueSubmit(queue, 1, &si, fence);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkQueueSubmit (%s) failed: %s (kbase JOB_SUBMIT; see the log)\n",
              what, result_str(r));
      if (r == VK_ERROR_DEVICE_LOST)
         print_fault(f, device);
      return 1;
   }
   r = f->WaitForFences(device, 1, &fence, VK_TRUE, WAIT_NS);
   const int64_t t1 = now_ns();
   printf("  vkWaitForFences (%s): %s after %.3f ms\n", what, result_str(r),
          (double)(t1 - t0) / 1e6);
   if (r == VK_TIMEOUT) {
      printf("FAIL: the fence of the %s never signalled: an atom did not complete. Check dmesg "
             "for mali/kbase lines.\n", what);
      return 1;
   }
   if (r != VK_SUCCESS) {
      printf("FAIL: vkWaitForFences (%s) returned %s\n", what, result_str(r));
      if (r == VK_ERROR_DEVICE_LOST)
         print_fault(f, device);
      return 1;
   }
   return 0;
}

enum upload_mode { UPLOAD_AUTO, UPLOAD_COPY, UPLOAD_HOST };

/* ---------------------------------------------------------------------- */

int
main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IOLBF, 0);

   const char *path = NULL;
   bool tiled = false;
   enum upload_mode upload = UPLOAD_AUTO;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--tiled")) {
         tiled = true;
      } else if (!strcmp(argv[i], "--upload=copy")) {
         upload = UPLOAD_COPY;
      } else if (!strcmp(argv[i], "--upload=host")) {
         upload = UPLOAD_HOST;
      } else if (!strcmp(argv[i], "--upload=auto")) {
         upload = UPLOAD_AUTO;
      } else if (argv[i][0] == '-') {
         fprintf(stderr, "tex_g52: unknown option %s\n", argv[i]);
         return 2;
      } else {
         path = argv[i];
      }
   }
   if (!path)
      path = getenv("MALISX2_ICD");
   if (!path || !*path)
      path = TEX_G52_DEFAULT_ICD;
   if (tiled && upload == UPLOAD_HOST) {
      fprintf(stderr, "tex_g52: a tiled texture cannot be written from the host\n");
      return 2;
   }

   printf("tex_g52: loading %s\n", path);
   void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
   if (!lib) {
      fprintf(stderr, "tex_g52: dlopen(%s) failed: %s\n", path, dlerror());
      return 2;
   }

   struct fns f;
   memset(&f, 0, sizeof(f));
   f.gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vk_icdGetInstanceProcAddr");
   if (!f.gipa) {
      fprintf(stderr, "tex_g52: %s exports no vk_icdGetInstanceProcAddr: %s\n", path, dlerror());
      return 2;
   }

   bool ok = true;
   f.CreateInstance = (PFN_vkCreateInstance)load_instance_fn(&f, NULL, "vkCreateInstance", &ok);
   if (!ok)
      return 2;

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "tex_g52",
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance = VK_NULL_HANDLE;
   VkResult r = f.CreateInstance(&ici, NULL, &instance);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateInstance failed: %s\n", result_str(r));
      return 2;
   }

#define INST(name) f.name = (PFN_vk##name)load_instance_fn(&f, instance, "vk" #name, &ok)
   INST(DestroyInstance);
   INST(EnumeratePhysicalDevices);
   INST(GetPhysicalDeviceProperties2);
   INST(GetPhysicalDeviceMemoryProperties);
   INST(EnumerateDeviceExtensionProperties);
   INST(CreateDevice);
   INST(GetDeviceProcAddr);
#undef INST
   if (!ok)
      return 2;

   uint32_t pdev_count = 0;
   r = f.EnumeratePhysicalDevices(instance, &pdev_count, NULL);
   if (r != VK_SUCCESS || pdev_count == 0) {
      fprintf(stderr, "tex_g52: no physical device (%s, count %u): the driver refused /dev/mali0; "
              "see the driver log\n", result_str(r), pdev_count);
      return 2;
   }
   VkPhysicalDevice pdevs[8];
   if (pdev_count > 8)
      pdev_count = 8;
   r = f.EnumeratePhysicalDevices(instance, &pdev_count, pdevs);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      fprintf(stderr, "tex_g52: vkEnumeratePhysicalDevices failed: %s\n", result_str(r));
      return 2;
   }
   VkPhysicalDevice pdev = pdevs[0];

   VkPhysicalDeviceDriverProperties drv = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &drv,
   };
   f.GetPhysicalDeviceProperties2(pdev, &props2);
   const VkPhysicalDeviceProperties *p = &props2.properties;
   printf("  device: %s, apiVersion %u.%u.%u, driverInfo \"%s\"\n", p->deviceName,
          VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion),
          VK_API_VERSION_PATCH(p->apiVersion), drv.driverInfo);

   VkPhysicalDeviceMemoryProperties mem_props;
   f.GetPhysicalDeviceMemoryProperties(pdev, &mem_props);

   bool have_fault_ext = false;
   {
      uint32_t n = 0;
      if (f.EnumerateDeviceExtensionProperties(pdev, NULL, &n, NULL) == VK_SUCCESS && n) {
         VkExtensionProperties *ext = calloc(n, sizeof(*ext));
         if (ext && f.EnumerateDeviceExtensionProperties(pdev, NULL, &n, ext) == VK_SUCCESS) {
            for (uint32_t i = 0; i < n; i++)
               if (!strcmp(ext[i].extensionName, VK_EXT_DEVICE_FAULT_EXTENSION_NAME))
                  have_fault_ext = true;
         }
         free(ext);
      }
   }

   const float prio = 1.0f;
   const VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceFaultFeaturesEXT fault_feat = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT,
      .deviceFault = VK_TRUE,
   };
   const char *dev_exts[] = {VK_EXT_DEVICE_FAULT_EXTENSION_NAME};
   const VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = have_fault_ext ? &fault_feat : NULL,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .enabledExtensionCount = have_fault_ext ? 1 : 0,
      .ppEnabledExtensionNames = dev_exts,
   };
   VkDevice device = VK_NULL_HANDLE;
   r = f.CreateDevice(pdev, &dci, NULL, &device);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateDevice failed: %s (see the log)\n", result_str(r));
      return 2;
   }
   printf("  VkDevice created%s\n", have_fault_ext ? " (VK_EXT_device_fault enabled)" : "");

#define DEV(name) \
   f.name = (PFN_vk##name)load_device_fn(&f, device, "vk" #name, NULL, true, &ok)
   DEV(DestroyDevice);
   DEV(GetDeviceQueue);
   DEV(CreateImage);
   DEV(DestroyImage);
   DEV(GetImageMemoryRequirements);
   DEV(GetImageSubresourceLayout);
   DEV(CreateBuffer);
   DEV(DestroyBuffer);
   DEV(GetBufferMemoryRequirements);
   DEV(BindBufferMemory);
   DEV(AllocateMemory);
   DEV(FreeMemory);
   DEV(BindImageMemory);
   DEV(MapMemory);
   DEV(UnmapMemory);
   DEV(FlushMappedMemoryRanges);
   DEV(InvalidateMappedMemoryRanges);
   DEV(CreateImageView);
   DEV(DestroyImageView);
   DEV(CreateSampler);
   DEV(DestroySampler);
   DEV(CreateDescriptorSetLayout);
   DEV(DestroyDescriptorSetLayout);
   DEV(CreateDescriptorPool);
   DEV(DestroyDescriptorPool);
   DEV(AllocateDescriptorSets);
   DEV(UpdateDescriptorSets);
   DEV(CreateShaderModule);
   DEV(DestroyShaderModule);
   DEV(CreatePipelineLayout);
   DEV(DestroyPipelineLayout);
   DEV(CreateGraphicsPipelines);
   DEV(DestroyPipeline);
   DEV(CreateCommandPool);
   DEV(DestroyCommandPool);
   DEV(AllocateCommandBuffers);
   DEV(BeginCommandBuffer);
   DEV(EndCommandBuffer);
   DEV(CmdPipelineBarrier);
   DEV(CmdCopyBufferToImage);
   DEV(CmdBindPipeline);
   DEV(CmdBindDescriptorSets);
   DEV(CmdBindVertexBuffers);
   DEV(CmdDraw);
   DEV(CreateFence);
   DEV(DestroyFence);
   DEV(ResetFences);
   DEV(QueueSubmit);
   DEV(WaitForFences);
   DEV(DeviceWaitIdle);
#undef DEV
   f.CmdBeginRendering = (PFN_vkCmdBeginRendering)load_device_fn(
      &f, device, "vkCmdBeginRendering", "vkCmdBeginRenderingKHR", true, &ok);
   f.CmdEndRendering = (PFN_vkCmdEndRendering)load_device_fn(
      &f, device, "vkCmdEndRendering", "vkCmdEndRenderingKHR", true, &ok);
   if (have_fault_ext)
      f.GetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)load_device_fn(
         &f, device, "vkGetDeviceFaultInfoEXT", NULL, false, &ok);
   if (!ok)
      return 1;

   VkQueue queue = VK_NULL_HANDLE;
   f.GetDeviceQueue(device, 0, 0, &queue);

   const VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence = VK_NULL_HANDLE;
   r = f.CreateFence(device, &fci, NULL, &fence);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateFence failed: %s\n", result_str(r));
      return 1;
   }
   const VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkCommandPool pool = VK_NULL_HANDLE;
   r = f.CreateCommandPool(device, &cpci, NULL, &pool);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateCommandPool failed: %s\n", result_str(r));
      return 1;
   }
   const VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 2,
   };
   VkCommandBuffer cmds[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
   r = f.AllocateCommandBuffers(device, &cbai, cmds);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkAllocateCommandBuffers failed: %s\n", result_str(r));
      return 1;
   }
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };

   /* ------------------------------------------------------------------- */
   /* The texture                                                          */

   const VkImageLayout tex_initial =
      tiled ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_PREINITIALIZED;
   const VkImageCreateInfo tci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {TEX_W, TEX_H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = tiled ? VK_IMAGE_TILING_OPTIMAL : VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = tex_initial,
   };
   VkImage tex = VK_NULL_HANDLE;
   r = f.CreateImage(device, &tci, NULL, &tex);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateImage (texture) failed: %s\n", result_str(r));
      return 1;
   }
   VkMemoryRequirements treqs;
   f.GetImageMemoryRequirements(device, tex, &treqs);
   bool tcoherent = false;
   int ttype = pick_memory_type(&mem_props, treqs.memoryTypeBits, &tcoherent);
   if (ttype < 0) {
      fprintf(stderr, "tex_g52: no host-visible memory type for the texture\n");
      return 1;
   }
   const VkMemoryAllocateInfo tmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = treqs.size,
      .memoryTypeIndex = (uint32_t)ttype,
   };
   VkDeviceMemory tmemory = VK_NULL_HANDLE;
   r = f.AllocateMemory(device, &tmai, NULL, &tmemory);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkAllocateMemory (texture) failed: %s\n", result_str(r));
      return 1;
   }
   r = f.BindImageMemory(device, tex, tmemory, 0);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkBindImageMemory (texture) failed: %s\n", result_str(r));
      return 1;
   }
   void *tmap = NULL;
   r = f.MapMemory(device, tmemory, 0, VK_WHOLE_SIZE, 0, &tmap);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkMapMemory (texture) failed: %s\n", result_str(r));
      return 1;
   }
   /* Zero, so a texel the upload missed reads as zero (the gray signature). */
   memset(tmap, 0, (size_t)treqs.size);
   const VkMappedMemoryRange twhole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = tmemory,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
   };
   if (!tcoherent)
      f.FlushMappedMemoryRanges(device, 1, &twhole);
   VkSubresourceLayout tlayout = {.rowPitch = TEX_W * 4, .offset = 0};
   if (!tiled) {
      const VkImageSubresource sub = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT};
      f.GetImageSubresourceLayout(device, tex, &sub, &tlayout);
   }
   printf("  texture: %ux%u R8G8B8A8_UNORM %s, %llu bytes, row pitch %llu, memory type %d\n",
          TEX_W, TEX_H, tiled ? "optimal (tiled)" : "linear", (unsigned long long)treqs.size,
          (unsigned long long)tlayout.rowPitch, ttype);

   /* The texels, in a staging buffer. */
   uint8_t texels[TEX_W * TEX_H * 4];
   for (unsigned y = 0; y < TEX_H; y++)
      for (unsigned x = 0; x < TEX_W; x++)
         texel(x, y, &texels[(y * TEX_W + x) * 4]);

   struct host_buffer staging;
   if (!host_buffer_create(&f, device, &mem_props, sizeof(texels),
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "staging buffer", &staging))
      return 1;
   memcpy(staging.map, texels, sizeof(texels));
   host_buffer_flush(&f, device, &staging);

   /* Upload: the copy command, or the host. */
   const char *upload_how = "host write";
   bool copied = false;
   if (upload != UPLOAD_HOST) {
      VkCommandBuffer cmd = cmds[0];
      r = f.BeginCommandBuffer(cmd, &bi);
      if (r != VK_SUCCESS) {
         fprintf(stderr, "tex_g52: vkBeginCommandBuffer (copy) failed: %s\n", result_str(r));
         return 1;
      }
      const VkImageMemoryBarrier to_dst = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = tex_initial,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = tex,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, 0, NULL, 0, NULL, 1, &to_dst);
      const VkBufferImageCopy region = {
         .bufferOffset = 0,
         .bufferRowLength = 0,
         .bufferImageHeight = 0,
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageOffset = {0, 0, 0},
         .imageExtent = {TEX_W, TEX_H, 1},
      };
      f.CmdCopyBufferToImage(cmd, staging.buffer, tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                             &region);
      const VkImageMemoryBarrier to_read = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = tex,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1,
                           &to_read);
      r = f.EndCommandBuffer(cmd);
      if (r == VK_SUCCESS) {
         int st = submit_and_wait(&f, device, queue, fence, cmd, "copy");
         if (st)
            return st;
         upload_how = "vkCmdCopyBufferToImage (the driver's compute copy)";
         copied = true;
      } else if (r == VK_ERROR_FEATURE_NOT_PRESENT && upload == UPLOAD_AUTO && !tiled) {
         /* A driver that refuses the copy (a TODO v7 stub): the host write
          * below, and the verdict says so. */
         printf("  vkCmdCopyBufferToImage: recording refused with %s (the driver has no copy "
                "for this upload; see the log's TODO v7 line); writing from the host\n",
                result_str(r));
      } else {
         /* Any other failure is the copy path's own: a copy shader that did
          * not compile, its dispatch or command memory. */
         printf("  vkCmdCopyBufferToImage: recording failed with %s (the compute copy path; "
                "see the log)\n", result_str(r));
         const char *why = "";
         if (upload == UPLOAD_COPY)
            why = " (--upload=copy)";
         else if (tiled)
            why = " (a tiled texture needs the copy)";
         printf("FAIL: the texture upload through vkCmdCopyBufferToImage did not record%s\n", why);
         return 1;
      }
   }
   if (!copied) {
      /* Straight into the linear image. */
      uint8_t *base = (uint8_t *)tmap + tlayout.offset;
      for (unsigned y = 0; y < TEX_H; y++)
         memcpy(base + (size_t)y * tlayout.rowPitch, &texels[y * TEX_W * 4], TEX_W * 4);
      if (!tcoherent)
         f.FlushMappedMemoryRanges(device, 1, &twhole);
   }
   printf("  texture uploaded by %s\n", upload_how);

   const VkImageViewCreateInfo tvci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = tex,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkImageView tview = VK_NULL_HANDLE;
   r = f.CreateImageView(device, &tvci, NULL, &tview);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateImageView (texture) failed: %s (the Bifrost Texture "
              "descriptor; see the log)\n", result_str(r));
      return 1;
   }
   const VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = 0.25f,
   };
   VkSampler sampler = VK_NULL_HANDLE;
   r = f.CreateSampler(device, &sci, NULL, &sampler);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateSampler failed: %s\n", result_str(r));
      return 1;
   }

   /* The uniform buffer. */
   struct host_buffer ubo;
   if (!host_buffer_create(&f, device, &mem_props, sizeof(params),
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "uniform buffer", &ubo))
      return 1;
   memcpy(ubo.map, params, sizeof(params));
   host_buffer_flush(&f, device, &ubo);

   /* The dynamic uniform buffer: dyn.add of each half at DYN_ADD_OFFSET of
    * its block, the padding filled with a value that would show. */
   struct host_buffer dynbuf;
   if (!host_buffer_create(&f, device, &mem_props, 2 * DYN_STRIDE,
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "dynamic uniform buffer", &dynbuf))
      return 1;
   for (unsigned i = 0; i < 2 * DYN_STRIDE / 4; i++)
      ((float *)dynbuf.map)[i] = 0.75f;
   for (unsigned h = 0; h < 2; h++)
      memcpy((uint8_t *)dynbuf.map + h * DYN_STRIDE + DYN_ADD_OFFSET, dyn_add[h], 16);
   host_buffer_flush(&f, device, &dynbuf);
   struct host_buffer vdynbuf;
   if (!host_buffer_create(&f, device, &mem_props, 2 * DYN_STRIDE,
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "vertex dynamic uniform buffer",
                           &vdynbuf))
      return 1;
   for (unsigned i = 0; i < 2 * DYN_STRIDE / 4; i++)
      ((float *)vdynbuf.map)[i] = 0.75f;
   for (unsigned h = 0; h < 2; h++)
      memcpy((uint8_t *)vdynbuf.map + h * DYN_STRIDE + VDYN_K_OFFSET, vdyn_k, 16);
   host_buffer_flush(&f, device, &vdynbuf);

   /* The descriptor set: binding 0 the combined image sampler, binding 1
    * the uniform buffer, both fragment. */
   const VkDescriptorSetLayoutBinding bindings[4] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
         .binding = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
         .binding = 3,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      },
   };
   const VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 4,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
   r = f.CreateDescriptorSetLayout(device, &dslci, NULL, &dsl);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateDescriptorSetLayout failed: %s\n", result_str(r));
      return 1;
   }
   const VkDescriptorPoolSize pool_sizes[3] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2},
   };
   const VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 3,
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool dpool = VK_NULL_HANDLE;
   r = f.CreateDescriptorPool(device, &dpci, NULL, &dpool);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateDescriptorPool failed: %s\n", result_str(r));
      return 1;
   }
   const VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool,
      .descriptorSetCount = 1,
      .pSetLayouts = &dsl,
   };
   VkDescriptorSet set = VK_NULL_HANDLE;
   r = f.AllocateDescriptorSets(device, &dsai, &set);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkAllocateDescriptorSets failed: %s\n", result_str(r));
      return 1;
   }
   const VkDescriptorImageInfo dii = {
      .sampler = sampler,
      .imageView = tview,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkDescriptorBufferInfo dbi = {
      .buffer = ubo.buffer,
      .offset = 0,
      .range = sizeof(params),
   };
   const VkDescriptorBufferInfo ddbi = {
      .buffer = dynbuf.buffer,
      .offset = 0,
      .range = DYN_BLOCK_SIZE,
   };
   const VkDescriptorBufferInfo vddbi = {
      .buffer = vdynbuf.buffer,
      .offset = 0,
      .range = VDYN_BLOCK_SIZE,
   };
   const VkWriteDescriptorSet writes[4] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &dii,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pBufferInfo = &dbi,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .pBufferInfo = &ddbi,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 3,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .pBufferInfo = &vddbi,
      },
   };
   f.UpdateDescriptorSets(device, 4, writes, 0, NULL);
   printf("  descriptor set written (combined image sampler, uniform buffer, dynamic uniform "
          "buffer, vertex dynamic uniform buffer)\n");

   /* ------------------------------------------------------------------- */
   /* The target                                                           */

   const VkImageCreateInfo ici_img = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {IMG_W, IMG_H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image = VK_NULL_HANDLE;
   r = f.CreateImage(device, &ici_img, NULL, &image);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateImage (target) failed: %s\n", result_str(r));
      return 1;
   }
   VkMemoryRequirements reqs;
   f.GetImageMemoryRequirements(device, image, &reqs);
   bool coherent = false;
   int type = pick_memory_type(&mem_props, reqs.memoryTypeBits, &coherent);
   if (type < 0) {
      fprintf(stderr, "tex_g52: no host-visible memory type for the target\n");
      return 1;
   }
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)type,
   };
   VkDeviceMemory memory = VK_NULL_HANDLE;
   r = f.AllocateMemory(device, &mai, NULL, &memory);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkAllocateMemory (target) failed: %s\n", result_str(r));
      return 1;
   }
   r = f.BindImageMemory(device, image, memory, 0);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkBindImageMemory (target) failed: %s\n", result_str(r));
      return 1;
   }
   void *map = NULL;
   r = f.MapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &map);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkMapMemory (target) failed: %s\n", result_str(r));
      return 1;
   }
   memset(map, SENTINEL, (size_t)reqs.size);
   const VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = memory,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
   };
   if (!coherent)
      f.FlushMappedMemoryRanges(device, 1, &whole);
   VkSubresourceLayout layout = {.rowPitch = IMG_W * 4, .offset = 0};
   {
      const VkImageSubresource sub = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT};
      f.GetImageSubresourceLayout(device, image, &sub, &layout);
   }
   printf("  target: %ux%u R8G8B8A8_UNORM linear, %llu bytes, row pitch %llu, memory type %d\n",
          IMG_W, IMG_H, (unsigned long long)reqs.size, (unsigned long long)layout.rowPitch, type);

   const VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkImageView view = VK_NULL_HANDLE;
   r = f.CreateImageView(device, &vci, NULL, &view);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateImageView (target) failed: %s\n", result_str(r));
      return 1;
   }

   /* The vertex buffer: twelve vec2. */
   struct host_buffer vbuf;
   if (!host_buffer_create(&f, device, &mem_props, sizeof(vertices),
                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "vertex buffer", &vbuf))
      return 1;
   memcpy(vbuf.map, vertices, sizeof(vertices));
   host_buffer_flush(&f, device, &vbuf);

   /* ------------------------------------------------------------------- */
   /* Shaders and the pipeline                                             */

   const VkShaderModuleCreateInfo vsmci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(vert_spv),
      .pCode = vert_spv,
   };
   const VkShaderModuleCreateInfo fsmci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(frag_spv),
      .pCode = frag_spv,
   };
   VkShaderModule vsm = VK_NULL_HANDLE, fsm = VK_NULL_HANDLE;
   r = f.CreateShaderModule(device, &vsmci, NULL, &vsm);
   if (r == VK_SUCCESS)
      r = f.CreateShaderModule(device, &fsmci, NULL, &fsm);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateShaderModule failed: %s\n", result_str(r));
      return 1;
   }
   const VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl = VK_NULL_HANDLE;
   r = f.CreatePipelineLayout(device, &plci, NULL, &pl);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreatePipelineLayout failed: %s\n", result_str(r));
      return 1;
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
      .offset = 0,
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
   const VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = VK_FALSE,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &cba,
   };
   const VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
   const VkPipelineRenderingCreateInfo prci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &color_format,
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
      .pColorBlendState = &cb,
      .layout = pl,
   };
   VkPipeline pipeline = VK_NULL_HANDLE;
   const int64_t tc0 = now_ns();
   r = f.CreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, NULL, &pipeline);
   const int64_t tc1 = now_ns();
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkCreateGraphicsPipelines failed: %s (the Bifrost compile, the "
              "descriptor lowering or a TODO v7 refusal; see the log)\n", result_str(r));
      return 1;
   }
   printf("  pipeline created in %.2f ms (fragment shader samples one texture and reads one "
          "uniform buffer)\n", (double)(tc1 - tc0) / 1e6);

   /* ------------------------------------------------------------------- */
   /* The draw                                                             */

   VkCommandBuffer cmd = cmds[1];
   r = f.BeginCommandBuffer(cmd, &bi);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkBeginCommandBuffer (draw) failed: %s\n", result_str(r));
      return 1;
   }

   VkImageMemoryBarrier pre[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = image,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      },
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = copied ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_HOST_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
         .oldLayout = copied ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : tex_initial,
         .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = tex,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      },
   };
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_HOST_BIT |
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        0, 0, NULL, 0, NULL, 2, pre);

   const VkRenderingAttachmentInfo color = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .resolveMode = VK_RESOLVE_MODE_NONE,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = {clear_rgba[0], clear_rgba[1], clear_rgba[2], clear_rgba[3]},
   };
   const VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {IMG_W, IMG_H}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color,
   };
   f.CmdBeginRendering(cmd, &ri);
   f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   const VkDeviceSize vb_offset = 0;
   f.CmdBindVertexBuffers(cmd, 0, 1, &vbuf.buffer, &vb_offset);
   for (uint32_t h = 0; h < 2; h++) {
      /* Binding order: the fragment shader's buffer, then the vertex
       * shader's, which moves the other way. */
      const uint32_t dyn_offsets[2] = {h * DYN_STRIDE, (1 - h) * DYN_STRIDE};
      f.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &set, 2,
                              dyn_offsets);
      f.CmdDraw(cmd, 6, 1, 6 * h, 0);
   }
   f.CmdEndRendering(cmd);

   VkImageMemoryBarrier to_host = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 0, NULL, 1, &to_host);

   r = f.EndCommandBuffer(cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "tex_g52: vkEndCommandBuffer (draw) failed: %s (recording failed: the "
              "descriptor tables, a TODO v7 stub or command memory; see the log)\n",
              result_str(r));
      return 1;
   }
   printf("  command buffer recorded (clear, two textured draws of 6 vertices, the set bound "
          "at dynamic offsets 0 and %u)\n", DYN_STRIDE);

   int st = submit_and_wait(&f, device, queue, fence, cmd, "draw");
   if (st)
      return st;

   if (!coherent)
      f.InvalidateMappedMemoryRanges(device, 1, &whole);

   /* ------------------------------------------------------------------- */
   /* Verify                                                               */

   const uint8_t *base = (const uint8_t *)map + layout.offset;
   uint32_t bad = 0, untouched = 0, black = 0, gray = 0;
   int first_x = -1, first_y = -1;
   uint8_t first_got[4] = {0}, first_want[4] = {0};
   uint32_t block_bad[TEX_H][TEX_W];
   memset(block_bad, 0, sizeof(block_bad));
   for (uint32_t y = 0; y < IMG_H; y++) {
      const uint8_t *row = base + (size_t)y * layout.rowPitch;
      for (uint32_t x = 0; x < IMG_W; x++) {
         const uint8_t *px = row + (size_t)x * 4;
         uint8_t t[4], want[4];
         texel(x / BLOCK, y / BLOCK, t);
         expected_pixel(t, y >= IMG_H / 2, want);
         bool okpx = true;
         for (unsigned c = 0; c < 4; c++) {
            const int d = (int)px[c] - (int)want[c];
            if (d < -1 || d > 1)
               okpx = false;
         }
         if (okpx)
            continue;
         if (px[0] == SENTINEL && px[1] == SENTINEL && px[2] == SENTINEL && px[3] == SENTINEL)
            untouched++;
         else if (px[0] == 0 && px[1] == 0 && px[2] == 0)
            black++;
         else if (px[0] >= 63 && px[0] <= 65 && px[1] >= 63 && px[1] <= 65 && px[2] >= 63 &&
                  px[2] <= 65)
            gray++;
         block_bad[y / BLOCK][x / BLOCK]++;
         if (!bad) {
            first_x = (int)x;
            first_y = (int)y;
            memcpy(first_got, px, 4);
            memcpy(first_want, want, 4);
         }
         bad++;
      }
   }

   /* The centre pixel of each texel block, for the record. */
   for (unsigned by = 0; by < TEX_H; by++) {
      char line[256];
      int n = snprintf(line, sizeof(line), "  block row %u:", by);
      for (unsigned bx = 0; bx < TEX_W; bx++) {
         const uint8_t *px = base + (size_t)(by * BLOCK + BLOCK / 2) * layout.rowPitch +
                             (size_t)(bx * BLOCK + BLOCK / 2) * 4;
         uint8_t t[4], want[4];
         texel(bx, by, t);
         expected_pixel(t, by * BLOCK >= IMG_H / 2, want);
         n += snprintf(line + n, sizeof(line) - n, "  %3u %3u %3u %3u (want %3u %3u %3u %3u)%s",
                       px[0], px[1], px[2], px[3], want[0], want[1], want[2], want[3],
                       block_bad[by][bx] ? "!" : " ");
      }
      printf("%s\n", line);
   }

   int status;
   if (bad == 0) {
      printf("PASS: all %u pixels show their texel (%u texel blocks of %ux%u), uploaded by %s\n",
             IMG_W * IMG_H, TEX_W * TEX_H, BLOCK, BLOCK, upload_how);
      status = 0;
   } else {
      printf("FAIL: %u of %u pixels wrong (%u black: the uniform buffer read as zero; %u gray "
             "64: the texture read as zero; %u still hold the 0x%02x sentinel); first at "
             "(%d,%d) = %u %u %u %u, expected %u %u %u %u (texel %d,%d); texture uploaded by %s\n",
             bad, IMG_W * IMG_H, black, gray, untouched, SENTINEL, first_x, first_y,
             first_got[0], first_got[1], first_got[2], first_got[3], first_want[0],
             first_want[1], first_want[2], first_want[3], first_x / BLOCK, first_y / BLOCK,
             upload_how);
      status = 1;
   }

   f.DeviceWaitIdle(device);
   f.UnmapMemory(device, memory);
   f.UnmapMemory(device, tmemory);
   f.DestroyFence(device, fence, NULL);
   f.DestroyCommandPool(device, pool, NULL);
   f.DestroyPipeline(device, pipeline, NULL);
   f.DestroyPipelineLayout(device, pl, NULL);
   f.DestroyShaderModule(device, vsm, NULL);
   f.DestroyShaderModule(device, fsm, NULL);
   f.DestroyDescriptorPool(device, dpool, NULL);
   f.DestroyDescriptorSetLayout(device, dsl, NULL);
   f.DestroySampler(device, sampler, NULL);
   host_buffer_destroy(&f, device, &vbuf);
   host_buffer_destroy(&f, device, &ubo);
   host_buffer_destroy(&f, device, &dynbuf);
   host_buffer_destroy(&f, device, &vdynbuf);
   host_buffer_destroy(&f, device, &staging);
   f.DestroyImageView(device, tview, NULL);
   f.DestroyImage(device, tex, NULL);
   f.FreeMemory(device, tmemory, NULL);
   f.DestroyImageView(device, view, NULL);
   f.DestroyImage(device, image, NULL);
   f.FreeMemory(device, memory, NULL);
   f.DestroyDevice(device, NULL);
   f.DestroyInstance(instance, NULL);
   dlclose(lib);
   return status;
}
