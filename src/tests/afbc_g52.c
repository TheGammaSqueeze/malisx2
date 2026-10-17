/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * afbc_g52: AFBC colour images on the Bifrost (v7) path of this driver.
 * Every case runs twice, each in its own process: once with
 * MALISX2_V7_AFBC=1 (the eligible images must come out AFBC) and once with
 * MALISX2_V7_AFBC=0 (they must not), so a difference between the two runs
 * is the compression and nothing else. Every image is 128x128 (two header
 * tiles of 8x8 superblocks each way) unless a case says otherwise, every
 * result is read back through vkCmdCopyImageToBuffer (the decompressing
 * copy) or from a linear image, and checked on the CPU:
 *
 *   L  layout: the eligible images (colour attachment, sampled, transfer,
 *      optimal) are AFBC in the on run and not in the off run, judged by
 *      their memory size against the same image without attachment usage
 *      (AFBC adds a 4 KiB aligned header block); a storage image, a 32x32
 *      image and a mutable-format image are never AFBC;
 *   C  copies, per format (R8G8B8A8_UNORM, R8G8B8A8_SRGB, R8G8_UNORM,
 *      R8_UNORM): a pattern uploaded with vkCmdCopyBufferToImage, a second
 *      pattern over a ragged rectangle (37, 21) 50x45 that cuts
 *      superblocks, read back exactly; vkCmdCopyImage of an offset
 *      rectangle into a second prefilled image of the same kind, and of the
 *      whole image into an image without attachment usage (never AFBC),
 *      both read back exactly;
 *   A  draw and blend (RGBA8): cleared to grey, a full-screen triangle
 *      blended SRC_ALPHA / ONE_MINUS_SRC_ALPHA, then a second pass that
 *      loads it (loadOp LOAD, render area (5, 3) 100x90, a partly covered
 *      superblock at every edge) and draws an opaque colour, then a third
 *      that loads and draws nothing; within one LSB;
 *   T  texture sampling: the uploaded pattern of C and the rendered target
 *      of A sampled (nearest, texel centres) by a full-screen triangle into
 *      a linear target; within one LSB;
 *   B  blits: the pattern blitted 1:1 into a linear image (read through
 *      its mapping), that linear image blitted back mirrored on X into a
 *      fresh AFBC image, read back; within one LSB;
 *   M  eight AFBC render targets in one pass (the smallest tiles the G52
 *      picks, smaller than a superblock on the 8 KiB tile buffer part, so
 *      the reverse issue order and the forced clean tile writes of v7
 *      AFBC), cleared, then loaded with a render area (5, 3) 61x45 and
 *      drawn; within one LSB. Skipped when maxColorAttachments < 8.
 *
 * The colour cases allow one LSB per channel (a float round trip through
 * a shader and the blender, as blit_g52); the pattern's neighbouring
 * texels differ by at least 7 in some channel, so a wrong texel is never
 * within it. The copies compare exactly.
 *
 * The shaders are SPIR-V embedded as arrays, generated with glslangValidator
 * 16.6.0 (`glslangValidator -V --target-env vulkan1.1`) from:
 *
 *   pos2.vert:  #version 450
 *               layout(location = 0) in vec2 pos;
 *               void main() { gl_Position = vec4(pos, 0.0, 1.0); }
 *
 *   color.frag: #version 450
 *               layout(push_constant) uniform PC { vec4 color; vec2 inv_size; } pc;
 *               layout(location = 0) out vec4 color;
 *               void main() { color = pc.color; }
 *
 *   tex.frag:   #version 450
 *               layout(set = 0, binding = 0) uniform sampler2D tex;
 *               layout(push_constant) uniform PC { vec4 color; vec2 inv_size; } pc;
 *               layout(location = 0) out vec4 color;
 *               void main() { color = texture(tex, gl_FragCoord.xy * pc.inv_size); }
 *
 *   mrt.frag:   #version 450
 *               layout(push_constant) uniform PC { vec4 color; vec2 inv_size; } pc;
 *               layout(location = 0) out vec4 o0;  ...  layout(location = 7) out vec4 o7;
 *               vec4 rt(int i) {
 *                  return vec4(pc.color.rg, float(i * 16 + 8) / 255.0, pc.color.a);
 *               }
 *               void main() { o0 = rt(0); ... o7 = rt(7); }
 *
 * Usage: afbc_g52 [path/to/libvulkan_malisx2.so] [--only=on|off]
 *   The path defaults to $MALISX2_ICD, then to AFBC_G52_DEFAULT_ICD.
 *   --only runs one of the two passes in this process.
 *
 * Exit status: 0 PASS (both runs); 1 FAIL (wrong pixels, a wrong layout, a
 * timeout or a Vulkan error once the device exists); 2 a run did not get
 * as far as a VkDevice.
 *
 * Driver log: MESA_LOG=file sends it to stderr; MALISX2_DUMP_DRAW=1 dumps
 * every draw's descriptors (see MALISX2_G52.md).
 */

#include "g52_harness.h"

#include <math.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef AFBC_G52_DEFAULT_ICD
#define AFBC_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define IMG 128
#define SENTINEL 0xa5
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* ---------------------------------------------------------------------- */
/* Shaders (see the file comment)                                          */

static const uint32_t pos2_vert_spv[] = {
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
   0x00000014, 0x00000000, 0x0004002b, 0x00000006, 0x00000015, 0x3f800000,
   0x00040020, 0x00000019, 0x00000003, 0x00000007, 0x00050036, 0x00000002,
   0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d,
   0x00000010, 0x00000013, 0x00000012, 0x00050051, 0x00000006, 0x00000016,
   0x00000013, 0x00000000, 0x00050051, 0x00000006, 0x00000017, 0x00000013,
   0x00000001, 0x00070050, 0x00000007, 0x00000018, 0x00000016, 0x00000017,
   0x00000014, 0x00000015, 0x00050041, 0x00000019, 0x0000001a, 0x0000000d,
   0x0000000f, 0x0003003e, 0x0000001a, 0x00000018, 0x000100fd, 0x00010038,
};

static const uint32_t color_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000013, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
   0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
   0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63, 0x00000072,
   0x00030005, 0x0000000b, 0x00004350, 0x00050006, 0x0000000b, 0x00000000,
   0x6f6c6f63, 0x00000072, 0x00060006, 0x0000000b, 0x00000001, 0x5f766e69,
   0x657a6973, 0x00000000, 0x00030005, 0x0000000d, 0x00006370, 0x00040047,
   0x00000009, 0x0000001e, 0x00000000, 0x00030047, 0x0000000b, 0x00000002,
   0x00050048, 0x0000000b, 0x00000000, 0x00000023, 0x00000000, 0x00050048,
   0x0000000b, 0x00000001, 0x00000023, 0x00000010, 0x00020013, 0x00000002,
   0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
   0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008,
   0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003,
   0x00040017, 0x0000000a, 0x00000006, 0x00000002, 0x0004001e, 0x0000000b,
   0x00000007, 0x0000000a, 0x00040020, 0x0000000c, 0x00000009, 0x0000000b,
   0x0004003b, 0x0000000c, 0x0000000d, 0x00000009, 0x00040015, 0x0000000e,
   0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000,
   0x00040020, 0x00000010, 0x00000009, 0x00000007, 0x00050036, 0x00000002,
   0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041,
   0x00000010, 0x00000011, 0x0000000d, 0x0000000f, 0x0004003d, 0x00000007,
   0x00000012, 0x00000011, 0x0003003e, 0x00000009, 0x00000012, 0x000100fd,
   0x00010038,
};

static const uint32_t tex_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x0000001e, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00000010, 0x00030010,
   0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
   0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63,
   0x00000072, 0x00030005, 0x0000000d, 0x00786574, 0x00060005, 0x00000010,
   0x465f6c67, 0x43676172, 0x64726f6f, 0x00000000, 0x00030005, 0x00000014,
   0x00004350, 0x00050006, 0x00000014, 0x00000000, 0x6f6c6f63, 0x00000072,
   0x00060006, 0x00000014, 0x00000001, 0x5f766e69, 0x657a6973, 0x00000000,
   0x00030005, 0x00000016, 0x00006370, 0x00040047, 0x00000009, 0x0000001e,
   0x00000000, 0x00040047, 0x0000000d, 0x00000021, 0x00000000, 0x00040047,
   0x0000000d, 0x00000022, 0x00000000, 0x00040047, 0x00000010, 0x0000000b,
   0x0000000f, 0x00030047, 0x00000014, 0x00000002, 0x00050048, 0x00000014,
   0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000014, 0x00000001,
   0x00000023, 0x00000010, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
   0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007,
   0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007,
   0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00090019, 0x0000000a,
   0x00000006, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000001,
   0x00000000, 0x0003001b, 0x0000000b, 0x0000000a, 0x00040020, 0x0000000c,
   0x00000000, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000000,
   0x00040020, 0x0000000f, 0x00000001, 0x00000007, 0x0004003b, 0x0000000f,
   0x00000010, 0x00000001, 0x00040017, 0x00000011, 0x00000006, 0x00000002,
   0x0004001e, 0x00000014, 0x00000007, 0x00000011, 0x00040020, 0x00000015,
   0x00000009, 0x00000014, 0x0004003b, 0x00000015, 0x00000016, 0x00000009,
   0x00040015, 0x00000017, 0x00000020, 0x00000001, 0x0004002b, 0x00000017,
   0x00000018, 0x00000001, 0x00040020, 0x00000019, 0x00000009, 0x00000011,
   0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
   0x00000005, 0x0004003d, 0x0000000b, 0x0000000e, 0x0000000d, 0x0004003d,
   0x00000007, 0x00000012, 0x00000010, 0x0007004f, 0x00000011, 0x00000013,
   0x00000012, 0x00000012, 0x00000000, 0x00000001, 0x00050041, 0x00000019,
   0x0000001a, 0x00000016, 0x00000018, 0x0004003d, 0x00000011, 0x0000001b,
   0x0000001a, 0x00050085, 0x00000011, 0x0000001c, 0x00000013, 0x0000001b,
   0x00050057, 0x00000007, 0x0000001d, 0x0000000e, 0x0000001c, 0x0003003e,
   0x00000009, 0x0000001d, 0x000100fd, 0x00010038,
};

static const uint32_t mrt_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000049, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x000d000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000002a, 0x0000002d, 0x00000031,
   0x00000035, 0x00000039, 0x0000003d, 0x00000041, 0x00000045, 0x00030010,
   0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
   0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x0000000c, 0x69287472,
   0x00003b31, 0x00030005, 0x0000000b, 0x00000069, 0x00030005, 0x0000000f,
   0x00004350, 0x00050006, 0x0000000f, 0x00000000, 0x6f6c6f63, 0x00000072,
   0x00060006, 0x0000000f, 0x00000001, 0x5f766e69, 0x657a6973, 0x00000000,
   0x00030005, 0x00000011, 0x00006370, 0x00030005, 0x0000002a, 0x0000306f,
   0x00040005, 0x0000002b, 0x61726170, 0x0000006d, 0x00030005, 0x0000002d,
   0x0000316f, 0x00040005, 0x0000002f, 0x61726170, 0x0000006d, 0x00030005,
   0x00000031, 0x0000326f, 0x00040005, 0x00000033, 0x61726170, 0x0000006d,
   0x00030005, 0x00000035, 0x0000336f, 0x00040005, 0x00000037, 0x61726170,
   0x0000006d, 0x00030005, 0x00000039, 0x0000346f, 0x00040005, 0x0000003b,
   0x61726170, 0x0000006d, 0x00030005, 0x0000003d, 0x0000356f, 0x00040005,
   0x0000003f, 0x61726170, 0x0000006d, 0x00030005, 0x00000041, 0x0000366f,
   0x00040005, 0x00000043, 0x61726170, 0x0000006d, 0x00030005, 0x00000045,
   0x0000376f, 0x00040005, 0x00000047, 0x61726170, 0x0000006d, 0x00030047,
   0x0000000f, 0x00000002, 0x00050048, 0x0000000f, 0x00000000, 0x00000023,
   0x00000000, 0x00050048, 0x0000000f, 0x00000001, 0x00000023, 0x00000010,
   0x00040047, 0x0000002a, 0x0000001e, 0x00000000, 0x00040047, 0x0000002d,
   0x0000001e, 0x00000001, 0x00040047, 0x00000031, 0x0000001e, 0x00000002,
   0x00040047, 0x00000035, 0x0000001e, 0x00000003, 0x00040047, 0x00000039,
   0x0000001e, 0x00000004, 0x00040047, 0x0000003d, 0x0000001e, 0x00000005,
   0x00040047, 0x00000041, 0x0000001e, 0x00000006, 0x00040047, 0x00000045,
   0x0000001e, 0x00000007, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
   0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000001, 0x00040020,
   0x00000007, 0x00000007, 0x00000006, 0x00030016, 0x00000008, 0x00000020,
   0x00040017, 0x00000009, 0x00000008, 0x00000004, 0x00040021, 0x0000000a,
   0x00000009, 0x00000007, 0x00040017, 0x0000000e, 0x00000008, 0x00000002,
   0x0004001e, 0x0000000f, 0x00000009, 0x0000000e, 0x00040020, 0x00000010,
   0x00000009, 0x0000000f, 0x0004003b, 0x00000010, 0x00000011, 0x00000009,
   0x0004002b, 0x00000006, 0x00000012, 0x00000000, 0x00040020, 0x00000013,
   0x00000009, 0x00000009, 0x0004002b, 0x00000006, 0x00000018, 0x00000010,
   0x0004002b, 0x00000006, 0x0000001a, 0x00000008, 0x0004002b, 0x00000008,
   0x0000001d, 0x437f0000, 0x00040015, 0x0000001f, 0x00000020, 0x00000000,
   0x0004002b, 0x0000001f, 0x00000020, 0x00000003, 0x00040020, 0x00000021,
   0x00000009, 0x00000008, 0x00040020, 0x00000029, 0x00000003, 0x00000009,
   0x0004003b, 0x00000029, 0x0000002a, 0x00000003, 0x0004003b, 0x00000029,
   0x0000002d, 0x00000003, 0x0004002b, 0x00000006, 0x0000002e, 0x00000001,
   0x0004003b, 0x00000029, 0x00000031, 0x00000003, 0x0004002b, 0x00000006,
   0x00000032, 0x00000002, 0x0004003b, 0x00000029, 0x00000035, 0x00000003,
   0x0004002b, 0x00000006, 0x00000036, 0x00000003, 0x0004003b, 0x00000029,
   0x00000039, 0x00000003, 0x0004002b, 0x00000006, 0x0000003a, 0x00000004,
   0x0004003b, 0x00000029, 0x0000003d, 0x00000003, 0x0004002b, 0x00000006,
   0x0000003e, 0x00000005, 0x0004003b, 0x00000029, 0x00000041, 0x00000003,
   0x0004002b, 0x00000006, 0x00000042, 0x00000006, 0x0004003b, 0x00000029,
   0x00000045, 0x00000003, 0x0004002b, 0x00000006, 0x00000046, 0x00000007,
   0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
   0x00000005, 0x0004003b, 0x00000007, 0x0000002b, 0x00000007, 0x0004003b,
   0x00000007, 0x0000002f, 0x00000007, 0x0004003b, 0x00000007, 0x00000033,
   0x00000007, 0x0004003b, 0x00000007, 0x00000037, 0x00000007, 0x0004003b,
   0x00000007, 0x0000003b, 0x00000007, 0x0004003b, 0x00000007, 0x0000003f,
   0x00000007, 0x0004003b, 0x00000007, 0x00000043, 0x00000007, 0x0004003b,
   0x00000007, 0x00000047, 0x00000007, 0x0003003e, 0x0000002b, 0x00000012,
   0x00050039, 0x00000009, 0x0000002c, 0x0000000c, 0x0000002b, 0x0003003e,
   0x0000002a, 0x0000002c, 0x0003003e, 0x0000002f, 0x0000002e, 0x00050039,
   0x00000009, 0x00000030, 0x0000000c, 0x0000002f, 0x0003003e, 0x0000002d,
   0x00000030, 0x0003003e, 0x00000033, 0x00000032, 0x00050039, 0x00000009,
   0x00000034, 0x0000000c, 0x00000033, 0x0003003e, 0x00000031, 0x00000034,
   0x0003003e, 0x00000037, 0x00000036, 0x00050039, 0x00000009, 0x00000038,
   0x0000000c, 0x00000037, 0x0003003e, 0x00000035, 0x00000038, 0x0003003e,
   0x0000003b, 0x0000003a, 0x00050039, 0x00000009, 0x0000003c, 0x0000000c,
   0x0000003b, 0x0003003e, 0x00000039, 0x0000003c, 0x0003003e, 0x0000003f,
   0x0000003e, 0x00050039, 0x00000009, 0x00000040, 0x0000000c, 0x0000003f,
   0x0003003e, 0x0000003d, 0x00000040, 0x0003003e, 0x00000043, 0x00000042,
   0x00050039, 0x00000009, 0x00000044, 0x0000000c, 0x00000043, 0x0003003e,
   0x00000041, 0x00000044, 0x0003003e, 0x00000047, 0x00000046, 0x00050039,
   0x00000009, 0x00000048, 0x0000000c, 0x00000047, 0x0003003e, 0x00000045,
   0x00000048, 0x000100fd, 0x00010038, 0x00050036, 0x00000009, 0x0000000c,
   0x00000000, 0x0000000a, 0x00030037, 0x00000007, 0x0000000b, 0x000200f8,
   0x0000000d, 0x00050041, 0x00000013, 0x00000014, 0x00000011, 0x00000012,
   0x0004003d, 0x00000009, 0x00000015, 0x00000014, 0x0007004f, 0x0000000e,
   0x00000016, 0x00000015, 0x00000015, 0x00000000, 0x00000001, 0x0004003d,
   0x00000006, 0x00000017, 0x0000000b, 0x00050084, 0x00000006, 0x00000019,
   0x00000017, 0x00000018, 0x00050080, 0x00000006, 0x0000001b, 0x00000019,
   0x0000001a, 0x0004006f, 0x00000008, 0x0000001c, 0x0000001b, 0x00050088,
   0x00000008, 0x0000001e, 0x0000001c, 0x0000001d, 0x00060041, 0x00000021,
   0x00000022, 0x00000011, 0x00000012, 0x00000020, 0x0004003d, 0x00000008,
   0x00000023, 0x00000022, 0x00050051, 0x00000008, 0x00000024, 0x00000016,
   0x00000000, 0x00050051, 0x00000008, 0x00000025, 0x00000016, 0x00000001,
   0x00070050, 0x00000009, 0x00000026, 0x00000024, 0x00000025, 0x0000001e,
   0x00000023, 0x000200fe, 0x00000026, 0x00010038,
};

/* A triangle covering clip space. */
static const float vertices[6] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};

/* The push constant block of every fragment shader here. */
struct push {
   float color[4];
   float inv_size[2];
   float pad[2];
};

#define MAX_RTS 8

/* Entry points the harness does not load. */
struct x_fns {
   PFN_vkCreateSampler CreateSampler;
   PFN_vkDestroySampler DestroySampler;
   PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
   PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
   PFN_vkCreateDescriptorPool CreateDescriptorPool;
   PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
   PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
   PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
   PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
   PFN_vkCmdPushConstants CmdPushConstants;
   PFN_vkCmdCopyImage CmdCopyImage;
   PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties;
};

struct t {
   struct g52 g;
   struct x_fns x;
   const char *run;          /* "on" or "off" */
   bool on;                  /* AFBC expected on the eligible images */
   unsigned passes, fails;
   uint32_t max_rts;
   VkCommandPool pool;
   VkCommandBuffer cmd;
   VkPipelineLayout pl;      /* push constants */
   VkDescriptorSetLayout dsl;
   VkPipelineLayout pl_tex;  /* set 0 (one combined image sampler) + push constants */
   VkDescriptorPool dpool;
   VkSampler sampler;
   struct g52_buffer vbuf;
};

#define ATT_USAGE                                                                          \
   (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |                      \
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
#define PLAIN_USAGE                                                                        \
   (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |                          \
    VK_IMAGE_USAGE_TRANSFER_DST_BIT)

static void
verdict(struct t *t, bool ok, const char *name, const char *what)
{
   printf("  %s %s [%s]: %s\n", ok ? "PASS" : "FAIL", name, t->run, what);
   if (ok)
      t->passes++;
   else
      t->fails++;
}

static unsigned
format_bpp(VkFormat f)
{
   switch (f) {
   case VK_FORMAT_R8_UNORM: return 1;
   case VK_FORMAT_R8G8_UNORM: return 2;
   default: return 4;
   }
}

static const char *
format_name(VkFormat f)
{
   switch (f) {
   case VK_FORMAT_R8_UNORM: return "R8_UNORM";
   case VK_FORMAT_R8G8_UNORM: return "R8G8_UNORM";
   case VK_FORMAT_R8G8B8A8_SRGB: return "R8G8B8A8_SRGB";
   default: return "R8G8B8A8_UNORM";
   }
}

/* Byte c of texel (x, y) of pattern `seed`: neighbours differ by at least
 * 7 along x and 3 along y. */
static uint8_t
pat(unsigned seed, unsigned x, unsigned y, unsigned c)
{
   return (uint8_t)(x * (7 + seed) + y * (3 + 2 * seed) + c * 61 + seed * 17);
}

static void
fill_pattern(uint8_t *dst, unsigned bpp, unsigned w, unsigned h, unsigned seed)
{
   for (unsigned y = 0; y < h; y++)
      for (unsigned x = 0; x < w; x++)
         for (unsigned c = 0; c < bpp; c++)
            dst[(y * w + x) * bpp + c] = pat(seed, x, y, c);
}

/* Compares w x h texels of `bpp` bytes, rows of `got_pitch` bytes, with a
 * tight `want`; returns the number of wrong texels and prints the first. */
static unsigned
compare(const char *what, const uint8_t *got, size_t got_pitch, const uint8_t *want, unsigned bpp,
        unsigned w, unsigned h, int tolerance)
{
   unsigned bad = 0, sentinel = 0;
   int maxd = 0;
   for (unsigned y = 0; y < h; y++) {
      for (unsigned x = 0; x < w; x++) {
         const uint8_t *g = got + y * got_pitch + x * bpp;
         const uint8_t *e = want + (y * w + x) * bpp;
         bool okpx = true, sent = true;
         for (unsigned c = 0; c < bpp; c++) {
            const int d = abs((int)g[c] - (int)e[c]);
            maxd = d > maxd ? d : maxd;
            if (d > tolerance)
               okpx = false;
            if (g[c] != SENTINEL)
               sent = false;
         }
         if (okpx)
            continue;
         if (sent)
            sentinel++;
         if (!bad) {
            printf("    %s: first wrong texel (%u,%u) =", what, x, y);
            for (unsigned c = 0; c < bpp; c++)
               printf(" %u", g[c]);
            printf(", expected");
            for (unsigned c = 0; c < bpp; c++)
               printf(" %u", e[c]);
            printf("\n");
         }
         bad++;
      }
   }
   if (bad)
      printf("    %s: %u of %u texels wrong (%u hold the sentinel), largest channel "
             "difference %d\n", what, bad, w * h, sentinel, maxd);
   else if (maxd)
      printf("    %s: largest channel difference %d (allowed %d)\n", what, maxd, tolerance);
   return bad;
}

/* ---------------------------------------------------------------------- */
/* Recording                                                               */

static bool
begin(struct t *t)
{
   const struct g52_fns *f = &t->g.f;
   VkResult r = f->ResetCommandPool(t->g.device, t->pool, 0);
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   if (r == VK_SUCCESS)
      r = f->BeginCommandBuffer(t->cmd, &bi);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "afbc_g52: starting a command buffer failed: %s\n", g52_result_str(r));
      return false;
   }
   return true;
}

static bool
submit(struct t *t, const char *what)
{
   VkResult r = t->g.f.EndCommandBuffer(t->cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "afbc_g52: vkEndCommandBuffer (%s) failed: %s (recording failed: a TODO v7 "
              "refusal, a meta shader that would not compile, or command memory; see the "
              "log)\n", what, g52_result_str(r));
      return false;
   }
   return g52_submit_wait(&t->g, t->cmd, what) == VK_SUCCESS;
}

static void
to_general(struct t *t, VkImage image)
{
   g52_cmd_transition(&t->g, t->cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}

static void
barrier(struct t *t)
{
   g52_cmd_full_barrier(&t->g, t->cmd);
}

static void
upload(struct t *t, const struct g52_hostbuf *b, VkImage image, int x, int y, unsigned w,
       unsigned h)
{
   const VkBufferImageCopy r = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageOffset = {x, y, 0},
      .imageExtent = {w, h, 1},
   };
   t->g.f.CmdCopyBufferToImage(t->cmd, b->buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
}

static void
readback(struct t *t, VkImage image, const struct g52_hostbuf *b, unsigned w, unsigned h)
{
   const VkBufferImageCopy r = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {w, h, 1},
   };
   t->g.f.CmdCopyImageToBuffer(t->cmd, image, VK_IMAGE_LAYOUT_GENERAL, b->buffer, 1, &r);
}

static void
copy_image(struct t *t, VkImage src, int sx, int sy, VkImage dst, int dx, int dy, unsigned w,
           unsigned h)
{
   const VkImageCopy r = {
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffset = {sx, sy, 0},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffset = {dx, dy, 0},
      .extent = {w, h, 1},
   };
   t->x.CmdCopyImage(t->cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
}

/* A pass over `views` (one per render target) with `area`, every target
 * cleared to clear[i] or loaded. The caller draws between begin_pass and
 * the vkCmdEndRendering. */
static void
begin_pass(struct t *t, const VkImageView *views, unsigned n, bool load, const float (*clear)[4],
           VkRect2D area)
{
   VkRenderingAttachmentInfo att[MAX_RTS];
   for (unsigned i = 0; i < n; i++) {
      att[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = views[i],
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .resolveMode = VK_RESOLVE_MODE_NONE,
         .loadOp = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      if (!load)
         memcpy(att[i].clearValue.color.float32, clear[i], 16);
   }
   const VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = area,
      .layerCount = 1,
      .colorAttachmentCount = n,
      .pColorAttachments = att,
   };
   t->g.f.CmdBeginRendering(t->cmd, &ri);
}

static void
draw(struct t *t, VkPipeline pipe, VkPipelineLayout layout, const float color[4])
{
   struct push pc = {.inv_size = {1.0f / IMG, 1.0f / IMG}};
   memcpy(pc.color, color, sizeof(pc.color));
   const VkDeviceSize off = 0;
   t->g.f.CmdBindPipeline(t->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   t->x.CmdPushConstants(t->cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
   t->g.f.CmdBindVertexBuffers(t->cmd, 0, 1, &t->vbuf.buffer, &off);
   t->g.f.CmdDraw(t->cmd, 3, 1, 0, 0);
}

/* ---------------------------------------------------------------------- */
/* Setup                                                                   */

static bool
load_fns(struct t *t)
{
   struct g52 *g = &t->g;
   bool ok = true;
#define DEV(name) t->x.name = (PFN_vk##name)g52_load_device_fn(g, "vk" #name, NULL, true, &ok)
   DEV(CreateSampler);
   DEV(DestroySampler);
   DEV(CreateDescriptorSetLayout);
   DEV(DestroyDescriptorSetLayout);
   DEV(CreateDescriptorPool);
   DEV(DestroyDescriptorPool);
   DEV(AllocateDescriptorSets);
   DEV(UpdateDescriptorSets);
   DEV(CmdBindDescriptorSets);
   DEV(CmdPushConstants);
   DEV(CmdCopyImage);
#undef DEV
   t->x.GetPhysicalDeviceFormatProperties = g->f.GetPhysicalDeviceFormatProperties;
   if (!g->f.CmdCopyBufferToImage || !g->f.CmdCopyImageToBuffer || !g->f.CmdBlitImage) {
      fprintf(stderr, "afbc_g52: the driver lacks a transfer entry point\n");
      ok = false;
   }
   return ok;
}

static bool
setup(struct t *t)
{
   const struct g52_fns *f = &t->g.f;
   VkDevice dev = t->g.device;

   VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
   f->GetPhysicalDeviceProperties2(t->g.pdev, &props);
   t->max_rts = props.properties.limits.maxColorAttachments;

   const VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkResult r = f->CreateCommandPool(dev, &cpci, NULL, &t->pool);
   const VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = t->pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (r == VK_SUCCESS)
      r = f->AllocateCommandBuffers(dev, &cbai, &t->cmd);

   const VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      .size = sizeof(struct push),
   };
   const VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr,
   };
   if (r == VK_SUCCESS)
      r = f->CreatePipelineLayout(dev, &plci, NULL, &t->pl);

   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   if (r == VK_SUCCESS)
      r = t->x.CreateDescriptorSetLayout(dev, &dslci, NULL, &t->dsl);
   const VkPipelineLayoutCreateInfo plci_tex = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &t->dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr,
   };
   if (r == VK_SUCCESS)
      r = f->CreatePipelineLayout(dev, &plci_tex, NULL, &t->pl_tex);

   const VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4};
   const VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 4,
      .poolSizeCount = 1,
      .pPoolSizes = &ps,
   };
   if (r == VK_SUCCESS)
      r = t->x.CreateDescriptorPool(dev, &dpci, NULL, &t->dpool);

   const VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = 0.0f,
   };
   if (r == VK_SUCCESS)
      r = t->x.CreateSampler(dev, &sci, NULL, &t->sampler);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "afbc_g52: setup failed: %s\n", g52_result_str(r));
      return false;
   }
   return g52_buffer_create(&t->g, &t->vbuf, vertices, sizeof(vertices),
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
}

/* A pipeline drawing `nrt` RGBA8 targets with `fs`, the scissor `sc`;
 * with `blend`, SRC_ALPHA / ONE_MINUS_SRC_ALPHA on colour and ONE / ZERO
 * on alpha, else opaque. */
static VkPipeline
make_pipeline(struct t *t, const uint32_t *fs, size_t fs_size, VkPipelineLayout layout,
              unsigned nrt, bool blend, VkRect2D sc, const char *what)
{
   const struct g52_fns *f = &t->g.f;
   VkDevice dev = t->g.device;
   const VkShaderModuleCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(pos2_vert_spv),
      .pCode = pos2_vert_spv,
   };
   const VkShaderModuleCreateInfo fci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = fs_size,
      .pCode = fs,
   };
   VkShaderModule vsm = VK_NULL_HANDLE, fsm = VK_NULL_HANDLE;
   VkResult r = f->CreateShaderModule(dev, &vci, NULL, &vsm);
   if (r == VK_SUCCESS)
      r = f->CreateShaderModule(dev, &fci, NULL, &fsm);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "afbc_g52: vkCreateShaderModule (%s) failed: %s\n", what, g52_result_str(r));
      return VK_NULL_HANDLE;
   }
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vsm, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fsm, .pName = "main"},
   };
   const VkVertexInputBindingDescription vb = {0, 2 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
   const VkVertexInputAttributeDescription va = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
   const VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vb,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &va,
   };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {0.0f, 0.0f, (float)IMG, (float)IMG, 0.0f, 1.0f};
   const VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &sc,
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
   const VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
   };
   VkPipelineColorBlendAttachmentState bs[MAX_RTS];
   VkFormat formats[MAX_RTS];
   for (unsigned i = 0; i < nrt; i++) {
      bs[i] = (VkPipelineColorBlendAttachmentState){
         .blendEnable = blend,
         .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
         .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
         .colorBlendOp = VK_BLEND_OP_ADD,
         .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
         .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
         .alphaBlendOp = VK_BLEND_OP_ADD,
         .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      };
      formats[i] = VK_FORMAT_R8G8B8A8_UNORM;
   }
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = nrt,
      .pAttachments = bs,
   };
   const VkPipelineRenderingCreateInfo prci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = nrt,
      .pColorAttachmentFormats = formats,
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
      .pDepthStencilState = &ds,
      .pColorBlendState = &cb,
      .layout = layout,
   };
   VkPipeline p = VK_NULL_HANDLE;
   r = f->CreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &p);
   f->DestroyShaderModule(dev, vsm, NULL);
   f->DestroyShaderModule(dev, fsm, NULL);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "afbc_g52: vkCreateGraphicsPipelines (%s) failed: %s\n", what,
              g52_result_str(r));
      return VK_NULL_HANDLE;
   }
   return p;
}

/* An optimal 128x128 image of `usage`. */
static bool
make_image(struct t *t, struct g52_image *im, VkFormat format, VkImageUsageFlags usage)
{
   return g52_image_create(&t->g, im, format, IMG, IMG, VK_IMAGE_TILING_OPTIMAL, usage,
                           VK_IMAGE_ASPECT_COLOR_BIT, false, 0);
}

/* The memory size of an image created with these parameters (no memory
 * bound); 0 on failure. */
static VkDeviceSize
image_size(struct t *t, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage,
           VkImageCreateFlags flags)
{
   const VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = flags,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {w, h, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   if (t->g.f.CreateImage(t->g.device, &ici, NULL, &image) != VK_SUCCESS)
      return 0;
   VkMemoryRequirements reqs;
   t->g.f.GetImageMemoryRequirements(t->g.device, image, &reqs);
   t->g.f.DestroyImage(t->g.device, image, NULL);
   return reqs.size;
}

/* AFBC adds a header block in front of the texels: an image is AFBC when
 * it needs more memory than the same image without attachment usage. */
static int
is_afbc(struct t *t, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage,
        VkImageCreateFlags flags)
{
   const VkDeviceSize s = image_size(t, format, w, h, usage, flags);
   const VkDeviceSize ref = image_size(t, format, w, h, PLAIN_USAGE, 0);
   if (!s || !ref)
      return -1;
   return s > ref;
}

/* ---------------------------------------------------------------------- */
/* Cases                                                                   */

static const VkFormat copy_formats[] = {
   VK_FORMAT_R8G8B8A8_UNORM,
   VK_FORMAT_R8G8B8A8_SRGB,
   VK_FORMAT_R8G8_UNORM,
   VK_FORMAT_R8_UNORM,
};

static void
case_layout(struct t *t)
{
   char what[160];
   for (unsigned i = 0; i < ARRAY_LEN(copy_formats); i++) {
      const int a = is_afbc(t, copy_formats[i], IMG, IMG, ATT_USAGE, 0);
      snprintf(what, sizeof(what), "%s 128x128 colour attachment: %s (expected %s)",
               format_name(copy_formats[i]), a < 0 ? "creation failed" : a ? "AFBC" : "not AFBC",
               t->on ? "AFBC" : "not AFBC");
      verdict(t, a == (int)t->on, "L", what);
   }

   VkFormatProperties fp;
   t->x.GetPhysicalDeviceFormatProperties(t->g.pdev, VK_FORMAT_R8G8B8A8_UNORM, &fp);
   if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) {
      const int a = is_afbc(t, VK_FORMAT_R8G8B8A8_UNORM, IMG, IMG,
                            ATT_USAGE | VK_IMAGE_USAGE_STORAGE_BIT, 0);
      verdict(t, a == 0, "L", "RGBA8 with storage usage is not AFBC");
   }
   const int small = is_afbc(t, VK_FORMAT_R8G8B8A8_UNORM, 32, 32, ATT_USAGE, 0);
   verdict(t, small == 0, "L", "RGBA8 32x32 (under half a header tile) is not AFBC");
   const int mut = is_afbc(t, VK_FORMAT_R8G8B8A8_UNORM, IMG, IMG, ATT_USAGE,
                           VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
   verdict(t, mut == 0, "L", "RGBA8 with a mutable format is not AFBC");
}

/* The texels case C leaves in its RGBA8 UNORM image (the pattern used by
 * T and B). */
struct pattern_image {
   struct g52_image im;
   uint8_t *texels;
};

/* Case C for one format. Keeps the first image and its expected texels in
 * `keep` when given. */
static bool
case_copies(struct t *t, VkFormat format, struct pattern_image *keep)
{
   const unsigned bpp = format_bpp(format);
   const size_t size = (size_t)IMG * IMG * bpp;
   enum { RX = 37, RY = 21, RW = 50, RH = 45 };            /* second upload */
   enum { SX = 3, SY = 5, DX = 20, DY = 9, CW = 90, CH = 80 }; /* image copy */
   char what[200];

   struct g52_image a, b, n;
   if (!make_image(t, &a, format, ATT_USAGE) || !make_image(t, &b, format, ATT_USAGE) ||
       !make_image(t, &n, format, PLAIN_USAGE))
      return false;

   uint8_t *p1 = malloc(size), *p2 = malloc(size), *p3 = malloc(size);
   uint8_t *ea = malloc(size), *eb = malloc(size);
   struct g52_hostbuf b1, b2, b3, ra, rb, rn;
   if (!p1 || !p2 || !p3 || !ea || !eb || !g52_hostbuf_create(&t->g, &b1, size, 0) ||
       !g52_hostbuf_create(&t->g, &b2, (size_t)RW * RH * bpp, 0) ||
       !g52_hostbuf_create(&t->g, &b3, size, 0) ||
       !g52_hostbuf_create(&t->g, &ra, size, SENTINEL) ||
       !g52_hostbuf_create(&t->g, &rb, size, SENTINEL) ||
       !g52_hostbuf_create(&t->g, &rn, size, SENTINEL))
      return false;
   fill_pattern(p1, bpp, IMG, IMG, 1);
   fill_pattern(p2, bpp, RW, RH, 2);
   fill_pattern(p3, bpp, IMG, IMG, 3);
   memcpy(b1.map, p1, size);
   memcpy(b2.map, p2, (size_t)RW * RH * bpp);
   memcpy(b3.map, p3, size);
   g52_hostbuf_flush(&t->g, &b1);
   g52_hostbuf_flush(&t->g, &b2);
   g52_hostbuf_flush(&t->g, &b3);

   /* Expected: a = p1 with p2 over the rectangle; b = p3 with a's
    * (SX, SY) CWxCH at (DX, DY); n = a. */
   memcpy(ea, p1, size);
   for (unsigned y = 0; y < RH; y++)
      memcpy(ea + ((RY + y) * IMG + RX) * bpp, p2 + y * RW * bpp, (size_t)RW * bpp);
   memcpy(eb, p3, size);
   for (unsigned y = 0; y < CH; y++)
      memcpy(eb + ((DY + y) * IMG + DX) * bpp, ea + ((SY + y) * IMG + SX) * bpp, (size_t)CW * bpp);

   if (!begin(t))
      return false;
   to_general(t, a.image);
   to_general(t, b.image);
   to_general(t, n.image);
   upload(t, &b1, a.image, 0, 0, IMG, IMG);
   upload(t, &b3, b.image, 0, 0, IMG, IMG);
   barrier(t);
   upload(t, &b2, a.image, RX, RY, RW, RH);
   barrier(t);
   copy_image(t, a.image, SX, SY, b.image, DX, DY, CW, CH);
   copy_image(t, a.image, 0, 0, n.image, 0, 0, IMG, IMG);
   barrier(t);
   readback(t, a.image, &ra, IMG, IMG);
   readback(t, b.image, &rb, IMG, IMG);
   readback(t, n.image, &rn, IMG, IMG);
   barrier(t);
   snprintf(what, sizeof(what), "copies %s", format_name(format));
   if (!submit(t, what))
      return false;
   g52_hostbuf_invalidate(&t->g, &ra);
   g52_hostbuf_invalidate(&t->g, &rb);
   g52_hostbuf_invalidate(&t->g, &rn);

   snprintf(what, sizeof(what), "%s: buffer to image, a ragged rectangle over it, image to buffer",
            format_name(format));
   verdict(t, !compare("upload/readback", ra.map, (size_t)IMG * bpp, ea, bpp, IMG, IMG, 0), "C",
           what);
   snprintf(what, sizeof(what), "%s: image copy of an offset rectangle into a prefilled image",
            format_name(format));
   verdict(t, !compare("image copy", rb.map, (size_t)IMG * bpp, eb, bpp, IMG, IMG, 0), "C", what);
   snprintf(what, sizeof(what), "%s: image copy into an image without attachment usage",
            format_name(format));
   verdict(t, !compare("copy to plain", rn.map, (size_t)IMG * bpp, ea, bpp, IMG, IMG, 0), "C",
           what);

   g52_hostbuf_destroy(&t->g, &b1);
   g52_hostbuf_destroy(&t->g, &b2);
   g52_hostbuf_destroy(&t->g, &b3);
   g52_hostbuf_destroy(&t->g, &ra);
   g52_hostbuf_destroy(&t->g, &rb);
   g52_hostbuf_destroy(&t->g, &rn);
   g52_image_destroy(&t->g, &b);
   g52_image_destroy(&t->g, &n);
   free(p1);
   free(p2);
   free(p3);
   free(eb);
   if (keep) {
      keep->im = a;
      keep->texels = ea;
   } else {
      g52_image_destroy(&t->g, &a);
      free(ea);
   }
   return true;
}

static uint8_t
unorm8(float v)
{
   v = v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
   return (uint8_t)lrintf(v * 255.0f);
}

/* Case A; keeps the rendered image and its expected texels in `keep`. */
static bool
case_draw(struct t *t, struct pattern_image *keep)
{
   enum { AX = 5, AY = 3, AW = 100, AH = 90 };
   const float grey[1][4] = {{0.5f, 0.5f, 0.5f, 1.0f}};
   const float src[4] = {0.25f, 0.75f, 1.0f, 0.5f};
   const float opaque[4] = {1.0f, 0.0f, 0.25f, 1.0f};
   const VkRect2D full = {{0, 0}, {IMG, IMG}}, area = {{AX, AY}, {AW, AH}};

   struct g52_image d;
   if (!make_image(t, &d, VK_FORMAT_R8G8B8A8_UNORM, ATT_USAGE))
      return false;
   VkPipeline blend = make_pipeline(t, color_frag_spv, sizeof(color_frag_spv), t->pl, 1, true,
                                    full, "blend");
   VkPipeline solid = make_pipeline(t, color_frag_spv, sizeof(color_frag_spv), t->pl, 1, false,
                                    area, "opaque");
   if (!blend || !solid)
      return false;
   const size_t size = (size_t)IMG * IMG * 4;
   struct g52_hostbuf rd;
   uint8_t *e = malloc(size);
   if (!e || !g52_hostbuf_create(&t->g, &rd, size, SENTINEL))
      return false;

   uint8_t out[4], in[4];
   for (unsigned c = 0; c < 3; c++)
      out[c] = unorm8(src[c] * src[3] + grey[0][c] * (1.0f - src[3]));
   out[3] = unorm8(src[3]);
   for (unsigned c = 0; c < 4; c++)
      in[c] = unorm8(opaque[c]);
   for (unsigned y = 0; y < IMG; y++)
      for (unsigned x = 0; x < IMG; x++) {
         const bool inside = x >= AX && x < AX + AW && y >= AY && y < AY + AH;
         memcpy(e + (y * IMG + x) * 4, inside ? in : out, 4);
      }

   if (!begin(t))
      return false;
   to_general(t, d.image);
   begin_pass(t, &d.view, 1, false, grey, full);
   draw(t, blend, t->pl, src);
   t->g.f.CmdEndRendering(t->cmd);
   barrier(t);
   begin_pass(t, &d.view, 1, true, NULL, area);
   draw(t, solid, t->pl, opaque);
   t->g.f.CmdEndRendering(t->cmd);
   barrier(t);
   /* Loaded and stored with nothing drawn: no change. */
   begin_pass(t, &d.view, 1, true, NULL, full);
   t->g.f.CmdEndRendering(t->cmd);
   barrier(t);
   readback(t, d.image, &rd, IMG, IMG);
   barrier(t);
   if (!submit(t, "draw and blend"))
      return false;
   g52_hostbuf_invalidate(&t->g, &rd);
   verdict(t, !compare("draw", rd.map, (size_t)IMG * 4, e, 4, IMG, IMG, 1), "A",
           "clear + blended triangle, LOAD with area (5,3) 100x90 + opaque draw, empty LOAD pass");

   g52_hostbuf_destroy(&t->g, &rd);
   t->g.f.DestroyPipeline(t->g.device, blend, NULL);
   t->g.f.DestroyPipeline(t->g.device, solid, NULL);
   keep->im = d;
   keep->texels = e;
   return true;
}

/* Case T: `src` sampled into a linear target, compared with its texels. */
static bool
case_sample(struct t *t, const struct pattern_image *src, const char *what)
{
   const VkRect2D full = {{0, 0}, {IMG, IMG}};
   struct g52_image l;
   if (!g52_image_create(&t->g, &l, VK_FORMAT_R8G8B8A8_UNORM, IMG, IMG, VK_IMAGE_TILING_LINEAR,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, true,
                         SENTINEL))
      return false;
   VkPipeline pipe = make_pipeline(t, tex_frag_spv, sizeof(tex_frag_spv), t->pl_tex, 1, false,
                                   full, "sample");
   if (!pipe)
      return false;
   const VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = t->dpool,
      .descriptorSetCount = 1,
      .pSetLayouts = &t->dsl,
   };
   VkDescriptorSet set;
   VkResult r = t->x.AllocateDescriptorSets(t->g.device, &dsai, &set);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "afbc_g52: vkAllocateDescriptorSets failed: %s\n", g52_result_str(r));
      return false;
   }
   const VkDescriptorImageInfo dii = {t->sampler, src->im.view, VK_IMAGE_LAYOUT_GENERAL};
   const VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &dii,
   };
   t->x.UpdateDescriptorSets(t->g.device, 1, &w, 0, NULL);

   const float black[1][4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
   const float none[4] = {0};
   if (!begin(t))
      return false;
   to_general(t, l.image);
   begin_pass(t, &l.view, 1, false, black, full);
   t->x.CmdBindDescriptorSets(t->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, t->pl_tex, 0, 1, &set, 0,
                              NULL);
   draw(t, pipe, t->pl_tex, none);
   t->g.f.CmdEndRendering(t->cmd);
   g52_cmd_transition(&t->g, t->cmd, l.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL,
                      VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                      VK_ACCESS_HOST_READ_BIT);
   if (!submit(t, "sample"))
      return false;
   g52_image_sync_for_read(&t->g, &l);
   verdict(t, !compare("sample", g52_image_pixel(&l, 0, 0), l.layout.rowPitch, src->texels, 4,
                       IMG, IMG, 1), "T", what);
   t->g.f.DestroyPipeline(t->g.device, pipe, NULL);
   g52_image_destroy(&t->g, &l);
   return true;
}

static void
blit(struct t *t, VkImage src, VkImage dst, bool mirror_x)
{
   const VkImageBlit r = {
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets = {{0, 0, 0}, {IMG, IMG, 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets = {{mirror_x ? IMG : 0, 0, 0}, {mirror_x ? 0 : IMG, IMG, 1}},
   };
   t->g.f.CmdBlitImage(t->cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &r,
                       VK_FILTER_NEAREST);
}

/* Case B: AFBC -> linear 1:1, linear -> AFBC mirrored on X. */
static bool
case_blits(struct t *t, const struct pattern_image *src)
{
   VkFormatProperties fp;
   t->x.GetPhysicalDeviceFormatProperties(t->g.pdev, VK_FORMAT_R8G8B8A8_UNORM, &fp);
   const VkFormatFeatureFlags want = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
   if ((fp.linearTilingFeatures & want) != want || (fp.optimalTilingFeatures & want) != want) {
      printf("  SKIP B [%s]: RGBA8 is not a blit source and destination in both tilings\n",
             t->run);
      return true;
   }
   struct g52_image lin, e;
   if (!g52_image_create(&t->g, &lin, VK_FORMAT_R8G8B8A8_UNORM, IMG, IMG, VK_IMAGE_TILING_LINEAR,
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, true, SENTINEL) ||
       !make_image(t, &e, VK_FORMAT_R8G8B8A8_UNORM, ATT_USAGE))
      return false;
   const size_t size = (size_t)IMG * IMG * 4;
   struct g52_hostbuf re;
   uint8_t *mirrored = malloc(size);
   if (!mirrored || !g52_hostbuf_create(&t->g, &re, size, SENTINEL))
      return false;
   for (unsigned y = 0; y < IMG; y++)
      for (unsigned x = 0; x < IMG; x++)
         memcpy(mirrored + (y * IMG + x) * 4, src->texels + (y * IMG + (IMG - 1 - x)) * 4, 4);

   if (!begin(t))
      return false;
   to_general(t, lin.image);
   to_general(t, e.image);
   blit(t, src->im.image, lin.image, false);
   barrier(t);
   blit(t, lin.image, e.image, true);
   barrier(t);
   readback(t, e.image, &re, IMG, IMG);
   g52_cmd_transition(&t->g, t->cmd, lin.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL,
                      VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                      VK_ACCESS_HOST_READ_BIT);
   barrier(t);
   if (!submit(t, "blits"))
      return false;
   g52_image_sync_for_read(&t->g, &lin);
   g52_hostbuf_invalidate(&t->g, &re);
   verdict(t, !compare("blit to linear", g52_image_pixel(&lin, 0, 0), lin.layout.rowPitch,
                       src->texels, 4, IMG, IMG, 1), "B", "AFBC pattern blitted 1:1 into a linear image");
   verdict(t, !compare("blit back", re.map, (size_t)IMG * 4, mirrored, 4, IMG, IMG, 1), "B",
           "linear image blitted back mirrored on X into a fresh AFBC image");
   free(mirrored);
   g52_hostbuf_destroy(&t->g, &re);
   g52_image_destroy(&t->g, &lin);
   g52_image_destroy(&t->g, &e);
   return true;
}

/* Case M: eight targets, cleared, then loaded with a partial area and
 * drawn. */
static bool
case_mrt(struct t *t)
{
   enum { AX = 5, AY = 3, AW = 61, AH = 45 };
   if (t->max_rts < MAX_RTS) {
      printf("  SKIP M [%s]: maxColorAttachments %u < %u\n", t->run, t->max_rts, MAX_RTS);
      return true;
   }
   const VkRect2D full = {{0, 0}, {IMG, IMG}}, area = {{AX, AY}, {AW, AH}};
   const float color[4] = {0.75f, 0.25f, 0.0f, 1.0f};
   struct g52_image im[MAX_RTS];
   VkImageView views[MAX_RTS];
   float clear[MAX_RTS][4];
   struct g52_hostbuf rb[MAX_RTS];
   const size_t size = (size_t)IMG * IMG * 4;
   for (unsigned i = 0; i < MAX_RTS; i++) {
      if (!make_image(t, &im[i], VK_FORMAT_R8G8B8A8_UNORM, ATT_USAGE) ||
          !g52_hostbuf_create(&t->g, &rb[i], size, SENTINEL))
         return false;
      views[i] = im[i].view;
      clear[i][0] = (float)(i + 1) / 10.0f;
      clear[i][1] = 0.5f;
      clear[i][2] = 0.25f;
      clear[i][3] = 1.0f;
   }
   VkPipeline pipe = make_pipeline(t, mrt_frag_spv, sizeof(mrt_frag_spv), t->pl, MAX_RTS, false,
                                   area, "eight targets");
   if (!pipe)
      return false;

   if (!begin(t))
      return false;
   for (unsigned i = 0; i < MAX_RTS; i++)
      to_general(t, im[i].image);
   begin_pass(t, views, MAX_RTS, false, (const float(*)[4])clear, full);
   t->g.f.CmdEndRendering(t->cmd);
   barrier(t);
   begin_pass(t, views, MAX_RTS, true, NULL, area);
   draw(t, pipe, t->pl, color);
   t->g.f.CmdEndRendering(t->cmd);
   barrier(t);
   for (unsigned i = 0; i < MAX_RTS; i++)
      readback(t, im[i].image, &rb[i], IMG, IMG);
   barrier(t);
   if (!submit(t, "eight targets"))
      return false;

   uint8_t *e = malloc(size);
   if (!e)
      return false;
   unsigned bad_rts = 0;
   for (unsigned i = 0; i < MAX_RTS; i++) {
      const uint8_t in[4] = {unorm8(color[0]), unorm8(color[1]), (uint8_t)(i * 16 + 8),
                             unorm8(color[3])};
      const uint8_t out[4] = {unorm8(clear[i][0]), unorm8(clear[i][1]), unorm8(clear[i][2]),
                              unorm8(clear[i][3])};
      for (unsigned y = 0; y < IMG; y++)
         for (unsigned x = 0; x < IMG; x++) {
            const bool inside = x >= AX && x < AX + AW && y >= AY && y < AY + AH;
            memcpy(e + (y * IMG + x) * 4, inside ? in : out, 4);
         }
      g52_hostbuf_invalidate(&t->g, &rb[i]);
      char what[32];
      snprintf(what, sizeof(what), "target %u", i);
      bad_rts += compare(what, rb[i].map, (size_t)IMG * 4, e, 4, IMG, IMG, 1) != 0;
   }
   verdict(t, bad_rts == 0, "M",
           "eight targets cleared, then LOAD with area (5,3) 61x45 and one draw");
   free(e);
   t->g.f.DestroyPipeline(t->g.device, pipe, NULL);
   for (unsigned i = 0; i < MAX_RTS; i++) {
      g52_hostbuf_destroy(&t->g, &rb[i]);
      g52_image_destroy(&t->g, &im[i]);
   }
   return true;
}

/* ---------------------------------------------------------------------- */

static int
run(int argc, char **argv, bool on)
{
   struct t t;
   memset(&t, 0, sizeof(t));
   t.on = on;
   t.run = on ? "on" : "off";
   setenv("MALISX2_V7_AFBC", on ? "1" : "0", 1);
   printf("afbc_g52: run with AFBC %s (MALISX2_V7_AFBC=%s)\n", t.run, on ? "1" : "0");

   int rc = g52_open(&t.g, "afbc_g52", argc, argv, AFBC_G52_DEFAULT_ICD);
   if (rc)
      return rc;
   if (!load_fns(&t) || !setup(&t))
      return 1;

   struct pattern_image pattern = {0}, drawn = {0};
   bool ok = true;
   case_layout(&t);
   for (unsigned i = 0; ok && i < ARRAY_LEN(copy_formats); i++)
      ok = case_copies(&t, copy_formats[i], i == 0 ? &pattern : NULL);
   ok = ok && case_draw(&t, &drawn);
   ok = ok && case_sample(&t, &pattern, "the uploaded AFBC pattern sampled into a linear target");
   ok = ok && case_sample(&t, &drawn, "the rendered AFBC target of A sampled into a linear target");
   ok = ok && case_blits(&t, &pattern);
   ok = ok && case_mrt(&t);
   if (!ok) {
      printf("FAIL [%s]: a case could not run (see above)\n", t.run);
      t.fails++;
   }

   const struct g52_fns *f = &t.g.f;
   f->DeviceWaitIdle(t.g.device);
   g52_image_destroy(&t.g, &pattern.im);
   g52_image_destroy(&t.g, &drawn.im);
   free(pattern.texels);
   free(drawn.texels);
   g52_buffer_destroy(&t.g, &t.vbuf);
   if (t.sampler)
      t.x.DestroySampler(t.g.device, t.sampler, NULL);
   if (t.dpool)
      t.x.DestroyDescriptorPool(t.g.device, t.dpool, NULL);
   if (t.pl_tex)
      f->DestroyPipelineLayout(t.g.device, t.pl_tex, NULL);
   if (t.dsl)
      t.x.DestroyDescriptorSetLayout(t.g.device, t.dsl, NULL);
   if (t.pl)
      f->DestroyPipelineLayout(t.g.device, t.pl, NULL);
   if (t.pool)
      f->DestroyCommandPool(t.g.device, t.pool, NULL);
   g52_close(&t.g);
   printf("%s [%s]: %u passed, %u failed\n", t.fails ? "FAIL" : "PASS", t.run, t.passes, t.fails);
   return t.fails ? 1 : 0;
}

/* Each run in a child process: the driver reads MALISX2_V7_AFBC once per
 * process. */
static int
run_child(int argc, char **argv, bool on)
{
   fflush(stdout);
   fflush(stderr);
   pid_t pid = fork();
   if (pid < 0) {
      fprintf(stderr, "afbc_g52: fork failed: %s\n", strerror(errno));
      return 2;
   }
   if (pid == 0)
      _exit(run(argc, argv, on));
   int status = 0;
   if (waitpid(pid, &status, 0) < 0) {
      fprintf(stderr, "afbc_g52: waitpid failed: %s\n", strerror(errno));
      return 2;
   }
   if (WIFSIGNALED(status)) {
      printf("FAIL [%s]: the run died of signal %d\n", on ? "on" : "off", WTERMSIG(status));
      return 1;
   }
   return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

int
main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IOLBF, 0);
   int only = -1;
   int nargs = 0;
   char *args[3] = {argv[0], NULL, NULL};
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--only=on"))
         only = 1;
      else if (!strcmp(argv[i], "--only=off"))
         only = 0;
      else if (nargs < 1)
         args[1 + nargs++] = argv[i];
   }
   if (only >= 0)
      return run(1 + nargs, args, only);
   const int on = run_child(1 + nargs, args, true);
   const int off = run_child(1 + nargs, args, false);
   const int status = on > off ? on : off;
   printf("%s: AFBC on run %s, off run %s\n", status ? "FAIL" : "PASS", on ? "failed" : "passed",
          off ? "failed" : "passed");
   return status;
}
