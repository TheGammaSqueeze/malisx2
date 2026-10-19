/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * ia_g52: input attachments (subpassLoad) on the Bifrost (v7) path of this
 * driver. ARMSX2 reads its colour and depth targets this way ("feedback"
 * descriptors, mali_nir_lower_input_attachments.c) and pushes its texture
 * set with vkCmdPushDescriptorSetKHR; no other test reaches either.
 *
 * The driver is Vulkan 1.1 without VK_KHR_dynamic_rendering_local_read, so
 * the test uses a VkRenderPass with two subpasses. Mesa's common render
 * pass code runs each subpass as its own dynamic rendering and hands the
 * subpass's input attachment map to the pipeline and the command buffer;
 * the driver then reads an input attachment that is also an attachment of
 * the current subpass from the tile buffer (LD_TILE with the iam
 * conversions), and any other one as an image through its descriptor (the
 * v7 texture table).
 *
 * Attachments, 64x64: 0 an R8G8B8A8_UNORM colour image, 1 a depth/stencil
 * image (D24_UNORM_S8_UINT and D32_SFLOAT_S8_UINT, each when offered), 2
 * an R32G32B32A32_UINT output. Before the pass the stencil aspect is
 * prefilled through vkCmdCopyBufferToImage.
 *
 *   subpass 0: one full-screen triangle writes a different colour into
 *   every pixel (from gl_FragCoord), a depth gradient (depth test ALWAYS
 *   with writes; z = z0 + (x + 0.5) / 256 + (y + 0.5) / 32768, distinct in
 *   every pixel) and the stencil (stencilLoadOp LOAD, stencil op INVERT:
 *   the complement of the per-pixel prefill);
 *
 *   subpass 1: one full-screen triangle reads them back with subpassLoad:
 *   the colour (subpassInput, input attachment 0), the depth (subpassInput,
 *   input attachment 1, a depth-only view at binding 1) and the stencil
 *   (usubpassInput, the same input attachment 1, a stencil-only view at
 *   binding 2; Mesa's render pass gives depth and stencil one index), and
 *   writes into the output: x the colour as bytes, y the depth's bits, z
 *   the stencil, w (y << 16) | x of the fragment.
 *
 * Every combination of three read modes and two descriptor modes runs, for
 * each depth/stencil format:
 *
 *   image:    subpass 1's only attachment is the output: the colour and
 *             depth/stencil images are read through their descriptors;
 *   tile:     the colour image is also subpass 1's colour attachment 1
 *             (colour write mask 0, so independentBlend) and the
 *             depth/stencil image its read-only depth/stencil attachment:
 *             all three reads come from the tile buffer;
 *   feedback: as tile, and the shader also writes 1 - colour into colour
 *             attachment 1: ARMSX2's feedback loop, the target read and
 *             written in the same subpass;
 *
 *   sets:     the three input attachment descriptors in a descriptor set
 *             bound with vkCmdBindDescriptorSets;
 *   push:     the same three pushed with vkCmdPushDescriptorSetKHR (when
 *             the device offers VK_KHR_push_descriptor).
 *
 * Each run has its own seed, so nothing a previous run left in an image
 * can pass for this one's values: the colour's blue channel (a push
 * constant), the stencil prefill, the depth gradient's z0 and the output's
 * clear value change with it.
 *
 * All four images are read back through vkCmdCopyImageToBuffer and checked
 * pixel by pixel:
 *   - the output pixel was written by its own fragment (w);
 *   - each read equals what the attachment holds: the colour exactly (the
 *     image and tile modes, where subpass 1 does not write it), the depth
 *     exactly (D32) or within one 24-bit step (D24), the stencil exactly;
 *   - the attachments hold what subpass 0 wrote: the colour within one LSB
 *     (dithering; such pixels are counted apart), the depth within 1e-5 of
 *     the gradient, the stencil exactly. In the feedback mode the colour
 *     read is checked against the pattern (one LSB) and the colour image
 *     afterwards against 255 minus the value read (one LSB).
 *
 * The shaders are SPIR-V embedded as arrays, generated with glslangValidator
 * 16.6.0 (`glslangValidator -V --target-env vulkan1.1`) from:
 *
 *   pos3.vert:     #version 450
 *                  layout(location = 0) in vec3 pos;
 *                  void main() { gl_Position = vec4(pos, 1.0); }
 *
 *   ia_write.frag: #version 450
 *                  layout(push_constant) uniform Seed { uint seed; } pc;
 *                  layout(location = 0) out vec4 color;
 *                  void main() {
 *                     uvec2 p = uvec2(gl_FragCoord.xy);
 *                     uvec4 v = uvec4(p.x * 4u + 1u, p.y * 4u + 2u,
 *                                     (p.x * 7u + p.y * 13u + pc.seed) & 255u,
 *                                     255u - ((p.x + p.y * 3u) & 127u));
 *                     color = vec4(v) / 255.0;
 *                  }
 *
 *   ia_read.frag:  #version 450
 *                  layout(input_attachment_index = 0, set = 0, binding = 0)
 *                     uniform subpassInput in_color;
 *                  layout(input_attachment_index = 1, set = 0, binding = 1)
 *                     uniform subpassInput in_depth;
 *                  layout(input_attachment_index = 1, set = 0, binding = 2)
 *                     uniform usubpassInput in_stencil;
 *                  layout(location = 0) out uvec4 result;
 *                  void main() {
 *                     uvec4 c = uvec4(clamp(subpassLoad(in_color), 0.0, 1.0) * 255.0 + 0.5);
 *                     result = uvec4(c.r | (c.g << 8) | (c.b << 16) | (c.a << 24),
 *                                    floatBitsToUint(subpassLoad(in_depth).r),
 *                                    subpassLoad(in_stencil).r,
 *                                    (uint(gl_FragCoord.y) << 16) | uint(gl_FragCoord.x));
 *                  }
 *
 *   ia_feedback.frag: ia_read.frag with a second output, the colour read
 *                  written back complemented:
 *                  layout(location = 1) out vec4 color;
 *                  void main() {
 *                     vec4 f = clamp(subpassLoad(in_color), 0.0, 1.0);
 *                     uvec4 c = uvec4(f * 255.0 + 0.5);
 *                     result = ...;   (as ia_read.frag)
 *                     color = vec4(1.0) - f;
 *                  }
 *
 * Usage: ia_g52 [path/to/libvulkan_malisx2.so] [--only=image|tile|feedback]
 *               [--sets | --push]
 *   The path defaults to $MALISX2_ICD, then to IA_G52_DEFAULT_ICD.
 *
 * Exit status: 0 PASS (every run passed, at least one ran); 1 FAIL (wrong
 * pixels, a timeout or a Vulkan error once the device exists, or nothing
 * could be run); 2 the program did not get as far as a VkDevice. After a
 * fence timeout or a lost device no further run is made and nothing waits
 * on the device again, so a hung job ends the test instead of hanging it.
 *
 * Driver log: MESA_LOG=file MESA_LOG_LEVEL=info; MALISX2_DUMP_DRAW=1 dumps
 * every draw and pass (see MALISX2_G52.md).
 */

#include "g52_harness.h"

#include <math.h>

#ifndef IA_G52_DEFAULT_ICD
#define IA_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define IMG_W 64
#define IMG_H 64
#define NPIX ((unsigned)(IMG_W * IMG_H))
/* What the readback buffers hold before the copies. */
#define RB_SENTINEL 0xa5
/* The output's clear value: a pixel the second subpass never wrote. */
#define OUT_CLEAR 0xdeadbeefu
#define OUT_CLEAR_W(seed) (0xdead0000u | (seed))
/* Stored depth against the gradient: a third of one row's step (1/32768). */
#define DEPTH_MODEL_TOL 1e-5
/* Two formats, three read modes, two descriptor modes. */
#define MAX_RUNS 12
/* Example pixels printed per check. */
#define SHOW 4

/* ---------------------------------------------------------------------- */
/* Shaders (see the file comment)                                          */

static const uint32_t pos3_vert_spv[] = {
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
   0x00000006, 0x00000003, 0x00040020, 0x00000011, 0x00000001, 0x00000010,
   0x0004003b, 0x00000011, 0x00000012, 0x00000001, 0x0004002b, 0x00000006,
   0x00000014, 0x3f800000, 0x00040020, 0x00000019, 0x00000003, 0x00000007,
   0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
   0x00000005, 0x0004003d, 0x00000010, 0x00000013, 0x00000012, 0x00050051,
   0x00000006, 0x00000015, 0x00000013, 0x00000000, 0x00050051, 0x00000006,
   0x00000016, 0x00000013, 0x00000001, 0x00050051, 0x00000006, 0x00000017,
   0x00000013, 0x00000002, 0x00070050, 0x00000007, 0x00000018, 0x00000015,
   0x00000016, 0x00000017, 0x00000014, 0x00050041, 0x00000019, 0x0000001a,
   0x0000000d, 0x0000000f, 0x0003003e, 0x0000001a, 0x00000018, 0x000100fd,
   0x00010038,
};
static const uint32_t ia_write_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000048, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000042, 0x00030010,
   0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
   0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000009, 0x00000070,
   0x00060005, 0x0000000d, 0x465f6c67, 0x43676172, 0x64726f6f, 0x00000000,
   0x00030005, 0x00000014, 0x00000076, 0x00040005, 0x0000002b, 0x64656553,
   0x00000000, 0x00050006, 0x0000002b, 0x00000000, 0x64656573, 0x00000000,
   0x00030005, 0x0000002d, 0x00006370, 0x00040005, 0x00000042, 0x6f6c6f63,
   0x00000072, 0x00040047, 0x0000000d, 0x0000000b, 0x0000000f, 0x00030047,
   0x0000002b, 0x00000002, 0x00050048, 0x0000002b, 0x00000000, 0x00000023,
   0x00000000, 0x00040047, 0x00000042, 0x0000001e, 0x00000000, 0x00020013,
   0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006,
   0x00000020, 0x00000000, 0x00040017, 0x00000007, 0x00000006, 0x00000002,
   0x00040020, 0x00000008, 0x00000007, 0x00000007, 0x00030016, 0x0000000a,
   0x00000020, 0x00040017, 0x0000000b, 0x0000000a, 0x00000004, 0x00040020,
   0x0000000c, 0x00000001, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d,
   0x00000001, 0x00040017, 0x0000000e, 0x0000000a, 0x00000002, 0x00040017,
   0x00000012, 0x00000006, 0x00000004, 0x00040020, 0x00000013, 0x00000007,
   0x00000012, 0x0004002b, 0x00000006, 0x00000015, 0x00000000, 0x00040020,
   0x00000016, 0x00000007, 0x00000006, 0x0004002b, 0x00000006, 0x00000019,
   0x00000004, 0x0004002b, 0x00000006, 0x0000001b, 0x00000001, 0x0004002b,
   0x00000006, 0x00000020, 0x00000002, 0x0004002b, 0x00000006, 0x00000024,
   0x00000007, 0x0004002b, 0x00000006, 0x00000028, 0x0000000d, 0x0003001e,
   0x0000002b, 0x00000006, 0x00040020, 0x0000002c, 0x00000009, 0x0000002b,
   0x0004003b, 0x0000002c, 0x0000002d, 0x00000009, 0x00040015, 0x0000002e,
   0x00000020, 0x00000001, 0x0004002b, 0x0000002e, 0x0000002f, 0x00000000,
   0x00040020, 0x00000030, 0x00000009, 0x00000006, 0x0004002b, 0x00000006,
   0x00000034, 0x000000ff, 0x0004002b, 0x00000006, 0x0000003a, 0x00000003,
   0x0004002b, 0x00000006, 0x0000003d, 0x0000007f, 0x00040020, 0x00000041,
   0x00000003, 0x0000000b, 0x0004003b, 0x00000041, 0x00000042, 0x00000003,
   0x0004002b, 0x0000000a, 0x00000045, 0x437f0000, 0x00050036, 0x00000002,
   0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b,
   0x00000008, 0x00000009, 0x00000007, 0x0004003b, 0x00000013, 0x00000014,
   0x00000007, 0x0004003d, 0x0000000b, 0x0000000f, 0x0000000d, 0x0007004f,
   0x0000000e, 0x00000010, 0x0000000f, 0x0000000f, 0x00000000, 0x00000001,
   0x0004006d, 0x00000007, 0x00000011, 0x00000010, 0x0003003e, 0x00000009,
   0x00000011, 0x00050041, 0x00000016, 0x00000017, 0x00000009, 0x00000015,
   0x0004003d, 0x00000006, 0x00000018, 0x00000017, 0x00050084, 0x00000006,
   0x0000001a, 0x00000018, 0x00000019, 0x00050080, 0x00000006, 0x0000001c,
   0x0000001a, 0x0000001b, 0x00050041, 0x00000016, 0x0000001d, 0x00000009,
   0x0000001b, 0x0004003d, 0x00000006, 0x0000001e, 0x0000001d, 0x00050084,
   0x00000006, 0x0000001f, 0x0000001e, 0x00000019, 0x00050080, 0x00000006,
   0x00000021, 0x0000001f, 0x00000020, 0x00050041, 0x00000016, 0x00000022,
   0x00000009, 0x00000015, 0x0004003d, 0x00000006, 0x00000023, 0x00000022,
   0x00050084, 0x00000006, 0x00000025, 0x00000023, 0x00000024, 0x00050041,
   0x00000016, 0x00000026, 0x00000009, 0x0000001b, 0x0004003d, 0x00000006,
   0x00000027, 0x00000026, 0x00050084, 0x00000006, 0x00000029, 0x00000027,
   0x00000028, 0x00050080, 0x00000006, 0x0000002a, 0x00000025, 0x00000029,
   0x00050041, 0x00000030, 0x00000031, 0x0000002d, 0x0000002f, 0x0004003d,
   0x00000006, 0x00000032, 0x00000031, 0x00050080, 0x00000006, 0x00000033,
   0x0000002a, 0x00000032, 0x000500c7, 0x00000006, 0x00000035, 0x00000033,
   0x00000034, 0x00050041, 0x00000016, 0x00000036, 0x00000009, 0x00000015,
   0x0004003d, 0x00000006, 0x00000037, 0x00000036, 0x00050041, 0x00000016,
   0x00000038, 0x00000009, 0x0000001b, 0x0004003d, 0x00000006, 0x00000039,
   0x00000038, 0x00050084, 0x00000006, 0x0000003b, 0x00000039, 0x0000003a,
   0x00050080, 0x00000006, 0x0000003c, 0x00000037, 0x0000003b, 0x000500c7,
   0x00000006, 0x0000003e, 0x0000003c, 0x0000003d, 0x00050082, 0x00000006,
   0x0000003f, 0x00000034, 0x0000003e, 0x00070050, 0x00000012, 0x00000040,
   0x0000001c, 0x00000021, 0x00000035, 0x0000003f, 0x0003003e, 0x00000014,
   0x00000040, 0x0004003d, 0x00000012, 0x00000043, 0x00000014, 0x00040070,
   0x0000000b, 0x00000044, 0x00000043, 0x00070050, 0x0000000b, 0x00000046,
   0x00000045, 0x00000045, 0x00000045, 0x00000045, 0x00050088, 0x0000000b,
   0x00000047, 0x00000044, 0x00000046, 0x0003003e, 0x00000042, 0x00000047,
   0x000100fd, 0x00010038,
};
static const uint32_t ia_read_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x0000004f, 0x00000000, 0x00020011,
   0x00000001, 0x00020011, 0x00000028, 0x0006000b, 0x00000001, 0x4c534c47,
   0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
   0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000021,
   0x00000044, 0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000002,
   0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005,
   0x00000009, 0x00000063, 0x00050005, 0x0000000d, 0x635f6e69, 0x726f6c6f,
   0x00000000, 0x00040005, 0x00000021, 0x75736572, 0x0000746c, 0x00050005,
   0x00000038, 0x645f6e69, 0x68747065, 0x00000000, 0x00050005, 0x0000003f,
   0x735f6e69, 0x636e6574, 0x00006c69, 0x00060005, 0x00000044, 0x465f6c67,
   0x43676172, 0x64726f6f, 0x00000000, 0x00040047, 0x0000000d, 0x00000021,
   0x00000000, 0x00040047, 0x0000000d, 0x00000022, 0x00000000, 0x00040047,
   0x0000000d, 0x0000002b, 0x00000000, 0x00040047, 0x00000021, 0x0000001e,
   0x00000000, 0x00040047, 0x00000038, 0x00000021, 0x00000001, 0x00040047,
   0x00000038, 0x00000022, 0x00000000, 0x00040047, 0x00000038, 0x0000002b,
   0x00000001, 0x00040047, 0x0000003f, 0x00000021, 0x00000002, 0x00040047,
   0x0000003f, 0x00000022, 0x00000000, 0x00040047, 0x0000003f, 0x0000002b,
   0x00000001, 0x00040047, 0x00000044, 0x0000000b, 0x0000000f, 0x00020013,
   0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006,
   0x00000020, 0x00000000, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
   0x00040020, 0x00000008, 0x00000007, 0x00000007, 0x00030016, 0x0000000a,
   0x00000020, 0x00090019, 0x0000000b, 0x0000000a, 0x00000006, 0x00000000,
   0x00000000, 0x00000000, 0x00000002, 0x00000000, 0x00040020, 0x0000000c,
   0x00000000, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000000,
   0x00040015, 0x0000000f, 0x00000020, 0x00000001, 0x0004002b, 0x0000000f,
   0x00000010, 0x00000000, 0x00040017, 0x00000011, 0x0000000f, 0x00000002,
   0x0005002c, 0x00000011, 0x00000012, 0x00000010, 0x00000010, 0x00040017,
   0x00000013, 0x0000000a, 0x00000004, 0x0004002b, 0x0000000a, 0x00000015,
   0x00000000, 0x0004002b, 0x0000000a, 0x00000016, 0x3f800000, 0x0004002b,
   0x0000000a, 0x0000001a, 0x437f0000, 0x0004002b, 0x0000000a, 0x0000001c,
   0x3f000000, 0x00040020, 0x00000020, 0x00000003, 0x00000007, 0x0004003b,
   0x00000020, 0x00000021, 0x00000003, 0x0004002b, 0x00000006, 0x00000022,
   0x00000000, 0x00040020, 0x00000023, 0x00000007, 0x00000006, 0x0004002b,
   0x00000006, 0x00000026, 0x00000001, 0x0004002b, 0x0000000f, 0x00000029,
   0x00000008, 0x0004002b, 0x00000006, 0x0000002c, 0x00000002, 0x0004002b,
   0x0000000f, 0x0000002f, 0x00000010, 0x0004002b, 0x00000006, 0x00000032,
   0x00000003, 0x0004002b, 0x0000000f, 0x00000035, 0x00000018, 0x0004003b,
   0x0000000c, 0x00000038, 0x00000000, 0x00090019, 0x0000003d, 0x00000006,
   0x00000006, 0x00000000, 0x00000000, 0x00000000, 0x00000002, 0x00000000,
   0x00040020, 0x0000003e, 0x00000000, 0x0000003d, 0x0004003b, 0x0000003e,
   0x0000003f, 0x00000000, 0x00040020, 0x00000043, 0x00000001, 0x00000013,
   0x0004003b, 0x00000043, 0x00000044, 0x00000001, 0x00040020, 0x00000045,
   0x00000001, 0x0000000a, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
   0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000008, 0x00000009,
   0x00000007, 0x0004003d, 0x0000000b, 0x0000000e, 0x0000000d, 0x00050062,
   0x00000013, 0x00000014, 0x0000000e, 0x00000012, 0x00070050, 0x00000013,
   0x00000017, 0x00000015, 0x00000015, 0x00000015, 0x00000015, 0x00070050,
   0x00000013, 0x00000018, 0x00000016, 0x00000016, 0x00000016, 0x00000016,
   0x0008000c, 0x00000013, 0x00000019, 0x00000001, 0x0000002b, 0x00000014,
   0x00000017, 0x00000018, 0x0005008e, 0x00000013, 0x0000001b, 0x00000019,
   0x0000001a, 0x00070050, 0x00000013, 0x0000001d, 0x0000001c, 0x0000001c,
   0x0000001c, 0x0000001c, 0x00050081, 0x00000013, 0x0000001e, 0x0000001b,
   0x0000001d, 0x0004006d, 0x00000007, 0x0000001f, 0x0000001e, 0x0003003e,
   0x00000009, 0x0000001f, 0x00050041, 0x00000023, 0x00000024, 0x00000009,
   0x00000022, 0x0004003d, 0x00000006, 0x00000025, 0x00000024, 0x00050041,
   0x00000023, 0x00000027, 0x00000009, 0x00000026, 0x0004003d, 0x00000006,
   0x00000028, 0x00000027, 0x000500c4, 0x00000006, 0x0000002a, 0x00000028,
   0x00000029, 0x000500c5, 0x00000006, 0x0000002b, 0x00000025, 0x0000002a,
   0x00050041, 0x00000023, 0x0000002d, 0x00000009, 0x0000002c, 0x0004003d,
   0x00000006, 0x0000002e, 0x0000002d, 0x000500c4, 0x00000006, 0x00000030,
   0x0000002e, 0x0000002f, 0x000500c5, 0x00000006, 0x00000031, 0x0000002b,
   0x00000030, 0x00050041, 0x00000023, 0x00000033, 0x00000009, 0x00000032,
   0x0004003d, 0x00000006, 0x00000034, 0x00000033, 0x000500c4, 0x00000006,
   0x00000036, 0x00000034, 0x00000035, 0x000500c5, 0x00000006, 0x00000037,
   0x00000031, 0x00000036, 0x0004003d, 0x0000000b, 0x00000039, 0x00000038,
   0x00050062, 0x00000013, 0x0000003a, 0x00000039, 0x00000012, 0x00050051,
   0x0000000a, 0x0000003b, 0x0000003a, 0x00000000, 0x0004007c, 0x00000006,
   0x0000003c, 0x0000003b, 0x0004003d, 0x0000003d, 0x00000040, 0x0000003f,
   0x00050062, 0x00000007, 0x00000041, 0x00000040, 0x00000012, 0x00050051,
   0x00000006, 0x00000042, 0x00000041, 0x00000000, 0x00050041, 0x00000045,
   0x00000046, 0x00000044, 0x00000026, 0x0004003d, 0x0000000a, 0x00000047,
   0x00000046, 0x0004006d, 0x00000006, 0x00000048, 0x00000047, 0x000500c4,
   0x00000006, 0x00000049, 0x00000048, 0x0000002f, 0x00050041, 0x00000045,
   0x0000004a, 0x00000044, 0x00000022, 0x0004003d, 0x0000000a, 0x0000004b,
   0x0000004a, 0x0004006d, 0x00000006, 0x0000004c, 0x0000004b, 0x000500c5,
   0x00000006, 0x0000004d, 0x00000049, 0x0000004c, 0x00070050, 0x00000007,
   0x0000004e, 0x00000037, 0x0000003c, 0x00000042, 0x0000004d, 0x0003003e,
   0x00000021, 0x0000004e, 0x000100fd, 0x00010038,
};
static const uint32_t ia_feedback_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000057, 0x00000000, 0x00020011,
   0x00000001, 0x00020011, 0x00000028, 0x0006000b, 0x00000001, 0x4c534c47,
   0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
   0x0008000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000024,
   0x00000047, 0x00000053, 0x00030010, 0x00000004, 0x00000007, 0x00030003,
   0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
   0x00030005, 0x00000009, 0x00000066, 0x00050005, 0x0000000c, 0x635f6e69,
   0x726f6c6f, 0x00000000, 0x00030005, 0x0000001b, 0x00000063, 0x00040005,
   0x00000024, 0x75736572, 0x0000746c, 0x00050005, 0x0000003b, 0x645f6e69,
   0x68747065, 0x00000000, 0x00050005, 0x00000042, 0x735f6e69, 0x636e6574,
   0x00006c69, 0x00060005, 0x00000047, 0x465f6c67, 0x43676172, 0x64726f6f,
   0x00000000, 0x00040005, 0x00000053, 0x6f6c6f63, 0x00000072, 0x00040047,
   0x0000000c, 0x00000021, 0x00000000, 0x00040047, 0x0000000c, 0x00000022,
   0x00000000, 0x00040047, 0x0000000c, 0x0000002b, 0x00000000, 0x00040047,
   0x00000024, 0x0000001e, 0x00000000, 0x00040047, 0x0000003b, 0x00000021,
   0x00000001, 0x00040047, 0x0000003b, 0x00000022, 0x00000000, 0x00040047,
   0x0000003b, 0x0000002b, 0x00000001, 0x00040047, 0x00000042, 0x00000021,
   0x00000002, 0x00040047, 0x00000042, 0x00000022, 0x00000000, 0x00040047,
   0x00000042, 0x0000002b, 0x00000001, 0x00040047, 0x00000047, 0x0000000b,
   0x0000000f, 0x00040047, 0x00000053, 0x0000001e, 0x00000001, 0x00020013,
   0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
   0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020,
   0x00000008, 0x00000007, 0x00000007, 0x00090019, 0x0000000a, 0x00000006,
   0x00000006, 0x00000000, 0x00000000, 0x00000000, 0x00000002, 0x00000000,
   0x00040020, 0x0000000b, 0x00000000, 0x0000000a, 0x0004003b, 0x0000000b,
   0x0000000c, 0x00000000, 0x00040015, 0x0000000e, 0x00000020, 0x00000001,
   0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040017, 0x00000010,
   0x0000000e, 0x00000002, 0x0005002c, 0x00000010, 0x00000011, 0x0000000f,
   0x0000000f, 0x0004002b, 0x00000006, 0x00000013, 0x00000000, 0x0004002b,
   0x00000006, 0x00000014, 0x3f800000, 0x00040015, 0x00000018, 0x00000020,
   0x00000000, 0x00040017, 0x00000019, 0x00000018, 0x00000004, 0x00040020,
   0x0000001a, 0x00000007, 0x00000019, 0x0004002b, 0x00000006, 0x0000001d,
   0x437f0000, 0x0004002b, 0x00000006, 0x0000001f, 0x3f000000, 0x00040020,
   0x00000023, 0x00000003, 0x00000019, 0x0004003b, 0x00000023, 0x00000024,
   0x00000003, 0x0004002b, 0x00000018, 0x00000025, 0x00000000, 0x00040020,
   0x00000026, 0x00000007, 0x00000018, 0x0004002b, 0x00000018, 0x00000029,
   0x00000001, 0x0004002b, 0x0000000e, 0x0000002c, 0x00000008, 0x0004002b,
   0x00000018, 0x0000002f, 0x00000002, 0x0004002b, 0x0000000e, 0x00000032,
   0x00000010, 0x0004002b, 0x00000018, 0x00000035, 0x00000003, 0x0004002b,
   0x0000000e, 0x00000038, 0x00000018, 0x0004003b, 0x0000000b, 0x0000003b,
   0x00000000, 0x00090019, 0x00000040, 0x00000018, 0x00000006, 0x00000000,
   0x00000000, 0x00000000, 0x00000002, 0x00000000, 0x00040020, 0x00000041,
   0x00000000, 0x00000040, 0x0004003b, 0x00000041, 0x00000042, 0x00000000,
   0x00040020, 0x00000046, 0x00000001, 0x00000007, 0x0004003b, 0x00000046,
   0x00000047, 0x00000001, 0x00040020, 0x00000048, 0x00000001, 0x00000006,
   0x00040020, 0x00000052, 0x00000003, 0x00000007, 0x0004003b, 0x00000052,
   0x00000053, 0x00000003, 0x0007002c, 0x00000007, 0x00000054, 0x00000014,
   0x00000014, 0x00000014, 0x00000014, 0x00050036, 0x00000002, 0x00000004,
   0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000008,
   0x00000009, 0x00000007, 0x0004003b, 0x0000001a, 0x0000001b, 0x00000007,
   0x0004003d, 0x0000000a, 0x0000000d, 0x0000000c, 0x00050062, 0x00000007,
   0x00000012, 0x0000000d, 0x00000011, 0x00070050, 0x00000007, 0x00000015,
   0x00000013, 0x00000013, 0x00000013, 0x00000013, 0x00070050, 0x00000007,
   0x00000016, 0x00000014, 0x00000014, 0x00000014, 0x00000014, 0x0008000c,
   0x00000007, 0x00000017, 0x00000001, 0x0000002b, 0x00000012, 0x00000015,
   0x00000016, 0x0003003e, 0x00000009, 0x00000017, 0x0004003d, 0x00000007,
   0x0000001c, 0x00000009, 0x0005008e, 0x00000007, 0x0000001e, 0x0000001c,
   0x0000001d, 0x00070050, 0x00000007, 0x00000020, 0x0000001f, 0x0000001f,
   0x0000001f, 0x0000001f, 0x00050081, 0x00000007, 0x00000021, 0x0000001e,
   0x00000020, 0x0004006d, 0x00000019, 0x00000022, 0x00000021, 0x0003003e,
   0x0000001b, 0x00000022, 0x00050041, 0x00000026, 0x00000027, 0x0000001b,
   0x00000025, 0x0004003d, 0x00000018, 0x00000028, 0x00000027, 0x00050041,
   0x00000026, 0x0000002a, 0x0000001b, 0x00000029, 0x0004003d, 0x00000018,
   0x0000002b, 0x0000002a, 0x000500c4, 0x00000018, 0x0000002d, 0x0000002b,
   0x0000002c, 0x000500c5, 0x00000018, 0x0000002e, 0x00000028, 0x0000002d,
   0x00050041, 0x00000026, 0x00000030, 0x0000001b, 0x0000002f, 0x0004003d,
   0x00000018, 0x00000031, 0x00000030, 0x000500c4, 0x00000018, 0x00000033,
   0x00000031, 0x00000032, 0x000500c5, 0x00000018, 0x00000034, 0x0000002e,
   0x00000033, 0x00050041, 0x00000026, 0x00000036, 0x0000001b, 0x00000035,
   0x0004003d, 0x00000018, 0x00000037, 0x00000036, 0x000500c4, 0x00000018,
   0x00000039, 0x00000037, 0x00000038, 0x000500c5, 0x00000018, 0x0000003a,
   0x00000034, 0x00000039, 0x0004003d, 0x0000000a, 0x0000003c, 0x0000003b,
   0x00050062, 0x00000007, 0x0000003d, 0x0000003c, 0x00000011, 0x00050051,
   0x00000006, 0x0000003e, 0x0000003d, 0x00000000, 0x0004007c, 0x00000018,
   0x0000003f, 0x0000003e, 0x0004003d, 0x00000040, 0x00000043, 0x00000042,
   0x00050062, 0x00000019, 0x00000044, 0x00000043, 0x00000011, 0x00050051,
   0x00000018, 0x00000045, 0x00000044, 0x00000000, 0x00050041, 0x00000048,
   0x00000049, 0x00000047, 0x00000029, 0x0004003d, 0x00000006, 0x0000004a,
   0x00000049, 0x0004006d, 0x00000018, 0x0000004b, 0x0000004a, 0x000500c4,
   0x00000018, 0x0000004c, 0x0000004b, 0x00000032, 0x00050041, 0x00000048,
   0x0000004d, 0x00000047, 0x00000025, 0x0004003d, 0x00000006, 0x0000004e,
   0x0000004d, 0x0004006d, 0x00000018, 0x0000004f, 0x0000004e, 0x000500c5,
   0x00000018, 0x00000050, 0x0000004c, 0x0000004f, 0x00070050, 0x00000019,
   0x00000051, 0x0000003a, 0x0000003f, 0x00000045, 0x00000050, 0x0003003e,
   0x00000024, 0x00000051, 0x0004003d, 0x00000007, 0x00000055, 0x00000009,
   0x00050083, 0x00000007, 0x00000056, 0x00000054, 0x00000055, 0x0003003e,
   0x00000053, 0x00000056, 0x000100fd, 0x00010038,
};

/* ---------------------------------------------------------------------- */
/* Formats, modes, state                                                   */

enum ia_mode { MODE_IMAGE, MODE_TILE, MODE_FEEDBACK, MODE_COUNT };
static const char *const mode_names[MODE_COUNT] = {"image", "tile", "feedback"};

struct zs_format {
   VkFormat format;
   const char *name;
   bool d24;   /* depth stored as 24-bit UNORM, else as a 32-bit float */
};

static const struct zs_format zs_formats[] = {
   {VK_FORMAT_D24_UNORM_S8_UINT, "D24_UNORM_S8_UINT", true},
   {VK_FORMAT_D32_SFLOAT_S8_UINT, "D32_SFLOAT_S8_UINT", false},
};
#define ZS_FORMAT_COUNT (sizeof(zs_formats) / sizeof(zs_formats[0]))
/* Every run has its own triangle in the vertex buffer. */
_Static_assert(ZS_FORMAT_COUNT * MODE_COUNT * 2 <= MAX_RUNS, "MAX_RUNS too small");

/* Entry points beyond the harness's. */
struct ia_fns {
   PFN_vkCreateRenderPass CreateRenderPass;
   PFN_vkDestroyRenderPass DestroyRenderPass;
   PFN_vkCreateFramebuffer CreateFramebuffer;
   PFN_vkDestroyFramebuffer DestroyFramebuffer;
   PFN_vkCmdBeginRenderPass CmdBeginRenderPass;
   PFN_vkCmdNextSubpass CmdNextSubpass;
   PFN_vkCmdEndRenderPass CmdEndRenderPass;
   PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
   PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
   PFN_vkCreateDescriptorPool CreateDescriptorPool;
   PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
   PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
   PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
   PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
   PFN_vkCmdPushConstants CmdPushConstants;
   PFN_vkCmdPushDescriptorSetKHR CmdPushDescriptorSetKHR;   /* with VK_KHR_push_descriptor */
};

static struct g52 g;
static struct ia_fns ia;
static VkCommandPool pool;
static VkCommandBuffer cmd;
static struct g52_buffer vbuf;
/* Subpass 0's layout (the colour seed as a push constant), and subpass 1's
 * per descriptor mode: [0] a descriptor set, [1] a push descriptor set. */
static VkPipelineLayout write_layout;
static VkDescriptorSetLayout read_set_layout[2];
static VkPipelineLayout read_layout[2];

/* The images of one depth/stencil format, shared by its runs. */
struct ia_images {
   const struct zs_format *zf;
   struct g52_image color, zs, out;
   VkImageView depth_view, stencil_view;   /* for the input attachment descriptors */
};

/* ---------------------------------------------------------------------- */
/* What each run writes (see the file comment)                             */

static uint32_t
color_seed(unsigned run)
{
   return 17u * run + 3u;
}

static void
expect_color(unsigned px, unsigned py, unsigned run, uint8_t out[4])
{
   out[0] = (uint8_t)(px * 4 + 1);
   out[1] = (uint8_t)(py * 4 + 2);
   out[2] = (uint8_t)((px * 7 + py * 13 + color_seed(run)) & 255);
   out[3] = (uint8_t)(255 - ((px + py * 3) & 127));
}

static uint8_t
stencil_prefill(unsigned px, unsigned py, unsigned run)
{
   return (uint8_t)(px * 5 + py * 11 + run * 37 + 1);
}

/* Subpass 0 inverts the prefill. */
static uint8_t
expect_stencil(unsigned px, unsigned py, unsigned run)
{
   return (uint8_t)~stencil_prefill(px, py, run);
}

/* Dyadic, so the vertex buffer holds it exactly. */
static float
depth_z0(unsigned run)
{
   return 1.0f / 16.0f + (float)run / 64.0f;
}

static double
expect_depth(unsigned px, unsigned py, unsigned run)
{
   return (double)depth_z0(run) + (px + 0.5) / 256.0 + (py + 0.5) / 32768.0;
}

static float
bits_to_float(uint32_t u)
{
   float f;
   memcpy(&f, &u, sizeof(f));
   return f;
}

/* ---------------------------------------------------------------------- */
/* Objects                                                                 */

static bool
images_create(struct ia_images *im, const struct zs_format *zf)
{
   memset(im, 0, sizeof(*im));
   im->zf = zf;
   const VkImageUsageFlags read_back = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   const VkImageUsageFlags input = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   if (!g52_image_create(&g, &im->color, VK_FORMAT_R8G8B8A8_UNORM, IMG_W, IMG_H,
                         VK_IMAGE_TILING_OPTIMAL,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | input | read_back,
                         VK_IMAGE_ASPECT_COLOR_BIT, false, 0) ||
       !g52_image_create(&g, &im->zs, zf->format, IMG_W, IMG_H, VK_IMAGE_TILING_OPTIMAL,
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | input | read_back |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, false, 0) ||
       !g52_image_create(&g, &im->out, VK_FORMAT_R32G32B32A32_UINT, IMG_W, IMG_H,
                         VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | read_back,
                         VK_IMAGE_ASPECT_COLOR_BIT, false, 0))
      return false;

   /* A depth/stencil input attachment descriptor names one aspect. */
   const VkImageAspectFlags aspects[2] = {VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_STENCIL_BIT};
   VkImageView *views[2] = {&im->depth_view, &im->stencil_view};
   for (unsigned i = 0; i < 2; i++) {
      const VkImageViewCreateInfo vci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = im->zs.image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = zf->format,
         .subresourceRange = {aspects[i], 0, 1, 0, 1},
      };
      const VkResult r = g.f.CreateImageView(g.device, &vci, NULL, views[i]);
      if (r != VK_SUCCESS) {
         printf("  vkCreateImageView (%s, %s only): %s\n", zf->name, i ? "stencil" : "depth",
                g52_result_str(r));
         return false;
      }
   }
   return true;
}

static void
images_destroy(struct ia_images *im)
{
   if (im->depth_view)
      g.f.DestroyImageView(g.device, im->depth_view, NULL);
   if (im->stencil_view)
      g.f.DestroyImageView(g.device, im->stencil_view, NULL);
   g52_image_destroy(&g, &im->color);
   g52_image_destroy(&g, &im->zs);
   g52_image_destroy(&g, &im->out);
   memset(im, 0, sizeof(*im));
}

/*
 * The pass. Attachments: 0 colour, 1 depth/stencil, 2 output. Subpass 0
 * writes 0 and 1; subpass 1 reads 0 (input attachment 0) and 1 (input
 * attachment 1, depth and stencil) and writes 2, and in the tile and
 * feedback modes also has 0 as its colour attachment 1 and 1 as its
 * read-only depth/stencil attachment (both GENERAL / read-only layouts, as
 * an attachment that is an input attachment of the same subpass must be).
 */
static VkRenderPass
make_render_pass(VkFormat zs_format, enum ia_mode mode)
{
   const bool in_pass = mode != MODE_IMAGE;
   const VkAttachmentDescription atts[3] = {
      {
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      },
      {
         .format = zs_format,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         /* The prefill copied in before the pass. */
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
         .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      },
      {
         .format = VK_FORMAT_R32G32B32A32_UINT,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      },
   };
   const VkAttachmentReference sp0_color = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   const VkAttachmentReference sp0_zs = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
   const VkImageLayout color_in = in_pass ? VK_IMAGE_LAYOUT_GENERAL
                                          : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   const VkAttachmentReference sp1_inputs[2] = {
      {0, color_in},
      {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
   };
   const VkAttachmentReference sp1_colors[2] = {
      {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
      {0, VK_IMAGE_LAYOUT_GENERAL},
   };
   const VkAttachmentReference sp1_zs = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
   const VkSubpassDescription subpasses[2] = {
      {
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
         .colorAttachmentCount = 1,
         .pColorAttachments = &sp0_color,
         .pDepthStencilAttachment = &sp0_zs,
      },
      {
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
         .inputAttachmentCount = 2,
         .pInputAttachments = sp1_inputs,
         .colorAttachmentCount = in_pass ? 2 : 1,
         .pColorAttachments = sp1_colors,
         .pDepthStencilAttachment = in_pass ? &sp1_zs : NULL,
      },
   };
   /* Input attachment 1 is read as depth and as stencil. */
   const VkInputAttachmentAspectReference aspect_refs[2] = {
      {1, 0, VK_IMAGE_ASPECT_COLOR_BIT},
      {1, 1, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT},
   };
   const VkRenderPassInputAttachmentAspectCreateInfo aspect_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO,
      .aspectReferenceCount = 2,
      .pAspectReferences = aspect_refs,
   };
   const VkPipelineStageFlags frag_tests = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
   const VkSubpassDependency deps[3] = {
      {
         /* The stencil prefill before the stencil LOAD. */
         .srcSubpass = VK_SUBPASS_EXTERNAL,
         .dstSubpass = 0,
         .srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .dstStageMask = frag_tests | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      },
      {
         /* Subpass 0's writes before subpass 1's reads (and its tile
          * loads in the tile and feedback modes). */
         .srcSubpass = 0,
         .dstSubpass = 1,
         .srcStageMask = frag_tests | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | frag_tests |
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT |
                          VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
         .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      },
      {
         /* Everything before the readback copies. */
         .srcSubpass = 1,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = frag_tests | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      },
   };
   const VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .pNext = &aspect_info,
      .attachmentCount = 3,
      .pAttachments = atts,
      .subpassCount = 2,
      .pSubpasses = subpasses,
      .dependencyCount = 3,
      .pDependencies = deps,
   };
   VkRenderPass rp = VK_NULL_HANDLE;
   const VkResult r = ia.CreateRenderPass(g.device, &rpci, NULL, &rp);
   if (r != VK_SUCCESS) {
      printf("  vkCreateRenderPass: %s\n", g52_result_str(r));
      return VK_NULL_HANDLE;
   }
   return rp;
}

/* A pipeline for one subpass of `rp`: pos3.vert, the given fragment
 * shader, triangle list, no culling, static viewport and scissor (the whole
 * image), `color_count` colour attachments. Prints the creation time. */
static VkPipeline
make_pipeline(VkRenderPass rp, uint32_t subpass, VkPipelineLayout layout, const uint32_t *fs_spv,
              size_t fs_size, uint32_t color_count,
              const VkPipelineColorBlendAttachmentState *blends,
              const VkPipelineDepthStencilStateCreateInfo *ds, const char *what)
{
   const VkShaderModuleCreateInfo vsmci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(pos3_vert_spv),
      .pCode = pos3_vert_spv,
   };
   const VkShaderModuleCreateInfo fsmci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = fs_size,
      .pCode = fs_spv,
   };
   VkShaderModule vsm = VK_NULL_HANDLE, fsm = VK_NULL_HANDLE;
   VkResult r = g.f.CreateShaderModule(g.device, &vsmci, NULL, &vsm);
   if (r == VK_SUCCESS)
      r = g.f.CreateShaderModule(g.device, &fsmci, NULL, &fsm);
   if (r != VK_SUCCESS) {
      printf("  vkCreateShaderModule (%s): %s\n", what, g52_result_str(r));
      if (vsm)
         g.f.DestroyShaderModule(g.device, vsm, NULL);
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
      .stride = 3 * sizeof(float),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription va_desc = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32B32_SFLOAT,
      .offset = 0,
   };
   const VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vb_desc,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &va_desc,
   };
   const VkPipelineInputAssemblyStateCreateInfo ias = {
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
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = color_count,
      .pAttachments = blends,
   };
   const VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ias,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pDepthStencilState = ds,
      .pColorBlendState = &cb,
      .layout = layout,
      .renderPass = rp,
      .subpass = subpass,
   };
   VkPipeline pipe = VK_NULL_HANDLE;
   const int64_t t0 = g52_now_ns();
   r = g.f.CreateGraphicsPipelines(g.device, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe);
   const int64_t t1 = g52_now_ns();
   g.f.DestroyShaderModule(g.device, vsm, NULL);
   g.f.DestroyShaderModule(g.device, fsm, NULL);
   if (r != VK_SUCCESS) {
      printf("  vkCreateGraphicsPipelines (%s): %s (the Bifrost compile or a TODO v7 refusal; "
             "see the log)\n", what, g52_result_str(r));
      return VK_NULL_HANDLE;
   }
   printf("  pipeline %s created in %.2f ms\n", what, (double)(t1 - t0) / 1e6);
   return pipe;
}

/* The three input attachment descriptors of subpass 1: the colour image,
 * the depth-only view, the stencil-only view, in the layouts the subpass
 * uses them in. `set` is ignored by a push. */
static void
input_writes(const struct ia_images *im, enum ia_mode mode, VkDescriptorSet set,
             VkDescriptorImageInfo info[3], VkWriteDescriptorSet w[3])
{
   const VkImageLayout zs_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
   info[0] = (VkDescriptorImageInfo){
      VK_NULL_HANDLE, im->color.view,
      mode == MODE_IMAGE ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL,
   };
   info[1] = (VkDescriptorImageInfo){VK_NULL_HANDLE, im->depth_view, zs_layout};
   info[2] = (VkDescriptorImageInfo){VK_NULL_HANDLE, im->stencil_view, zs_layout};
   for (unsigned b = 0; b < 3; b++) {
      w[b] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = b,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
         .pImageInfo = &info[b],
      };
   }
}

static bool
begin(void)
{
   VkResult r = g.f.ResetCommandPool(g.device, pool, 0);
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   if (r == VK_SUCCESS)
      r = g.f.BeginCommandBuffer(cmd, &bi);
   if (r != VK_SUCCESS)
      printf("  vkBeginCommandBuffer: %s\n", g52_result_str(r));
   return r == VK_SUCCESS;
}

static void
transition(VkImage image, VkImageAspectFlags aspects, VkImageLayout from, VkImageLayout to)
{
   g52_cmd_transition(&g, cmd, image, aspects, from, to, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}

/* ---------------------------------------------------------------------- */
/* The verdict                                                             */

struct check {
   const char *what;
   unsigned bad, off1, shown;
};

static void
check_fail(struct check *k, unsigned px, unsigned py, const char *got, const char *want)
{
   k->bad++;
   if (k->shown++ < SHOW)
      printf("    %s (%u, %u): got %s, want %s\n", k->what, px, py, got, want);
}

/* Two RGBA8 pixels: equal, within `tolerance` in every channel (counted
 * apart in off1: dithering), or wrong. */
static void
check_rgba(struct check *k, unsigned px, unsigned py, const uint8_t *got, const uint8_t *want,
           int tolerance)
{
   int worst = 0;
   for (int c = 0; c < 4; c++) {
      const int d = abs((int)got[c] - (int)want[c]);
      worst = d > worst ? d : worst;
   }
   if (worst == 0)
      return;
   if (worst <= tolerance) {
      k->off1++;
      return;
   }
   char gs[32], ws[32];
   snprintf(gs, sizeof(gs), "%u %u %u %u", got[0], got[1], got[2], got[3]);
   snprintf(ws, sizeof(ws), "%u %u %u %u", want[0], want[1], want[2], want[3]);
   check_fail(k, px, py, gs, ws);
}

/* Every pixel of the four readbacks (see the file comment). */
static bool
check_run(const struct ia_images *im, enum ia_mode mode, unsigned run, const char *name,
          const struct g52_hostbuf *rb_c, const struct g52_hostbuf *rb_z,
          const struct g52_hostbuf *rb_s, const struct g52_hostbuf *rb_o)
{
   const bool d24 = im->zf->d24;
   struct check marker = {"output"}, c_read = {"colour read"}, c_image = {"colour image"};
   struct check z_read = {"depth read"}, z_image = {"depth image"};
   struct check s_read = {"stencil read"}, s_image = {"stencil image"};
   unsigned out_clear = 0;
   double z_read_max = 0.0;   /* D24: the largest |read - image| passed, in 24-bit steps */
   char gs[64], ws[64];

   for (unsigned py = 0; py < IMG_H; py++) {
      for (unsigned px = 0; px < IMG_W; px++) {
         const unsigned i = py * IMG_W + px;
         const uint32_t *o = (const uint32_t *)rb_o->map + (size_t)i * 4;
         const uint8_t *sc = (const uint8_t *)rb_c->map + (size_t)i * 4;
         const uint32_t zraw = ((const uint32_t *)rb_z->map)[i];
         const uint8_t ss = ((const uint8_t *)rb_s->map)[i];
         const uint8_t ic[4] = {(uint8_t)o[0], (uint8_t)(o[0] >> 8), (uint8_t)(o[0] >> 16),
                                (uint8_t)(o[0] >> 24)};
         uint8_t ec[4];
         expect_color(px, py, run, ec);

         /* What the images hold: subpass 0's writes, and in the feedback
          * mode the colour subpass 1 wrote back. */
         if (mode == MODE_FEEDBACK) {
            const uint8_t inv[4] = {(uint8_t)(255 - ic[0]), (uint8_t)(255 - ic[1]),
                                    (uint8_t)(255 - ic[2]), (uint8_t)(255 - ic[3])};
            check_rgba(&c_image, px, py, sc, inv, 1);
         } else {
            check_rgba(&c_image, px, py, sc, ec, 1);
         }
         const double zimg = d24 ? (double)(zraw & 0xffffff) / 16777215.0
                                 : (double)bits_to_float(zraw);
         const double ez = expect_depth(px, py, run);
         if (!(fabs(zimg - ez) <= DEPTH_MODEL_TOL)) {
            snprintf(gs, sizeof(gs), "%.8f (0x%08x)", zimg, zraw);
            snprintf(ws, sizeof(ws), "%.8f", ez);
            check_fail(&z_image, px, py, gs, ws);
         }
         const uint8_t es = expect_stencil(px, py, run);
         if (ss != es) {
            snprintf(gs, sizeof(gs), "%u", ss);
            snprintf(ws, sizeof(ws), "%u", es);
            check_fail(&s_image, px, py, gs, ws);
         }

         /* The reads, from the output, once it is this fragment's. */
         const uint32_t marker_want = (py << 16) | px;
         if (o[3] != marker_want) {
            out_clear += o[0] == OUT_CLEAR && o[1] == OUT_CLEAR && o[2] == OUT_CLEAR &&
                         o[3] == OUT_CLEAR_W(run);
            snprintf(gs, sizeof(gs), "%08x %08x %08x %08x", o[0], o[1], o[2], o[3]);
            snprintf(ws, sizeof(ws), "w = %08x", marker_want);
            check_fail(&marker, px, py, gs, ws);
            continue;
         }
         if (mode == MODE_FEEDBACK)
            check_rgba(&c_read, px, py, ic, ec, 1);
         else
            check_rgba(&c_read, px, py, ic, sc, 0);

         const float zr = bits_to_float(o[1]);
         if (d24) {
            const double steps = fabs((double)zr * 16777215.0 - (double)(zraw & 0xffffff));
            if (steps <= 1.0) {
               z_read_max = steps > z_read_max ? steps : z_read_max;
            } else {
               snprintf(gs, sizeof(gs), "%.8f (0x%08x)", (double)zr, o[1]);
               snprintf(ws, sizeof(ws), "%.8f (D24 0x%06x)", zimg, zraw & 0xffffff);
               check_fail(&z_read, px, py, gs, ws);
            }
         } else if (o[1] != zraw) {
            snprintf(gs, sizeof(gs), "%.8f (0x%08x)", (double)zr, o[1]);
            snprintf(ws, sizeof(ws), "%.8f (0x%08x)", zimg, zraw);
            check_fail(&z_read, px, py, gs, ws);
         }

         if (o[2] != ss) {
            snprintf(gs, sizeof(gs), "%u", o[2]);
            snprintf(ws, sizeof(ws), "%u", ss);
            check_fail(&s_read, px, py, gs, ws);
         }
      }
   }

   printf("  output: %u of %u pixels not written by their own fragment (%u still the clear "
          "value)\n", marker.bad, NPIX, out_clear);
   printf("  colour: %u reads wrong%s (%u off by one), %u image pixels wrong (%u off by one)\n",
          c_read.bad, mode == MODE_FEEDBACK ? " against the pattern" : "", c_read.off1,
          c_image.bad, c_image.off1);
   if (d24)
      printf("  depth: %u reads wrong (largest accepted difference %.3f of a 24-bit step), %u "
             "image pixels off the gradient\n", z_read.bad, z_read_max, z_image.bad);
   else
      printf("  depth: %u reads wrong (bit-exact), %u image pixels off the gradient\n",
             z_read.bad, z_image.bad);
   printf("  stencil: %u reads wrong, %u image pixels wrong\n", s_read.bad, s_image.bad);

   const bool tiled = mode != MODE_IMAGE;
   if (marker.bad == NPIX)
      printf("  subpass 1 wrote no output%s: its rendering (vkCmdNextSubpass), its pipeline or "
             "its descriptors\n", out_clear == NPIX ? " (all the clear value)" : "");
   if (c_read.bad)
      printf("  the colour read %s\n",
             tiled ? "is a tile read: LD_TILE with the colour conversion from the iam system "
                     "values (mali_cmd_state.h, mali_nir_lower_input_attachments.c)"
                   : "goes through its texture descriptor: input attachments in the v7 texture "
                     "table (mali_nir_lower_descriptors_v7.c)");
   if (z_read.bad)
      printf("  the depth read %s\n",
             tiled ? "is a depth tile read: v7 needs the depth format's conversion word "
                     "(iam in mali_cmd_state.h)"
                   : "goes through the depth-only view's texture (mali_image_view.c)");
   if (s_read.bad)
      printf("  the stencil read %s\n",
             tiled ? "is a stencil tile read: the MALI_V7_STENCIL_IA_CONV constant "
                     "(mali_shader.c) and the 8-bit mask"
                   : "goes through the stencil-only view's texture (the S8 plane, "
                     "mali_image_view.c)");
   if (s_image.bad == NPIX)
      printf("  no stencil pixel holds the inverted prefill: the prefill copy, subpass 0's "
             "stencil LOAD or its INVERT\n");
   if ((c_image.bad || z_image.bad || s_image.bad) && tiled)
      printf("  in the %s mode subpass 1 has these images as attachments too: a wrong image "
             "can also be its writeback\n", mode_names[mode]);

   const bool pass = !marker.bad && !c_read.bad && !c_image.bad && !z_read.bad &&
                     !z_image.bad && !s_read.bad && !s_image.bad;
   printf("%s %s\n", pass ? "PASS" : "FAIL", name);
   return pass;
}

/* ---------------------------------------------------------------------- */
/* One run                                                                 */

/* Set when a submit timed out or lost the device: nothing more is run, and
 * nothing waits on the device again (a hung job would hang the test). */
static bool device_broken;

/* What one run creates (run_objs_destroy). */
struct run_objs {
   VkRenderPass rp;
   VkFramebuffer fb;
   VkPipeline write_pipe, read_pipe;
   VkDescriptorPool dpool;
   VkDescriptorSet set;
   /* The stencil prefill; the colour, depth, stencil and output readbacks. */
   struct g52_hostbuf up_s, rb_c, rb_z, rb_s, rb_o;
};

/* The pass, its framebuffer, both pipelines, the descriptor set (set
 * mode), the prefill and the readback buffers. */
static bool
run_setup(const struct ia_images *im, enum ia_mode mode, bool push, unsigned run,
          struct run_objs *o)
{
   o->rp = make_render_pass(im->zf->format, mode);
   if (!o->rp)
      return false;
   const VkImageView fb_views[3] = {im->color.view, im->zs.view, im->out.view};
   const VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = o->rp,
      .attachmentCount = 3,
      .pAttachments = fb_views,
      .width = IMG_W,
      .height = IMG_H,
      .layers = 1,
   };
   VkResult r = ia.CreateFramebuffer(g.device, &fbci, NULL, &o->fb);
   if (r != VK_SUCCESS) {
      printf("  vkCreateFramebuffer: %s\n", g52_result_str(r));
      return false;
   }

   /* Subpass 0: colour, depth ALWAYS with writes, stencil ALWAYS / INVERT. */
   const VkPipelineColorBlendAttachmentState write_all = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendAttachmentState write_none = {.colorWriteMask = 0};
   const VkStencilOpState invert = {
      .failOp = VK_STENCIL_OP_KEEP,
      .passOp = VK_STENCIL_OP_INVERT,
      .depthFailOp = VK_STENCIL_OP_KEEP,
      .compareOp = VK_COMPARE_OP_ALWAYS,
      .compareMask = 0xff,
      .writeMask = 0xff,
      .reference = 0,
   };
   const VkPipelineDepthStencilStateCreateInfo ds_write = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_ALWAYS,
      .stencilTestEnable = VK_TRUE,
      .front = invert,
      .back = invert,
   };
   /* Subpass 1: no depth or stencil test, nothing written there. */
   const VkPipelineDepthStencilStateCreateInfo ds_off = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
   };
   o->write_pipe = make_pipeline(o->rp, 0, write_layout, ia_write_frag_spv,
                                 sizeof(ia_write_frag_spv), 1, &write_all, &ds_write,
                                 "subpass 0 (colour, depth ALWAYS + write, stencil INVERT)");
   const VkPipelineColorBlendAttachmentState read_blends[2] = {
      write_all, mode == MODE_TILE ? write_none : write_all,
   };
   const bool feedback = mode == MODE_FEEDBACK;
   o->read_pipe = make_pipeline(o->rp, 1, read_layout[push ? 1 : 0],
                                feedback ? ia_feedback_frag_spv : ia_read_frag_spv,
                                feedback ? sizeof(ia_feedback_frag_spv) : sizeof(ia_read_frag_spv),
                                mode == MODE_IMAGE ? 1 : 2, read_blends, &ds_off,
                                mode == MODE_IMAGE ? "subpass 1 (three subpassLoads)" :
                                mode == MODE_TILE  ? "subpass 1 (three subpassLoads, colour "
                                                     "attachment 1 read-only)"
                                                   : "subpass 1 (three subpassLoads, colour "
                                                     "attachment 1 written back)");
   if (!o->write_pipe || !o->read_pipe)
      return false;

   if (!push) {
      const VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 3};
      const VkDescriptorPoolCreateInfo dpci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = 1,
         .poolSizeCount = 1,
         .pPoolSizes = &ps,
      };
      r = ia.CreateDescriptorPool(g.device, &dpci, NULL, &o->dpool);
      const VkDescriptorSetAllocateInfo dsai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = o->dpool,
         .descriptorSetCount = 1,
         .pSetLayouts = &read_set_layout[0],
      };
      if (r == VK_SUCCESS)
         r = ia.AllocateDescriptorSets(g.device, &dsai, &o->set);
      if (r != VK_SUCCESS) {
         printf("  descriptor pool or set: %s\n", g52_result_str(r));
         return false;
      }
      VkDescriptorImageInfo info[3];
      VkWriteDescriptorSet writes[3];
      input_writes(im, mode, o->set, info, writes);
      ia.UpdateDescriptorSets(g.device, 3, writes, 0, NULL);
   }

   if (!g52_hostbuf_create(&g, &o->up_s, NPIX, 0) ||
       !g52_hostbuf_create(&g, &o->rb_c, NPIX * 4, RB_SENTINEL) ||
       !g52_hostbuf_create(&g, &o->rb_z, NPIX * 4, RB_SENTINEL) ||
       !g52_hostbuf_create(&g, &o->rb_s, NPIX, RB_SENTINEL) ||
       !g52_hostbuf_create(&g, &o->rb_o, NPIX * 16, RB_SENTINEL))
      return false;
   for (unsigned py = 0; py < IMG_H; py++) {
      for (unsigned px = 0; px < IMG_W; px++)
         ((uint8_t *)o->up_s.map)[py * IMG_W + px] = stencil_prefill(px, py, run);
   }
   g52_hostbuf_flush(&g, &o->up_s);
   return true;
}

/* Records the stencil prefill, the pass (one draw per subpass) and the four
 * readbacks. */
static bool
run_record(const struct ia_images *im, enum ia_mode mode, bool push, unsigned run,
           const struct run_objs *o)
{
   const VkImageAspectFlags zs_aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
   const VkPipelineLayout rlayout = read_layout[push ? 1 : 0];
   if (!begin())
      return false;
   transition(im->zs.image, zs_aspects, VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
   g52_cmd_copy_image(&g, cmd, im->zs.image, VK_IMAGE_ASPECT_STENCIL_BIT, IMG_W, IMG_H,
                      o->up_s.buffer, true);
   transition(im->zs.image, zs_aspects, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

   const VkClearValue clears[3] = {
      {.color.float32 = {0.0f, 0.0f, 0.0f, 0.0f}},
      {.depthStencil = {1.0f, 0}},
      {.color.uint32 = {OUT_CLEAR, OUT_CLEAR, OUT_CLEAR, OUT_CLEAR_W(run)}},
   };
   const VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = o->rp,
      .framebuffer = o->fb,
      .renderArea = {{0, 0}, {IMG_W, IMG_H}},
      .clearValueCount = 3,
      .pClearValues = clears,
   };
   const VkDeviceSize vb_offset = (VkDeviceSize)run * 9 * sizeof(float);
   const uint32_t seed = color_seed(run);
   ia.CmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   g.f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, o->write_pipe);
   ia.CmdPushConstants(cmd, write_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(seed), &seed);
   g.f.CmdBindVertexBuffers(cmd, 0, 1, &vbuf.buffer, &vb_offset);
   g.f.CmdDraw(cmd, 3, 1, 0, 0);

   ia.CmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
   g.f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, o->read_pipe);
   if (push) {
      /* Inside the subpass, after the pipeline, as ARMSX2 pushes its set. */
      VkDescriptorImageInfo info[3];
      VkWriteDescriptorSet writes[3];
      input_writes(im, mode, VK_NULL_HANDLE, info, writes);
      ia.CmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rlayout, 0, 3, writes);
   } else {
      ia.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rlayout, 0, 1, &o->set, 0,
                               NULL);
   }
   g.f.CmdBindVertexBuffers(cmd, 0, 1, &vbuf.buffer, &vb_offset);
   g.f.CmdDraw(cmd, 3, 1, 0, 0);
   ia.CmdEndRenderPass(cmd);

   /* The pass leaves every attachment in TRANSFER_SRC_OPTIMAL. */
   g52_cmd_full_barrier(&g, cmd);
   g52_cmd_copy_image(&g, cmd, im->color.image, VK_IMAGE_ASPECT_COLOR_BIT, IMG_W, IMG_H,
                      o->rb_c.buffer, false);
   g52_cmd_copy_image(&g, cmd, im->zs.image, VK_IMAGE_ASPECT_DEPTH_BIT, IMG_W, IMG_H,
                      o->rb_z.buffer, false);
   g52_cmd_copy_image(&g, cmd, im->zs.image, VK_IMAGE_ASPECT_STENCIL_BIT, IMG_W, IMG_H,
                      o->rb_s.buffer, false);
   g52_cmd_copy_image(&g, cmd, im->out.image, VK_IMAGE_ASPECT_COLOR_BIT, IMG_W, IMG_H,
                      o->rb_o.buffer, false);
   const VkResult r = g.f.EndCommandBuffer(cmd);
   if (r != VK_SUCCESS) {
      printf("  vkEndCommandBuffer: %s (recording failed: a TODO v7 refusal, the frame "
             "shaders or command memory; see the log)\n", g52_result_str(r));
      return false;
   }
   printf("  command buffer recorded (stencil prefill, two subpasses of one draw each, four "
          "readbacks)\n");
   return true;
}

static void
run_objs_destroy(struct run_objs *o)
{
   /* After a timeout the GPU may still use them: left to the process exit. */
   if (device_broken)
      return;
   g.f.DeviceWaitIdle(g.device);
   g52_hostbuf_destroy(&g, &o->up_s);
   g52_hostbuf_destroy(&g, &o->rb_c);
   g52_hostbuf_destroy(&g, &o->rb_z);
   g52_hostbuf_destroy(&g, &o->rb_s);
   g52_hostbuf_destroy(&g, &o->rb_o);
   if (o->dpool)
      ia.DestroyDescriptorPool(g.device, o->dpool, NULL);
   if (o->write_pipe)
      g.f.DestroyPipeline(g.device, o->write_pipe, NULL);
   if (o->read_pipe)
      g.f.DestroyPipeline(g.device, o->read_pipe, NULL);
   if (o->fb)
      ia.DestroyFramebuffer(g.device, o->fb, NULL);
   if (o->rp)
      ia.DestroyRenderPass(g.device, o->rp, NULL);
   memset(o, 0, sizeof(*o));
}

static bool
run_one(const struct ia_images *im, enum ia_mode mode, bool push, unsigned run)
{
   char name[96];
   snprintf(name, sizeof(name), "%s, %s reads, %s", im->zf->name, mode_names[mode],
            push ? "push descriptors" : "descriptor set");
   printf("run %u: %s\n", run, name);

   struct run_objs o;
   memset(&o, 0, sizeof(o));
   bool ok = false;
   bool ran = run_setup(im, mode, push, run, &o) && run_record(im, mode, push, run, &o);
   if (ran) {
      const VkResult r = g52_submit_wait(&g, cmd, name);
      device_broken = r == VK_TIMEOUT || r == VK_ERROR_DEVICE_LOST;
      ran = r == VK_SUCCESS;
   }
   if (ran) {
      g52_hostbuf_invalidate(&g, &o.rb_c);
      g52_hostbuf_invalidate(&g, &o.rb_z);
      g52_hostbuf_invalidate(&g, &o.rb_s);
      g52_hostbuf_invalidate(&g, &o.rb_o);
      ok = check_run(im, mode, run, name, &o.rb_c, &o.rb_z, &o.rb_s, &o.rb_o);
   } else {
      printf("FAIL %s: not run to the end\n", name);
   }
   run_objs_destroy(&o);
   return ok;
}

/* ---------------------------------------------------------------------- */
/* Shared objects                                                          */

/* The colour formats are rendered to and copied out. */
static bool
color_formats_offered(void)
{
   static const VkFormat formats[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32G32B32A32_UINT};
   const VkFormatFeatureFlags want = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                     VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
   for (unsigned i = 0; i < 2; i++) {
      VkFormatProperties fp;
      g.f.GetPhysicalDeviceFormatProperties(g.pdev, formats[i], &fp);
      if ((fp.optimalTilingFeatures & want) != want) {
         printf("FAIL: format %d optimal features 0x%x lack 0x%x\n", (int)formats[i],
                fp.optimalTilingFeatures, want);
         return false;
      }
   }
   return true;
}

/* The command pool and buffer, the pipeline and set layouts (the push set
 * layout only with VK_KHR_push_descriptor) and the vertex buffer. */
static bool
shared_setup(bool have_push)
{
   const VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkResult r = g.f.CreateCommandPool(g.device, &cpci, NULL, &pool);
   const VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (r == VK_SUCCESS)
      r = g.f.AllocateCommandBuffers(g.device, &cbai, &cmd);
   const VkPushConstantRange seed_range = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(uint32_t)};
   const VkPipelineLayoutCreateInfo wplci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &seed_range,
   };
   if (r == VK_SUCCESS)
      r = g.f.CreatePipelineLayout(g.device, &wplci, NULL, &write_layout);
   const VkDescriptorSetLayoutBinding bindings[3] = {
      {0, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL},
      {2, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL},
   };
   for (unsigned p = 0; p < (have_push ? 2u : 1u) && r == VK_SUCCESS; p++) {
      const VkDescriptorSetLayoutCreateInfo dslci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .flags = p ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0,
         .bindingCount = 3,
         .pBindings = bindings,
      };
      r = ia.CreateDescriptorSetLayout(g.device, &dslci, NULL, &read_set_layout[p]);
      const VkPipelineLayoutCreateInfo rplci = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .setLayoutCount = 1,
         .pSetLayouts = &read_set_layout[p],
      };
      if (r == VK_SUCCESS)
         r = g.f.CreatePipelineLayout(g.device, &rplci, NULL, &read_layout[p]);
   }
   if (r != VK_SUCCESS) {
      printf("FAIL: command pool, set layouts or pipeline layouts: %s\n", g52_result_str(r));
      return false;
   }

   /* One full-screen triangle per run: (-1,-1, z0), (3,-1, z0 + 1/2),
    * (-1,3, z0 + 1/256), the plane z = z0 + u / 4 + v / 512 (u and v 0 to 1
    * across the image) of the file comment's gradient, inside [0, 1] at
    * every vertex so nothing is clipped. */
   static float verts[MAX_RUNS][9];
   for (unsigned i = 0; i < MAX_RUNS; i++) {
      const float z0 = depth_z0(i);
      const float v[9] = {-1.0f, -1.0f, z0, 3.0f, -1.0f, z0 + 0.5f,
                          -1.0f, 3.0f, z0 + 1.0f / 256.0f};
      memcpy(verts[i], v, sizeof(v));
   }
   return g52_buffer_create(&g, &vbuf, verts, sizeof(verts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
}

static void
shared_destroy(void)
{
   g52_buffer_destroy(&g, &vbuf);
   for (unsigned p = 0; p < 2; p++) {
      if (read_layout[p])
         g.f.DestroyPipelineLayout(g.device, read_layout[p], NULL);
      if (read_set_layout[p])
         ia.DestroyDescriptorSetLayout(g.device, read_set_layout[p], NULL);
   }
   if (write_layout)
      g.f.DestroyPipelineLayout(g.device, write_layout, NULL);
   if (pool)
      g.f.DestroyCommandPool(g.device, pool, NULL);
}

/* ---------------------------------------------------------------------- */

enum run_state { RUN_NONE, RUN_PASS, RUN_FAIL, RUN_SKIP };

/* Which runs to make (the command line and what the device offers). */
struct plan {
   int only_mode;              /* -1: every read mode */
   int only_push;              /* -1: both descriptor modes, 0 sets, 1 push */
   bool have_push;
   bool have_independent_blend;
   uint32_t max_color_attachments;
};

/* Every run of one depth/stencil format; counts them into *ran and
 * *passed and records each in state[mode][push]. */
static void
run_format(const struct zs_format *zf, const struct plan *pl, unsigned *run, unsigned *ran,
           unsigned *passed, enum run_state state[MODE_COUNT][2])
{
   VkFormatProperties fp;
   g.f.GetPhysicalDeviceFormatProperties(g.pdev, zf->format, &fp);
   const VkFormatFeatureFlags zs_want = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                        VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   if ((fp.optimalTilingFeatures & zs_want) != zs_want) {
      printf("format %s: not offered as a depth/stencil attachment with copies (0x%x), "
             "skipped\n", zf->name, fp.optimalTilingFeatures);
      return;
   }
   printf("format %s:\n", zf->name);
   struct ia_images im;
   if (!images_create(&im, zf)) {
      printf("FAIL %s: images\n", zf->name);
      images_destroy(&im);
      for (unsigned m = 0; m < MODE_COUNT; m++)
         state[m][0] = state[m][1] = RUN_FAIL;
      (*ran)++;
      return;
   }
   for (unsigned m = 0; m < MODE_COUNT && !device_broken; m++) {
      for (unsigned p = 0; p < 2 && !device_broken; p++) {
         if ((pl->only_mode >= 0 && (unsigned)pl->only_mode != m) ||
             (pl->only_push >= 0 && (unsigned)pl->only_push != p))
            continue;
         const char *skip = NULL;
         if (p && !pl->have_push)
            skip = "VK_KHR_push_descriptor is not offered";
         else if (m != MODE_IMAGE && pl->max_color_attachments < 2)
            skip = "fewer than two colour attachments";
         else if (m == MODE_TILE && !pl->have_independent_blend)
            skip = "independentBlend is not offered";
         if (skip) {
            printf("SKIP %s, %s reads, %s: %s\n", zf->name, mode_names[m],
                   p ? "push descriptors" : "descriptor set", skip);
            state[m][p] = RUN_SKIP;
            continue;
         }
         const bool ok = run_one(&im, (enum ia_mode)m, p, (*run)++);
         state[m][p] = ok ? RUN_PASS : RUN_FAIL;
         (*ran)++;
         *passed += ok;
      }
   }
   if (!device_broken)
      images_destroy(&im);
}

int
main(int argc, char **argv)
{
   const char *icd = NULL;
   struct plan pl = {.only_mode = -1, .only_push = -1};
   for (int i = 1; i < argc; i++) {
      if (!strncmp(argv[i], "--only=", 7)) {
         pl.only_mode = MODE_COUNT;
         for (int m = 0; m < MODE_COUNT; m++) {
            if (!strcmp(argv[i] + 7, mode_names[m]))
               pl.only_mode = m;
         }
         if (pl.only_mode == MODE_COUNT) {
            fprintf(stderr, "ia_g52: unknown read mode \"%s\" (image, tile or feedback)\n",
                    argv[i] + 7);
            return 2;
         }
      } else if (!strcmp(argv[i], "--sets")) {
         pl.only_push = 0;
      } else if (!strcmp(argv[i], "--push")) {
         pl.only_push = 1;
      } else {
         icd = argv[i];
      }
   }
   char *args[2] = {argv[0], (char *)icd};
   const struct g52_device_wants wants = {
      .exts = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
      .ext_count = 1,
      /* The tile mode's two colour attachments have different write masks. */
      .features = {.independentBlend = VK_TRUE},
   };
   int rc = g52_open_ext(&g, "ia_g52", icd ? 2 : 1, args, IA_G52_DEFAULT_ICD, &wants);
   if (rc)
      return rc;
   pl.have_push = g.have_ext[0];
   pl.have_independent_blend = g.features.independentBlend;

   bool ok = true;
#define IA(name) ia.name = (PFN_vk##name)g52_load_device_fn(&g, "vk" #name, NULL, true, &ok)
   IA(CreateRenderPass);
   IA(DestroyRenderPass);
   IA(CreateFramebuffer);
   IA(DestroyFramebuffer);
   IA(CmdBeginRenderPass);
   IA(CmdNextSubpass);
   IA(CmdEndRenderPass);
   IA(CreateDescriptorSetLayout);
   IA(DestroyDescriptorSetLayout);
   IA(CreateDescriptorPool);
   IA(DestroyDescriptorPool);
   IA(AllocateDescriptorSets);
   IA(UpdateDescriptorSets);
   IA(CmdBindDescriptorSets);
   IA(CmdPushConstants);
#undef IA
   if (pl.have_push)
      ia.CmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)g52_load_device_fn(
         &g, "vkCmdPushDescriptorSetKHR", NULL, true, &ok);
   if (!g.f.CmdCopyBufferToImage || !g.f.CmdCopyImageToBuffer) {
      fprintf(stderr, "ia_g52: the driver lacks the buffer/image copies\n");
      ok = false;
   }
   if (!ok) {
      printf("FAIL: ia_g52: entry points missing\n");
      g52_close(&g);
      return 1;
   }

   VkPhysicalDeviceProperties2 props2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
   g.f.GetPhysicalDeviceProperties2(g.pdev, &props2);
   pl.max_color_attachments = props2.properties.limits.maxColorAttachments;
   printf("  maxColorAttachments %u, independentBlend %s, push descriptors %s\n",
          pl.max_color_attachments, pl.have_independent_blend ? "enabled" : "not offered",
          pl.have_push ? "enabled" : "not offered");

   unsigned run = 0, ran = 0, passed = 0;
   enum run_state state[ZS_FORMAT_COUNT][MODE_COUNT][2];
   memset(state, 0, sizeof(state));
   if (color_formats_offered() && shared_setup(pl.have_push)) {
      for (unsigned fi = 0; fi < ZS_FORMAT_COUNT && !device_broken; fi++)
         run_format(&zs_formats[fi], &pl, &run, &ran, &passed, state[fi]);
      if (device_broken)
         printf("the device timed out or was lost: the remaining runs are not made\n");

      static const char *const state_names[] = {"-", "PASS", "FAIL", "SKIP"};
      printf("summary (descriptor set / push descriptors):\n");
      bool push_only_fails = false;
      for (unsigned fi = 0; fi < ZS_FORMAT_COUNT; fi++) {
         printf("  %-20s", zs_formats[fi].name);
         for (unsigned m = 0; m < MODE_COUNT; m++) {
            printf("  %s %s/%s", mode_names[m], state_names[state[fi][m][0]],
                   state_names[state[fi][m][1]]);
            push_only_fails |= state[fi][m][0] == RUN_PASS && state[fi][m][1] == RUN_FAIL;
         }
         printf("\n");
      }
      if (push_only_fails)
         printf("  the reads fail only through pushed descriptors: vkCmdPushDescriptorSetKHR "
                "(mali_cmd_state.c, push_begin and the per-stage tables it marks dirty)\n");
   }
   const bool pass = ran && passed == ran && !device_broken;

   if (!device_broken) {
      shared_destroy();
      g52_close(&g);
   }
   if (pass)
      printf("PASS: ia_g52 (%u runs)\n", ran);
   else
      printf("FAIL: ia_g52: %u of %u runs passed%s\n", passed, ran,
             ran ? "" : " (nothing could be run)");
   return pass ? 0 : 1;
}
