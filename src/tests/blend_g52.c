/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * blend_g52: colour blending on the Bifrost (v7) path of this driver. Three
 * 64x64 linear RGBA8 images, each cleared to (0.5, 0.5, 0.5, 1.0) and drawn
 * once with one full-screen triangle, in one command buffer:
 *
 *   A. fixed-function: colour SRC_ALPHA / ONE_MINUS_SRC_ALPHA (add), alpha
 *      ONE / ZERO, write mask R, G and A (not B), fragment colour
 *      (0.25, 0.75, 1.0, 0.5): the v7 Blend descriptor in fixed-function
 *      mode with a partial colour mask (a tile-buffer read);
 *   B. blend shader: colour and alpha MIN (which the fixed-function blender
 *      cannot do), fragment colour (1.0, 0.25, 0.0, 1.0): the Bifrost blend
 *      shader, entered from the fragment shader's BLEND through the
 *      descriptor's PC and returning through its Return Value;
 *   C. fixed-function with the blend constant: CONSTANT_COLOR /
 *      ONE_MINUS_CONSTANT_COLOR and CONSTANT_ALPHA / ONE_MINUS_CONSTANT_ALPHA
 *      with the constant (0.5, 0.5, 0.5, 0.5), fragment colour
 *      (1.0, 0.0, 1.0, 0.0): the 16-bit UNORM blend constant of the
 *      descriptor.
 *
 * Each image is checked against the equation evaluated in floats and
 * rounded to 8 bits, one LSB of slack per channel (the fixed-function
 * blender works in the tile buffer's 8-bit precision and the render
 * target dithers).
 *
 * The shaders are SPIR-V embedded as arrays, generated with glslangValidator
 * 16.6.0 (`glslangValidator -V --target-env vulkan1.1`) from:
 *
 *   pos2.vert:  #version 450
 *               layout(location = 0) in vec2 pos;
 *               void main() { gl_Position = vec4(pos, 0.0, 1.0); }
 *
 *   src.frag, min.frag, cst.frag:
 *               #version 450
 *               layout(location = 0) out vec4 color;
 *               void main() { color = vec4(0.25, 0.75, 1.0, 0.5); }   (and so on)
 *
 * Usage: blend_g52 [path/to/libvulkan_malisx2.so]
 *   The path defaults to $MALISX2_ICD, then to BLEND_G52_DEFAULT_ICD.
 *
 * Exit status: 0 PASS; 1 FAIL (wrong pixels, a timeout or a Vulkan error
 * once the device exists, including a pipeline that would not compile); 2
 * the program did not get as far as a VkDevice.
 *
 * Driver log: MESA_LOG=file sends it to stderr, MESA_LOG_LEVEL=info adds
 * the atom layout and build lines; MALISX2_DUMP_DRAW=1 dumps every draw's
 * descriptors, the Blend descriptors included (see MALISX2_G52.md).
 */

#include "g52_harness.h"

#include <math.h>

#ifndef BLEND_G52_DEFAULT_ICD
#define BLEND_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define IMG_W 64
#define IMG_H 64
#define SENTINEL 0xa5

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

static const uint32_t src_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x0000000f, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
   0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
   0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63, 0x00000072,
   0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00020013, 0x00000002,
   0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
   0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008,
   0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003,
   0x0004002b, 0x00000006, 0x0000000a, 0x3e800000, 0x0004002b, 0x00000006,
   0x0000000b, 0x3f400000, 0x0004002b, 0x00000006, 0x0000000c, 0x3f800000,
   0x0004002b, 0x00000006, 0x0000000d, 0x3f000000, 0x0007002c, 0x00000007,
   0x0000000e, 0x0000000a, 0x0000000b, 0x0000000c, 0x0000000d, 0x00050036,
   0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
   0x0003003e, 0x00000009, 0x0000000e, 0x000100fd, 0x00010038,
};

static const uint32_t min_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x0000000e, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
   0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
   0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63, 0x00000072,
   0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00020013, 0x00000002,
   0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
   0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008,
   0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003,
   0x0004002b, 0x00000006, 0x0000000a, 0x3f800000, 0x0004002b, 0x00000006,
   0x0000000b, 0x3e800000, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000,
   0x0007002c, 0x00000007, 0x0000000d, 0x0000000a, 0x0000000b, 0x0000000c,
   0x0000000a, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
   0x000200f8, 0x00000005, 0x0003003e, 0x00000009, 0x0000000d, 0x000100fd,
   0x00010038,
};

static const uint32_t cst_frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x0000000d, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
   0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
   0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63, 0x00000072,
   0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00020013, 0x00000002,
   0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
   0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008,
   0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003,
   0x0004002b, 0x00000006, 0x0000000a, 0x3f800000, 0x0004002b, 0x00000006,
   0x0000000b, 0x00000000, 0x0007002c, 0x00000007, 0x0000000c, 0x0000000a,
   0x0000000b, 0x0000000a, 0x0000000b, 0x00050036, 0x00000002, 0x00000004,
   0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0003003e, 0x00000009,
   0x0000000c, 0x000100fd, 0x00010038,
};

/* A triangle covering clip space. */
static const float vertices[6] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};

static const float clear_rgba[4] = {0.5f, 0.5f, 0.5f, 1.0f};

struct blend_case {
   const char *name;
   const char *what;
   const uint32_t *fs_spv;
   size_t fs_size;
   float src[4];                 /* the fragment shader's colour */
   VkPipelineColorBlendAttachmentState blend;
   float constants[4];
   bool blend_shader;            /* expected to run as a blend shader */
};

#define MASK_RGBA                                                              \
   (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | \
    VK_COLOR_COMPONENT_A_BIT)

static const struct blend_case cases[3] = {
   {
      .name = "A",
      .what = "fixed-function SRC_ALPHA / ONE_MINUS_SRC_ALPHA, mask RGA",
      .fs_spv = src_frag_spv,
      .fs_size = sizeof(src_frag_spv),
      .src = {0.25f, 0.75f, 1.0f, 0.5f},
      .blend = {
         .blendEnable = VK_TRUE,
         .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
         .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
         .colorBlendOp = VK_BLEND_OP_ADD,
         .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
         .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
         .alphaBlendOp = VK_BLEND_OP_ADD,
         .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_A_BIT,
      },
   },
   {
      .name = "B",
      .what = "blend shader MIN (colour and alpha), mask RGBA",
      .fs_spv = min_frag_spv,
      .fs_size = sizeof(min_frag_spv),
      .src = {1.0f, 0.25f, 0.0f, 1.0f},
      .blend = {
         .blendEnable = VK_TRUE,
         .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
         .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
         .colorBlendOp = VK_BLEND_OP_MIN,
         .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
         .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
         .alphaBlendOp = VK_BLEND_OP_MIN,
         .colorWriteMask = MASK_RGBA,
      },
      .blend_shader = true,
   },
   {
      .name = "C",
      .what = "fixed-function CONSTANT / ONE_MINUS_CONSTANT with constant 0.5, mask RGBA",
      .fs_spv = cst_frag_spv,
      .fs_size = sizeof(cst_frag_spv),
      .src = {1.0f, 0.0f, 1.0f, 0.0f},
      .blend = {
         .blendEnable = VK_TRUE,
         .srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR,
         .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
         .colorBlendOp = VK_BLEND_OP_ADD,
         .srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA,
         .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
         .alphaBlendOp = VK_BLEND_OP_ADD,
         .colorWriteMask = MASK_RGBA,
      },
      .constants = {0.5f, 0.5f, 0.5f, 0.5f},
   },
};

static float
factor(VkBlendFactor f, unsigned c, const float *src, const float *dst, const float *cst)
{
   switch (f) {
   case VK_BLEND_FACTOR_ZERO: return 0.0f;
   case VK_BLEND_FACTOR_ONE: return 1.0f;
   case VK_BLEND_FACTOR_SRC_ALPHA: return src[3];
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return 1.0f - src[3];
   case VK_BLEND_FACTOR_CONSTANT_COLOR: return cst[c];
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return 1.0f - cst[c];
   case VK_BLEND_FACTOR_CONSTANT_ALPHA: return cst[3];
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return 1.0f - cst[3];
   default:
      fprintf(stderr, "blend_g52: factor %d not modelled\n", f);
      exit(1);
   }
}

/* The expected 8-bit result of a case: the equation in floats on the clear
 * colour, rounded to nearest. */
static void
expected(const struct blend_case *bc, uint8_t out[4])
{
   const VkPipelineColorBlendAttachmentState *b = &bc->blend;
   for (unsigned c = 0; c < 4; c++) {
      const float s = bc->src[c], d = clear_rgba[c];
      float v;
      const VkColorComponentFlags bit = 1u << c;
      if (!(b->colorWriteMask & bit)) {
         v = d;
      } else {
         const VkBlendOp op = c == 3 ? b->alphaBlendOp : b->colorBlendOp;
         const VkBlendFactor sf = c == 3 ? b->srcAlphaBlendFactor : b->srcColorBlendFactor;
         const VkBlendFactor df = c == 3 ? b->dstAlphaBlendFactor : b->dstColorBlendFactor;
         switch (op) {
         case VK_BLEND_OP_MIN: v = s < d ? s : d; break;
         case VK_BLEND_OP_MAX: v = s > d ? s : d; break;
         default:
            v = s * factor(sf, c, bc->src, clear_rgba, bc->constants) +
                d * factor(df, c, bc->src, clear_rgba, bc->constants);
            break;
         }
      }
      if (v < 0.0f)
         v = 0.0f;
      if (v > 1.0f)
         v = 1.0f;
      out[c] = (uint8_t)lrintf(v * 255.0f);
   }
}

int
main(int argc, char **argv)
{
   struct g52 g;
   int rc = g52_open(&g, "blend_g52", argc, argv, BLEND_G52_DEFAULT_ICD);
   if (rc)
      return rc;
   const struct g52_fns *f = &g.f;

   struct g52_image images[3];
   for (unsigned i = 0; i < 3; i++) {
      if (!g52_image_create(&g, &images[i], VK_FORMAT_R8G8B8A8_UNORM, IMG_W, IMG_H,
                            VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT, true, SENTINEL))
         return 1;
   }
   struct g52_buffer vbuf;
   if (!g52_buffer_create(&g, &vbuf, vertices, sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT))
      return 1;

   const VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout layout = VK_NULL_HANDLE;
   VkResult r = f->CreatePipelineLayout(g.device, &plci, NULL, &layout);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "blend_g52: vkCreatePipelineLayout failed: %s\n", g52_result_str(r));
      return 1;
   }

   VkPipeline pipes[3];
   for (unsigned i = 0; i < 3; i++) {
      const struct blend_case *bc = &cases[i];
      struct g52_pipeline_desc pd = {
         .vs_spv = pos2_vert_spv,
         .vs_size = sizeof(pos2_vert_spv),
         .fs_spv = bc->fs_spv,
         .fs_size = bc->fs_size,
         .attrib_format = VK_FORMAT_R32G32_SFLOAT,
         .vertex_stride = 2 * sizeof(float),
         .width = IMG_W,
         .height = IMG_H,
         .color_format = VK_FORMAT_R8G8B8A8_UNORM,
         .blend = bc->blend,
      };
      memcpy(pd.blend_constants, bc->constants, sizeof(pd.blend_constants));
      pipes[i] = g52_pipeline_create(&g, &pd, layout, bc->what);
      if (!pipes[i])
         return 1;
   }

   const VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkCommandPool pool = VK_NULL_HANDLE;
   r = f->CreateCommandPool(g.device, &cpci, NULL, &pool);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "blend_g52: vkCreateCommandPool failed: %s\n", g52_result_str(r));
      return 1;
   }
   const VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd = VK_NULL_HANDLE;
   r = f->AllocateCommandBuffers(g.device, &cbai, &cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "blend_g52: vkAllocateCommandBuffers failed: %s\n", g52_result_str(r));
      return 1;
   }
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   r = f->BeginCommandBuffer(cmd, &bi);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "blend_g52: vkBeginCommandBuffer failed: %s\n", g52_result_str(r));
      return 1;
   }

   const VkDeviceSize vb_offset = 0;
   for (unsigned i = 0; i < 3; i++) {
      g52_cmd_transition(&g, cmd, images[i].image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
      const VkRenderingAttachmentInfo color = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = images[i].view,
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
      f->CmdBeginRendering(cmd, &ri);
      f->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[i]);
      f->CmdBindVertexBuffers(cmd, 0, 1, &vbuf.buffer, &vb_offset);
      f->CmdDraw(cmd, 3, 1, 0, 0);
      f->CmdEndRendering(cmd);
      g52_cmd_transition(&g, cmd, images[i].image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_ACCESS_HOST_READ_BIT);
   }
   r = f->EndCommandBuffer(cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "blend_g52: vkEndCommandBuffer failed: %s (recording failed: the draw's "
              "descriptors, a blend shader that would not compile, a TODO v7 stub or command "
              "memory; see the log)\n", g52_result_str(r));
      return 1;
   }
   printf("  command buffer recorded (three passes of clear + one blended draw)\n");

   r = g52_submit_wait(&g, cmd, "(three vtc atoms, three fragment atoms)");
   if (r != VK_SUCCESS)
      return 1;

   unsigned passed = 0;
   for (unsigned i = 0; i < 3; i++) {
      const struct blend_case *bc = &cases[i];
      uint8_t want[4];
      expected(bc, want);
      g52_image_sync_for_read(&g, &images[i]);
      uint32_t bad = 0, sentinel = 0, unblended = 0;
      int first_x = -1, first_y = -1;
      uint8_t first_got[4] = {0};
      uint8_t src8[4];
      for (unsigned c = 0; c < 4; c++)
         src8[c] = (uint8_t)lrintf(bc->src[c] * 255.0f);
      for (uint32_t y = 0; y < IMG_H; y++) {
         for (uint32_t x = 0; x < IMG_W; x++) {
            const uint8_t *px = g52_image_pixel(&images[i], x, y);
            bool okpx = true;
            for (unsigned c = 0; c < 4; c++) {
               const int d = (int)px[c] - (int)want[c];
               if (d < -1 || d > 1)
                  okpx = false;
            }
            if (okpx)
               continue;
            if (px[0] == SENTINEL && px[1] == SENTINEL && px[2] == SENTINEL && px[3] == SENTINEL)
               sentinel++;
            else if (!memcmp(px, src8, 4))
               unblended++;
            if (!bad) {
               first_x = (int)x;
               first_y = (int)y;
               memcpy(first_got, px, 4);
            }
            bad++;
         }
      }
      const uint8_t *p0 = g52_image_pixel(&images[i], 0, 0);
      printf("  case %s (%s): pixel (0,0) = %u %u %u %u, expected %u %u %u %u\n", bc->name,
             bc->what, p0[0], p0[1], p0[2], p0[3], want[0], want[1], want[2], want[3]);
      if (bad == 0) {
         printf("  PASS %s\n", bc->name);
         passed++;
      } else {
         printf("  FAIL %s: %u of %u pixels wrong (%u still hold the sentinel, %u hold the "
                "unblended source); first at (%d,%d) = %u %u %u %u\n", bc->name, bad,
                IMG_W * IMG_H, sentinel, unblended, first_x, first_y, first_got[0], first_got[1],
                first_got[2], first_got[3]);
         if (unblended == IMG_W * IMG_H && bc->blend_shader)
            printf("    (every pixel is the source: the blend shader did not run, or its "
                   "fallback stored the source)\n");
      }
   }

   int status = passed == 3 ? 0 : 1;
   printf("%s: %u of 3 blend cases\n", status ? "FAIL" : "PASS", passed);

   f->DeviceWaitIdle(g.device);
   f->DestroyCommandPool(g.device, pool, NULL);
   for (unsigned i = 0; i < 3; i++)
      f->DestroyPipeline(g.device, pipes[i], NULL);
   f->DestroyPipelineLayout(g.device, layout, NULL);
   g52_buffer_destroy(&g, &vbuf);
   for (unsigned i = 0; i < 3; i++)
      g52_image_destroy(&g, &images[i]);
   g52_close(&g);
   return status;
}
