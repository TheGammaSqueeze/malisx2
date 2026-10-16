/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * zs_g52: depth and stencil on the Bifrost (v7) path of this driver. After
 * draw_g52 proved a draw, this one proves the depth/stencil attachment
 * (the ZS CRC extension of the framebuffer descriptor, its clear and
 * writeback), the depth test, write and clear values in the fragment
 * Renderer State Descriptor, and the stencil test, write and reference,
 * for each depth/stencil format the device offers for optimal tiling:
 * D24_UNORM_S8_UINT, D32_SFLOAT_S8_UINT, D32_SFLOAT and D16_UNORM.
 *
 * One 64x64 pass per format into a linear RGBA8 colour image (cleared to
 * black) with the depth/stencil image (depth cleared to 1.0, stencil to
 * 0), four draws of one triangle each, all with depth test LESS and depth
 * writes on where the depth test is used:
 *
 *   1. the near triangle (z = 0.25) over the right half (clip x >= 0),
 *      green;
 *   2. the far triangle (z = 0.75) over the whole image, red: it must lose
 *      the depth test where the near one was drawn, so without a working
 *      depth test the whole image would be red;
 *   3. (formats with stencil) a triangle over the top half (clip y <= 0)
 *      with the depth test off, stencil ALWAYS / REPLACE with reference 1
 *      and the colour write mask 0: writes stencil only;
 *   4. (formats with stencil) the whole image, blue, with the stencil test
 *      EQUAL 1 and the depth test off: only the top half may turn blue, so
 *      without a working stencil test the whole image would be blue.
 *
 * Expected colour image: top half blue (formats with stencil), bottom (or
 * all) rows left half red, right half green. Only the colour image is
 * read back. The edges fall on pixel boundaries (clip 0 is pixel 32); the
 * check skips a one-pixel margin on either side of them anyway and reports
 * it apart.
 *
 * The shaders are SPIR-V embedded as arrays, generated with glslangValidator
 * 16.6.0 (`glslangValidator -V --target-env vulkan1.1`) from:
 *
 *   pos3.vert:  #version 450
 *               layout(location = 0) in vec3 pos;
 *               void main() { gl_Position = vec4(pos, 1.0); }
 *
 *   red.frag, green.frag, blue.frag:
 *               #version 450
 *               layout(location = 0) out vec4 color;
 *               void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }   (and so on)
 *
 * Usage: zs_g52 [path/to/libvulkan_malisx2.so]
 *   The path defaults to $MALISX2_ICD, then to ZS_G52_DEFAULT_ICD.
 *
 * Exit status: 0 PASS (every format the device offers passed, at least
 * one was run); 1 FAIL (wrong pixels, a timeout or a Vulkan error once the
 * device exists, or no format at all could be run); 2 the program did not
 * get as far as a VkDevice.
 *
 * Driver log: MESA_LOG=file sends it to stderr, MESA_LOG_LEVEL=info adds
 * the atom layout and build lines; MALISX2_DUMP_DRAW=1 dumps every draw's
 * descriptors (see MALISX2_G52.md).
 */

#include "g52_harness.h"

#ifndef ZS_G52_DEFAULT_ICD
#define ZS_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define IMG_W 64
#define IMG_H 64
#define SENTINEL 0xa5
#define MARGIN 1

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

static const uint32_t red_frag_spv[] = {
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
   0x0000000b, 0x0000000b, 0x0000000a, 0x00050036, 0x00000002, 0x00000004,
   0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0003003e, 0x00000009,
   0x0000000c, 0x000100fd, 0x00010038,
};

static const uint32_t green_frag_spv[] = {
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
   0x0004002b, 0x00000006, 0x0000000a, 0x00000000, 0x0004002b, 0x00000006,
   0x0000000b, 0x3f800000, 0x0007002c, 0x00000007, 0x0000000c, 0x0000000a,
   0x0000000b, 0x0000000a, 0x0000000b, 0x00050036, 0x00000002, 0x00000004,
   0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0003003e, 0x00000009,
   0x0000000c, 0x000100fd, 0x00010038,
};

static const uint32_t blue_frag_spv[] = {
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
   0x0004002b, 0x00000006, 0x0000000a, 0x00000000, 0x0004002b, 0x00000006,
   0x0000000b, 0x3f800000, 0x0007002c, 0x00000007, 0x0000000c, 0x0000000a,
   0x0000000a, 0x0000000b, 0x0000000b, 0x00050036, 0x00000002, 0x00000004,
   0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0003003e, 0x00000009,
   0x0000000c, 0x000100fd, 0x00010038,
};

/* The triangles, 3 floats per vertex (x, y, z), 3 vertices each. */
#define TRI_NEAR 0   /* right half, z = 0.25 */
#define TRI_FAR 1    /* whole image, z = 0.75 */
#define TRI_TOP 2    /* top half, z = 0.5 */
#define TRI_ALL 3    /* whole image, z = 0.5 */
static const float vertices[4][9] = {
   {0.0f, -3.0f, 0.25f, 0.0f, 3.0f, 0.25f, 3.0f, 0.0f, 0.25f},
   {-1.0f, -1.0f, 0.75f, 3.0f, -1.0f, 0.75f, -1.0f, 3.0f, 0.75f},
   {-3.0f, 0.0f, 0.5f, 3.0f, 0.0f, 0.5f, 0.0f, -3.0f, 0.5f},
   {-1.0f, -1.0f, 0.5f, 3.0f, -1.0f, 0.5f, -1.0f, 3.0f, 0.5f},
};

static const uint8_t red[4] = {255, 0, 0, 255};
static const uint8_t green[4] = {0, 255, 0, 255};
static const uint8_t blue[4] = {0, 0, 255, 255};
static const uint8_t black[4] = {0, 0, 0, 255};

struct zs_format {
   VkFormat format;
   const char *name;
   bool has_stencil;
};

static const struct zs_format zs_formats[] = {
   {VK_FORMAT_D24_UNORM_S8_UINT, "D24_UNORM_S8_UINT", true},
   {VK_FORMAT_D32_SFLOAT_S8_UINT, "D32_SFLOAT_S8_UINT", true},
   {VK_FORMAT_D32_SFLOAT, "D32_SFLOAT", false},
   {VK_FORMAT_D16_UNORM, "D16_UNORM", false},
};

static bool
px_is(const uint8_t *px, const uint8_t *want)
{
   for (unsigned c = 0; c < 4; c++) {
      const int d = (int)px[c] - (int)want[c];
      if (d < -1 || d > 1)
         return false;
   }
   return true;
}

/* One format: the pass, the readback and the verdict. Returns 0 PASS,
 * 1 FAIL, -1 skipped (the format is not offered). */
static int
run_format(struct g52 *g, const struct zs_format *zf, struct g52_image *color,
           const struct g52_buffer *vbuf, VkPipelineLayout layout, VkCommandPool pool)
{
   const struct g52_fns *f = &g->f;
   VkFormatProperties fp;
   f->GetPhysicalDeviceFormatProperties(g->pdev, zf->format, &fp);
   if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
      printf("format %s: not offered as a depth/stencil attachment, skipped\n", zf->name);
      return -1;
   }
   printf("format %s:\n", zf->name);

   const VkImageAspectFlags zs_aspects =
      VK_IMAGE_ASPECT_DEPTH_BIT | (zf->has_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
   struct g52_image zs;
   if (!g52_image_create(g, &zs, zf->format, IMG_W, IMG_H, VK_IMAGE_TILING_OPTIMAL,
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, zs_aspects, false, 0))
      return 1;

   /* Pipelines: depth LESS with writes (near, far), stencil write only
    * (top), stencil EQUAL (all). */
   const VkPipelineDepthStencilStateCreateInfo ds_depth = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   const VkStencilOpState st_write = {
      .failOp = VK_STENCIL_OP_KEEP,
      .passOp = VK_STENCIL_OP_REPLACE,
      .depthFailOp = VK_STENCIL_OP_KEEP,
      .compareOp = VK_COMPARE_OP_ALWAYS,
      .compareMask = 0xff,
      .writeMask = 0xff,
      .reference = 1,
   };
   const VkPipelineDepthStencilStateCreateInfo ds_stencil_write = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .stencilTestEnable = VK_TRUE,
      .front = st_write,
      .back = st_write,
   };
   const VkStencilOpState st_test = {
      .failOp = VK_STENCIL_OP_KEEP,
      .passOp = VK_STENCIL_OP_KEEP,
      .depthFailOp = VK_STENCIL_OP_KEEP,
      .compareOp = VK_COMPARE_OP_EQUAL,
      .compareMask = 0xff,
      .writeMask = 0,
      .reference = 1,
   };
   const VkPipelineDepthStencilStateCreateInfo ds_stencil_test = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .stencilTestEnable = VK_TRUE,
      .front = st_test,
      .back = st_test,
   };
   const VkPipelineColorBlendAttachmentState write_all = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendAttachmentState write_none = {.colorWriteMask = 0};

   struct g52_pipeline_desc pd = {
      .vs_spv = pos3_vert_spv,
      .vs_size = sizeof(pos3_vert_spv),
      .attrib_format = VK_FORMAT_R32G32B32_SFLOAT,
      .vertex_stride = 3 * sizeof(float),
      .width = IMG_W,
      .height = IMG_H,
      .color_format = color->format,
      .depth_format = zf->format,
      .stencil_format = zf->has_stencil ? zf->format : VK_FORMAT_UNDEFINED,
   };
   VkPipeline pipes[4] = {VK_NULL_HANDLE};
   int status = 1;

   pd.fs_spv = green_frag_spv;
   pd.fs_size = sizeof(green_frag_spv);
   pd.ds = &ds_depth;
   pd.blend = write_all;
   pipes[TRI_NEAR] = g52_pipeline_create(g, &pd, layout, "near (green, depth LESS + write)");
   pd.fs_spv = red_frag_spv;
   pd.fs_size = sizeof(red_frag_spv);
   pipes[TRI_FAR] = g52_pipeline_create(g, &pd, layout, "far (red, depth LESS + write)");
   if (zf->has_stencil) {
      pd.fs_spv = blue_frag_spv;
      pd.fs_size = sizeof(blue_frag_spv);
      pd.ds = &ds_stencil_write;
      pd.blend = write_none;
      pipes[TRI_TOP] = g52_pipeline_create(g, &pd, layout, "top (stencil REPLACE 1, no colour)");
      pd.ds = &ds_stencil_test;
      pd.blend = write_all;
      pipes[TRI_ALL] = g52_pipeline_create(g, &pd, layout, "all (blue, stencil EQUAL 1)");
   }
   const unsigned draws = zf->has_stencil ? 4 : 2;
   for (unsigned i = 0; i < draws; i++) {
      if (!pipes[i])
         goto out;
   }

   /* Record. */
   f->ResetCommandPool(g->device, pool, 0);
   const VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd = VK_NULL_HANDLE;
   VkResult r = f->AllocateCommandBuffers(g->device, &cbai, &cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkAllocateCommandBuffers failed: %s\n", g->prog, g52_result_str(r));
      goto out;
   }
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   r = f->BeginCommandBuffer(cmd, &bi);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkBeginCommandBuffer failed: %s\n", g->prog, g52_result_str(r));
      goto out;
   }
   g52_image_refill(g, color, SENTINEL);
   g52_cmd_transition(g, cmd, color->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
   g52_cmd_transition(g, cmd, zs.image, zs_aspects, VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

   const VkRenderingAttachmentInfo color_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = color->view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .resolveMode = VK_RESOLVE_MODE_NONE,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = {0.0f, 0.0f, 0.0f, 1.0f},
   };
   const VkRenderingAttachmentInfo depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = zs.view,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .resolveMode = VK_RESOLVE_MODE_NONE,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.depthStencil = {1.0f, 0},
   };
   const VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {IMG_W, IMG_H}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_att,
      .pDepthAttachment = &depth_att,
      .pStencilAttachment = zf->has_stencil ? &depth_att : NULL,
   };
   f->CmdBeginRendering(cmd, &ri);
   for (unsigned i = 0; i < draws; i++) {
      const VkDeviceSize offset = i * sizeof(vertices[0]);
      f->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[i]);
      f->CmdBindVertexBuffers(cmd, 0, 1, &vbuf->buffer, &offset);
      f->CmdDraw(cmd, 3, 1, 0, 0);
   }
   f->CmdEndRendering(cmd);
   g52_cmd_transition(g, cmd, color->image, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                      VK_ACCESS_HOST_READ_BIT);
   r = f->EndCommandBuffer(cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkEndCommandBuffer failed: %s (recording failed: the draw's "
              "descriptors, a TODO v7 stub or command memory; see the log)\n", g->prog,
              g52_result_str(r));
      goto out;
   }
   printf("  command buffer recorded (clear, %u draws)\n", draws);

   r = g52_submit_wait(g, cmd, "(one vtc atom of vertex + tiler job pairs; one fragment atom)");
   if (r != VK_SUCCESS)
      goto out;

   /* Verdict. */
   g52_image_sync_for_read(g, color);
   uint32_t bad = 0, edge_bad = 0, n_red = 0, n_green = 0, n_blue = 0, n_black = 0, n_sent = 0;
   int first_x = -1, first_y = -1;
   uint8_t first_got[4] = {0}, first_want[4] = {0};
   for (uint32_t y = 0; y < IMG_H; y++) {
      for (uint32_t x = 0; x < IMG_W; x++) {
         const uint8_t *px = g52_image_pixel(color, x, y);
         if (px_is(px, red))
            n_red++;
         else if (px_is(px, green))
            n_green++;
         else if (px_is(px, blue))
            n_blue++;
         else if (px_is(px, black))
            n_black++;
         else if (px[0] == SENTINEL && px[1] == SENTINEL && px[2] == SENTINEL && px[3] == SENTINEL)
            n_sent++;
         const uint8_t *want;
         if (zf->has_stencil && y < IMG_H / 2)
            want = blue;
         else
            want = x < IMG_W / 2 ? red : green;
         if (px_is(px, want))
            continue;
         const bool on_edge = (x + MARGIN >= IMG_W / 2 && x < IMG_W / 2 + MARGIN) ||
                              (zf->has_stencil && y + MARGIN >= IMG_H / 2 && y < IMG_H / 2 + MARGIN);
         if (on_edge) {
            edge_bad++;
            continue;
         }
         if (!bad) {
            first_x = (int)x;
            first_y = (int)y;
            memcpy(first_got, px, 4);
            memcpy(first_want, want, 4);
         }
         bad++;
      }
   }
   printf("  pixels: %u red, %u green, %u blue, %u black (clear), %u sentinel, of %u\n", n_red,
          n_green, n_blue, n_black, n_sent, IMG_W * IMG_H);
   if (bad == 0) {
      printf("  PASS %s: %s%sleft half red, right half green (%u edge pixels not checked)\n",
             zf->name, zf->has_stencil ? "top half blue, " : "",
             zf->has_stencil ? "bottom rows " : "", edge_bad);
      status = 0;
   } else {
      printf("  FAIL %s: %u pixels wrong (plus %u on the edges); first at (%d,%d) = %u %u %u %u, "
             "expected %u %u %u %u.", zf->name, bad, edge_bad, first_x, first_y, first_got[0],
             first_got[1], first_got[2], first_got[3], first_want[0], first_want[1], first_want[2],
             first_want[3]);
      if (n_red == IMG_W * IMG_H)
         printf(" All red: the far triangle overwrote the near one, the depth test did nothing.");
      else if (zf->has_stencil && n_blue == IMG_W * IMG_H)
         printf(" All blue: the stencil test did nothing.");
      else if (zf->has_stencil && n_blue == 0 && n_red + n_green == IMG_W * IMG_H)
         printf(" No blue: the stencil write (draw 3) or the stencil test (draw 4) failed, or "
                "the stencil clear was not 0.");
      else if (n_sent)
         printf(" Sentinel pixels: the colour was not written back.");
      printf("\n");
   }

out:
   for (unsigned i = 0; i < 4; i++) {
      if (pipes[i])
         f->DestroyPipeline(g->device, pipes[i], NULL);
   }
   f->DeviceWaitIdle(g->device);
   g52_image_destroy(g, &zs);
   return status;
}

int
main(int argc, char **argv)
{
   struct g52 g;
   int rc = g52_open(&g, "zs_g52", argc, argv, ZS_G52_DEFAULT_ICD);
   if (rc)
      return rc;

   struct g52_image color;
   if (!g52_image_create(&g, &color, VK_FORMAT_R8G8B8A8_UNORM, IMG_W, IMG_H,
                         VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, true, SENTINEL))
      return 1;
   struct g52_buffer vbuf;
   if (!g52_buffer_create(&g, &vbuf, vertices, sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT))
      return 1;

   const VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout layout = VK_NULL_HANDLE;
   VkResult r = g.f.CreatePipelineLayout(g.device, &plci, NULL, &layout);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "zs_g52: vkCreatePipelineLayout failed: %s\n", g52_result_str(r));
      return 1;
   }
   const VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkCommandPool pool = VK_NULL_HANDLE;
   r = g.f.CreateCommandPool(g.device, &cpci, NULL, &pool);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "zs_g52: vkCreateCommandPool failed: %s\n", g52_result_str(r));
      return 1;
   }

   unsigned ran = 0, passed = 0;
   for (unsigned i = 0; i < sizeof(zs_formats) / sizeof(zs_formats[0]); i++) {
      int s = run_format(&g, &zs_formats[i], &color, &vbuf, layout, pool);
      if (s < 0)
         continue;
      ran++;
      passed += s == 0;
   }

   int status;
   if (ran && passed == ran) {
      printf("PASS: depth and stencil on %u format%s\n", ran, ran == 1 ? "" : "s");
      status = 0;
   } else {
      printf("FAIL: %u of %u formats passed%s\n", passed, ran,
             ran ? "" : " (no depth/stencil format could be run)");
      status = 1;
   }

   g.f.DestroyCommandPool(g.device, pool, NULL);
   g.f.DestroyPipelineLayout(g.device, layout, NULL);
   g52_buffer_destroy(&g, &vbuf);
   g52_image_destroy(&g, &color);
   g52_close(&g);
   return status;
}
