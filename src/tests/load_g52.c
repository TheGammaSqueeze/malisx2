/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * load_g52: loadOp LOAD and a render area that is not whole tiles on the
 * Bifrost (v7) path of this driver. Both need the frame shaders: the
 * colour and depth preloads (pre-frame shader Draw descriptors that read
 * the attachments through the texture unit into the tile buffer) and the
 * border loads of partly covered tiles, which keep the pixels outside the
 * render area as they were (the fragment job writes back whole tiles).
 *
 * A 64x64 R8G8B8A8_UNORM colour image (optimally tiled, or linear with
 * --linear) is filled with a pattern ((4x, 4y, 128, 255)) and a 64x64
 * D32_SFLOAT depth image with a gradient (0.25 + x / 128) through
 * vkCmdCopyBufferToImage, then one pass with a render area of 37x29 at
 * (5, 3) (no side on a 16-pixel tile boundary) and the draws' scissor set
 * to the same rectangle:
 *
 *   colour LOAD: a triangle covering the left half of the framebuffer
 *   (pixels x 0..31) in green. Inside the area x < 32 must be green, the
 *   rest of the area the loaded pattern; outside the area the pattern,
 *   untouched;
 *
 *   depth LOAD (colour loaded too): a full-framebuffer triangle at z 0.5,
 *   depth test LESS with writes, green. It passes only where the loaded
 *   depth is farther (x > 32), so inside the area x > 32 must be green
 *   with depth 0.5, x <= 32 the loaded colour and depth; outside the area
 *   both images untouched. A depth preload that did not run reads as all
 *   green (a cleared or farther tile buffer) or no green at all (a nearer
 *   one).
 *
 * Both images are read back through vkCmdCopyImageToBuffer. Colour allows
 * one LSB (dithering) and counts such pixels apart; depth is exact.
 *
 * The shaders are SPIR-V embedded as arrays, the same as zs_g52's
 * (glslangValidator 16.6.0, `glslangValidator -V --target-env vulkan1.1`):
 *
 *   pos3.vert:  #version 450
 *               layout(location = 0) in vec3 pos;
 *               void main() { gl_Position = vec4(pos, 1.0); }
 *
 *   green.frag: #version 450
 *               layout(location = 0) out vec4 color;
 *               void main() { color = vec4(0.0, 1.0, 0.0, 1.0); }
 *
 * Usage: load_g52 [path/to/libvulkan_malisx2.so] [--linear]
 *   The path defaults to $MALISX2_ICD, then to LOAD_G52_DEFAULT_ICD.
 *
 * Exit status: 0 PASS (both cases); 1 FAIL once the device exists; 2 the
 * program did not get as far as a VkDevice.
 *
 * Driver log: MESA_LOG=file MESA_LOG_LEVEL=info; MALISX2_DUMP_DRAW=1 dumps
 * the draw and the pass (see MALISX2_G52.md, milestone 5).
 */

#include "g52_harness.h"

#ifndef LOAD_G52_DEFAULT_ICD
#define LOAD_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

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

#define IMG_W 64
#define IMG_H 64
/* The render area: 37x29 at (5, 3), not whole 16x16 tiles on any side. */
#define AREA_X 5
#define AREA_Y 3
#define AREA_W 37
#define AREA_H 29

static struct g52 g;
static VkCommandPool pool;
static VkCommandBuffer cmd;
static VkPipelineLayout layout;
static bool linear_color;

static bool
in_area(int x, int y)
{
   return x >= AREA_X && x < AREA_X + AREA_W && y >= AREA_Y && y < AREA_Y + AREA_H;
}

/* The colour prefill and the depth prefill. */
static void
pattern(int x, int y, uint8_t out[4])
{
   out[0] = (uint8_t)(4 * x);
   out[1] = (uint8_t)(4 * y);
   out[2] = 0x80;
   out[3] = 0xff;
}

static float
depth_prefill(int x, int y)
{
   return 0.25f + (float)x / 128.0f;
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
      printf("FAIL: vkBeginCommandBuffer: %s\n", g52_result_str(r));
   return r == VK_SUCCESS;
}

static void
transition(VkImage image, VkImageAspectFlags aspects, VkImageLayout from, VkImageLayout to)
{
   g52_cmd_transition(&g, cmd, image, aspects, from, to, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}

struct counts {
   unsigned outside_bad, outside_off1, loaded_bad, loaded_off1, drawn_bad, shown;
};

static void
show(struct counts *c, const char *what, int x, int y, const uint8_t *got, const uint8_t *want)
{
   if (c->shown++ < 8)
      printf("    %s pixel (%d, %d): got %d %d %d %d, want %d %d %d %d\n", what, x, y, got[0],
             got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
}

/* Compares a pixel with what it should hold: exact, or off by one LSB in
 * some channel (counted apart: dithering), else wrong. */
static int
cmp_px(const uint8_t *got, const uint8_t *want)
{
   int worst = 0;
   for (int c = 0; c < 4; c++) {
      const int d = abs((int)got[c] - (int)want[c]);
      worst = d > worst ? d : worst;
   }
   return worst == 0 ? 0 : worst == 1 ? 1 : 2;
}

/*
 * One pass over the area with the colour attachment loaded (and the depth
 * attachment loaded when `depth`), one triangle drawn: the left half of
 * the framebuffer in green (colour case), or the whole framebuffer at
 * z = 0.5 under depth LESS with writes (depth case). Every pixel of both
 * images is read back and sorted into outside the area (must be the
 * prefill), loaded (inside, not drawn: the prefill) and drawn (green).
 */
static bool
run_case(bool depth, struct g52_image *color, struct g52_image *zimg, VkPipeline pipe,
         struct g52_buffer *vbuf)
{
   const char *name = depth ? "depth LOAD" : "colour LOAD";
   printf("case %s: render area %dx%d at (%d, %d)\n", name, AREA_W, AREA_H, AREA_X, AREA_Y);

   struct g52_hostbuf up_c = {0}, up_z = {0}, rb_c = {0}, rb_z = {0};
   bool ok = g52_hostbuf_create(&g, &up_c, IMG_W * IMG_H * 4, 0) &&
             g52_hostbuf_create(&g, &rb_c, IMG_W * IMG_H * 4, 0xa5) &&
             (!depth || (g52_hostbuf_create(&g, &up_z, IMG_W * IMG_H * 4, 0) &&
                         g52_hostbuf_create(&g, &rb_z, IMG_W * IMG_H * 4, 0xa5))) &&
             begin();
   if (!ok) {
      printf("FAIL %s: setup\n", name);
      goto out;
   }
   for (int y = 0; y < IMG_H; y++) {
      for (int x = 0; x < IMG_W; x++) {
         pattern(x, y, (uint8_t *)up_c.map + (y * IMG_W + x) * 4);
         if (depth)
            ((float *)up_z.map)[y * IMG_W + x] = depth_prefill(x, y);
      }
   }
   g52_hostbuf_flush(&g, &up_c);
   if (depth)
      g52_hostbuf_flush(&g, &up_z);

   /* Prefill through the copies, then the pass. */
   transition(color->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
   g52_cmd_copy_image(&g, cmd, color->image, VK_IMAGE_ASPECT_COLOR_BIT, IMG_W, IMG_H,
                      up_c.buffer, true);
   transition(color->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
   if (depth) {
      transition(zimg->image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      g52_cmd_copy_image(&g, cmd, zimg->image, VK_IMAGE_ASPECT_DEPTH_BIT, IMG_W, IMG_H,
                         up_z.buffer, true);
      transition(zimg->image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
   }

   const VkRenderingAttachmentInfo ca = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = color->view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingAttachmentInfo za = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = depth ? zimg->view : VK_NULL_HANDLE,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{AREA_X, AREA_Y}, {AREA_W, AREA_H}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &ca,
      .pDepthAttachment = depth ? &za : NULL,
   };
   g.f.CmdBeginRendering(cmd, &ri);
   const VkDeviceSize zero = 0;
   g.f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   g.f.CmdBindVertexBuffers(cmd, 0, 1, &vbuf->buffer, &zero);
   g.f.CmdDraw(cmd, 3, 1, depth ? 3 : 0, 0);
   g.f.CmdEndRendering(cmd);

   transition(color->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   g52_cmd_copy_image(&g, cmd, color->image, VK_IMAGE_ASPECT_COLOR_BIT, IMG_W, IMG_H,
                      rb_c.buffer, false);
   if (depth) {
      transition(zimg->image, VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      g52_cmd_copy_image(&g, cmd, zimg->image, VK_IMAGE_ASPECT_DEPTH_BIT, IMG_W, IMG_H,
                         rb_z.buffer, false);
   }
   VkResult r = g.f.EndCommandBuffer(cmd);
   if (r != VK_SUCCESS) {
      printf("FAIL %s: vkEndCommandBuffer: %s (\"TODO v7: ... frame shader\" in the log means "
             "the preload is not ported in this library)\n", name, g52_result_str(r));
      ok = false;
      goto out;
   }
   printf("  command buffer recorded (prefill copies, the pass with one draw, readback copies)\n");
   if (g52_submit_wait(&g, cmd, name) != VK_SUCCESS) {
      ok = false;
      goto out;
   }
   g52_hostbuf_invalidate(&g, &rb_c);
   if (depth)
      g52_hostbuf_invalidate(&g, &rb_z);

   /* Colour. */
   struct counts c = {0};
   unsigned n_out = 0, n_loaded = 0, n_drawn = 0, n_sentinel = 0;
   static const uint8_t green[4] = {0, 255, 0, 255};
   for (int y = 0; y < IMG_H; y++) {
      for (int x = 0; x < IMG_W; x++) {
         const uint8_t *got = (const uint8_t *)rb_c.map + (y * IMG_W + x) * 4;
         uint8_t pre[4];
         pattern(x, y, pre);
         n_sentinel += got[0] == 0xa5 && got[1] == 0xa5 && got[2] == 0xa5 && got[3] == 0xa5;
         /* Drawn: the left half (colour case) or where z 0.5 passes LESS
          * against the prefill (depth case: x > 32). */
         const bool drawn = in_area(x, y) && (depth ? x > 32 : x < 32);
         if (!in_area(x, y)) {
            n_out++;
            const int d = cmp_px(got, pre);
            if (d == 2) {
               c.outside_bad++;
               show(&c, "outside", x, y, got, pre);
            } else {
               c.outside_off1 += d;
            }
         } else if (drawn) {
            n_drawn++;
            if (cmp_px(got, green)) {
               c.drawn_bad++;
               show(&c, "drawn", x, y, got, green);
            }
         } else {
            n_loaded++;
            const int d = cmp_px(got, pre);
            if (d == 2) {
               c.loaded_bad++;
               show(&c, "loaded", x, y, got, pre);
            } else {
               c.loaded_off1 += d;
            }
         }
      }
   }
   printf("  colour: %u outside (%u wrong, %u off by one), %u loaded (%u wrong, %u off by one), "
          "%u drawn (%u wrong), %u still hold the readback sentinel\n",
          n_out, c.outside_bad, c.outside_off1, n_loaded, c.loaded_bad, c.loaded_off1, n_drawn,
          c.drawn_bad, n_sentinel);
   ok = !c.outside_bad && !c.loaded_bad && !c.drawn_bad;
   if (c.loaded_bad && c.loaded_bad == n_loaded)
      printf("  every loaded pixel is wrong: the colour preload did not run or read the wrong "
             "image (frame shader DCD, its Texture, mali_cmd_meta_gfx.c meta_fs_dcd)\n");
   if (c.outside_bad)
      printf("  pixels outside the render area changed: the border of partly covered tiles was "
             "not reloaded (fb_begin partial_tiles, the colour preload's mode)\n");

   /* Depth. */
   if (depth) {
      unsigned z_bad = 0, z_out_bad = 0, shown = 0;
      for (int y = 0; y < IMG_H; y++) {
         for (int x = 0; x < IMG_W; x++) {
            const float got = ((const float *)rb_z.map)[y * IMG_W + x];
            const float want = in_area(x, y) && x > 32 ? 0.5f : depth_prefill(x, y);
            if (got != want) {
               if (in_area(x, y))
                  z_bad++;
               else
                  z_out_bad++;
               if (shown++ < 8)
                  printf("    depth (%d, %d): got %f, want %f\n", x, y, got, want);
            }
         }
      }
      printf("  depth: %u wrong inside the area, %u wrong outside\n", z_bad, z_out_bad);
      ok &= !z_bad && !z_out_bad;
      if (c.drawn_bad == n_drawn && n_drawn)
         printf("  no green where z 0.5 should pass: the loaded depth read lower than the "
                "prefill (0 in the tile buffer: the depth preload did not run)\n");
      if (c.loaded_bad && c.loaded_bad + c.loaded_off1 == n_loaded)
         printf("  green where the prefill is nearer: the depth preload did not run (the tile "
                "buffer held something farther)\n");
   }
   printf("%s %s\n", ok ? "PASS" : "FAIL", name);

out:
   g52_hostbuf_destroy(&g, &up_c);
   g52_hostbuf_destroy(&g, &up_z);
   g52_hostbuf_destroy(&g, &rb_c);
   g52_hostbuf_destroy(&g, &rb_z);
   return ok;
}

int
main(int argc, char **argv)
{
   const char *icd = NULL;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--linear"))
         linear_color = true;
      else
         icd = argv[i];
   }
   char *args[2] = {argv[0], (char *)icd};
   int rc = g52_open(&g, "load_g52", icd ? 2 : 1, args, LOAD_G52_DEFAULT_ICD);
   if (rc)
      return rc;
   if (!g.f.CmdCopyBufferToImage || !g.f.CmdCopyImageToBuffer) {
      fprintf(stderr, "load_g52: the driver lacks the buffer/image copies\n");
      g52_close(&g);
      return 1;
   }

   bool pass = false;
   struct g52_image color = {0}, zimg = {0};
   struct g52_buffer vbuf = {0};
   VkPipeline pipe_c = VK_NULL_HANDLE, pipe_z = VK_NULL_HANDLE;

   const VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   const VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkResult r = g.f.CreatePipelineLayout(g.device, &plci, NULL, &layout);
   if (r == VK_SUCCESS)
      r = g.f.CreateCommandPool(g.device, &pci, NULL, &pool);
   if (r == VK_SUCCESS) {
      const VkCommandBufferAllocateInfo cai = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = pool,
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      };
      r = g.f.AllocateCommandBuffers(g.device, &cai, &cmd);
   }
   if (r != VK_SUCCESS) {
      printf("FAIL: pipeline layout, command pool or buffer: %s\n", g52_result_str(r));
      goto done;
   }

   VkFormatProperties zp;
   g.f.GetPhysicalDeviceFormatProperties(g.pdev, VK_FORMAT_D32_SFLOAT, &zp);
   const VkFormatFeatureFlags zwant = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   if ((zp.optimalTilingFeatures & zwant) != zwant) {
      printf("FAIL: D32_SFLOAT optimal features 0x%x lack 0x%x\n", zp.optimalTilingFeatures, zwant);
      goto done;
   }

   const VkImageUsageFlags xfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                  VK_IMAGE_USAGE_SAMPLED_BIT;
   if (!g52_image_create(&g, &color, VK_FORMAT_R8G8B8A8_UNORM, IMG_W, IMG_H,
                         linear_color ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL,
                         xfer | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                         false, 0) ||
       !g52_image_create(&g, &zimg, VK_FORMAT_D32_SFLOAT, IMG_W, IMG_H, VK_IMAGE_TILING_OPTIMAL,
                         xfer | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                         VK_IMAGE_ASPECT_DEPTH_BIT, false, 0))
      goto done;

   /* Two triangles in one buffer: the left half (x <= 0 in clip space,
    * pixels 0..31) for the colour case, the whole framebuffer for the
    * depth case, both at z 0.5. */
   static const float verts[6][3] = {
      {0.0f, -3.0f, 0.5f}, {0.0f, 5.0f, 0.5f}, {-4.0f, 1.0f, 0.5f},
      {-1.0f, -1.0f, 0.5f}, {3.0f, -1.0f, 0.5f}, {-1.0f, 3.0f, 0.5f},
   };
   if (!g52_buffer_create(&g, &vbuf, verts, sizeof(verts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT))
      goto done;

   const VkPipelineColorBlendAttachmentState write_all = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineDepthStencilStateCreateInfo ds_less = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   struct g52_pipeline_desc pd = {
      .vs_spv = pos3_vert_spv,
      .vs_size = sizeof(pos3_vert_spv),
      .fs_spv = green_frag_spv,
      .fs_size = sizeof(green_frag_spv),
      .attrib_format = VK_FORMAT_R32G32B32_SFLOAT,
      .vertex_stride = 3 * sizeof(float),
      .width = IMG_W,
      .height = IMG_H,
      /* Vulkan leaves rendering outside the render area undefined: the
       * scissor keeps the draws inside it. */
      .scissor = {{AREA_X, AREA_Y}, {AREA_W, AREA_H}},
      .color_format = VK_FORMAT_R8G8B8A8_UNORM,
      .blend = write_all,
   };
   pipe_c = g52_pipeline_create(&g, &pd, layout, "green, colour only");
   pd.depth_format = VK_FORMAT_D32_SFLOAT;
   pd.ds = &ds_less;
   pipe_z = g52_pipeline_create(&g, &pd, layout, "green, depth LESS + write");
   if (!pipe_c || !pipe_z)
      goto done;
   printf("  colour image: 64x64 R8G8B8A8_UNORM %s, depth image: 64x64 D32_SFLOAT optimal\n",
          linear_color ? "linear" : "optimal");

   pass = run_case(false, &color, &zimg, pipe_c, &vbuf);
   pass &= run_case(true, &color, &zimg, pipe_z, &vbuf);

done:
   if (pipe_c)
      g.f.DestroyPipeline(g.device, pipe_c, NULL);
   if (pipe_z)
      g.f.DestroyPipeline(g.device, pipe_z, NULL);
   g52_buffer_destroy(&g, &vbuf);
   g52_image_destroy(&g, &color);
   g52_image_destroy(&g, &zimg);
   if (pool)
      g.f.DestroyCommandPool(g.device, pool, NULL);
   if (layout)
      g.f.DestroyPipelineLayout(g.device, layout, NULL);
   g52_close(&g);
   printf("%s: load_g52\n", pass ? "PASS" : "FAIL");
   return pass ? 0 : 1;
}
