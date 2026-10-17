/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * idx_g52: indexed draws on the Bifrost (v7) path of this driver. A v7
 * indexed draw shades the vertex range [min, max] of its indices, and the
 * driver finds that range when the draw is recorded (mali_jm_cmd_draw.c
 * index_range). A wrong range drops vertices the tiler then reads from
 * unshaded slots, so every case here draws triangles whose vertices sit at
 * different ends of the vertex buffer and checks each one landed.
 *
 * The 64x64 linear RGBA8 image is a 4x4 grid of 16x16 cells. The vertex
 * buffer holds four corners per cell (64 vertices, cell c at 4c to 4c+3)
 * and two triangles cover a cell. Each round clears the image to black,
 * draws a set of cells and checks that exactly those cells hold the
 * fragment colour. The rounds run once per host-visible memory type of the
 * index buffer (uncached coherent and cached non-coherent on the RG DS, the
 * latter flushed after every CPU write):
 *
 *   1. 16-bit indices for cells 5 and 6 at offset 0;
 *   2. the same 12 indices rewritten in place with cells 0 and 15 (the
 *      range grows to the whole buffer), submitted again with the same
 *      offset, count and type: a range kept from round 1 would drop them;
 *   3. rewritten in place again with cell 7 twice (the range shrinks);
 *   4. one command buffer with three draws from the same buffer: cell 3 at
 *      firstIndex 64, cell 12 at firstIndex 128, and cell 0's indices at
 *      firstIndex 192 with vertexOffset 36 (cell 9), the index buffer
 *      bound at offset 0 and then rebound at offset 384 for a last draw of
 *      cell 13 at firstIndex 64;
 *   5. 32-bit indices for cells 1, 10 and 14 at offset 1024.
 *
 * The shaders are SPIR-V embedded as arrays, the same as blend_g52's:
 *
 *   pos2.vert:  #version 450
 *               layout(location = 0) in vec2 pos;
 *               void main() { gl_Position = vec4(pos, 0.0, 1.0); }
 *
 *   src.frag:   #version 450
 *               layout(location = 0) out vec4 color;
 *               void main() { color = vec4(0.25, 0.75, 1.0, 0.5); }
 *
 * Usage: idx_g52 [path/to/libvulkan_malisx2.so]
 *   The path defaults to $MALISX2_ICD, then to IDX_G52_DEFAULT_ICD.
 *
 * Exit status: 0 PASS; 1 FAIL (wrong pixels, a timeout or a Vulkan error
 * once the device exists); 2 the program did not get as far as a VkDevice.
 */

#include "g52_harness.h"

#ifndef IDX_G52_DEFAULT_ICD
#define IDX_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define IMG_W 64
#define IMG_H 64
#define CELL 16
#define CELLS 16
#define SENTINEL 0xa5
#define IB_SIZE 4096

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

/* The fragment colour as stored (0.75 x 255 = 191.25), and the clear. */
static const uint8_t draw_rgba[4] = {64, 191, 255, 128};
static const uint8_t clear_rgba8[4] = {0, 0, 0, 255};

static PFN_vkCmdBindIndexBuffer CmdBindIndexBuffer;
static PFN_vkCmdDrawIndexed CmdDrawIndexed;

/* Four corners per cell: (x0,y0), (x1,y0), (x0,y1), (x1,y1) in NDC. */
static void
fill_vertices(float *v)
{
   for (unsigned c = 0; c < CELLS; c++) {
      const float x0 = -1.0f + 0.5f * (float)(c % 4), y0 = -1.0f + 0.5f * (float)(c / 4);
      const float x1 = x0 + 0.5f, y1 = y0 + 0.5f;
      const float q[8] = {x0, y0, x1, y0, x0, y1, x1, y1};
      memcpy(&v[8 * c], q, sizeof(q));
   }
}

/* The six indices that cover a cell. */
static void
cell_indices(unsigned c, uint32_t out[6])
{
   const uint32_t b = 4 * c;
   const uint32_t q[6] = {b, b + 1, b + 2, b + 2, b + 1, b + 3};
   memcpy(out, q, sizeof(q));
}

static void
put16(uint8_t *map, uint32_t first, const unsigned *cells, unsigned n)
{
   uint16_t *p = (uint16_t *)map + first;
   for (unsigned i = 0; i < n; i++) {
      uint32_t q[6];
      cell_indices(cells[i], q);
      for (unsigned k = 0; k < 6; k++)
         p[6 * i + k] = (uint16_t)q[k];
   }
}

static void
put32(uint8_t *map, uint32_t first, const unsigned *cells, unsigned n)
{
   uint32_t *p = (uint32_t *)map + first;
   for (unsigned i = 0; i < n; i++)
      cell_indices(cells[i], &p[6 * i]);
}

struct ibuf {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint8_t *map;
   bool coherent;
};

static bool
ibuf_create(struct g52 *g, struct ibuf *b, uint32_t type)
{
   const struct g52_fns *f = &g->f;
   memset(b, 0, sizeof(*b));
   const VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = IB_SIZE,
      .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkResult r = f->CreateBuffer(g->device, &bci, NULL, &b->buffer);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "idx_g52: vkCreateBuffer (index) failed: %s\n", g52_result_str(r));
      return false;
   }
   VkMemoryRequirements reqs;
   f->GetBufferMemoryRequirements(g->device, b->buffer, &reqs);
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = type,
   };
   r = f->AllocateMemory(g->device, &mai, NULL, &b->memory);
   if (r == VK_SUCCESS)
      r = f->BindBufferMemory(g->device, b->buffer, b->memory, 0);
   if (r == VK_SUCCESS)
      r = f->MapMemory(g->device, b->memory, 0, VK_WHOLE_SIZE, 0, (void **)&b->map);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "idx_g52: index buffer memory (type %u): %s\n", type, g52_result_str(r));
      return false;
   }
   b->coherent = g->mem_props.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   memset(b->map, 0, IB_SIZE);
   return true;
}

static void
ibuf_flush(struct g52 *g, struct ibuf *b)
{
   if (b->coherent)
      return;
   const VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = b->memory,
      .size = VK_WHOLE_SIZE,
   };
   g->f.FlushMappedMemoryRanges(g->device, 1, &whole);
}

static void
ibuf_destroy(struct g52 *g, struct ibuf *b)
{
   if (b->map)
      g->f.UnmapMemory(g->device, b->memory);
   if (b->buffer)
      g->f.DestroyBuffer(g->device, b->buffer, NULL);
   if (b->memory)
      g->f.FreeMemory(g->device, b->memory, NULL);
   memset(b, 0, sizeof(*b));
}

/* One draw of a round. */
struct idraw {
   VkDeviceSize ib_offset;
   VkIndexType type;
   uint32_t count, first_index;
   int32_t vertex_offset;
};

struct ctx {
   struct g52 *g;
   VkCommandPool pool;
   VkCommandBuffer cmd;
   VkPipeline pipe;
   struct g52_buffer vbuf;
   struct g52_image img;
};

/* Records clear + the draws, submits, waits and checks that exactly the
 * cells in `expect` (a 16-bit mask) hold the draw colour. */
static bool
run_round(struct ctx *c, struct ibuf *ib, const char *name, const struct idraw *draws,
          unsigned ndraws, uint32_t expect)
{
   struct g52 *g = c->g;
   const struct g52_fns *f = &g->f;
   VkResult r = f->ResetCommandPool(g->device, c->pool, 0);
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   if (r == VK_SUCCESS)
      r = f->BeginCommandBuffer(c->cmd, &bi);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "idx_g52: %s: begin failed: %s\n", name, g52_result_str(r));
      return false;
   }
   g52_image_refill(g, &c->img, SENTINEL);
   g52_cmd_transition(g, c->cmd, c->img.image, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
   const VkRenderingAttachmentInfo color = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = c->img.view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .resolveMode = VK_RESOLVE_MODE_NONE,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = {0.0f, 0.0f, 0.0f, 1.0f},
   };
   const VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {IMG_W, IMG_H}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color,
   };
   f->CmdBeginRendering(c->cmd, &ri);
   f->CmdBindPipeline(c->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, c->pipe);
   const VkDeviceSize vb_offset = 0;
   f->CmdBindVertexBuffers(c->cmd, 0, 1, &c->vbuf.buffer, &vb_offset);
   VkDeviceSize bound = ~(VkDeviceSize)0;
   VkIndexType bound_type = VK_INDEX_TYPE_MAX_ENUM;
   for (unsigned i = 0; i < ndraws; i++) {
      const struct idraw *d = &draws[i];
      if (d->ib_offset != bound || d->type != bound_type) {
         CmdBindIndexBuffer(c->cmd, ib->buffer, d->ib_offset, d->type);
         bound = d->ib_offset;
         bound_type = d->type;
      }
      CmdDrawIndexed(c->cmd, d->count, 1, d->first_index, d->vertex_offset, 0);
   }
   f->CmdEndRendering(c->cmd);
   g52_cmd_transition(g, c->cmd, c->img.image, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                      VK_ACCESS_HOST_READ_BIT);
   r = f->EndCommandBuffer(c->cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "idx_g52: %s: vkEndCommandBuffer failed: %s (see the log)\n", name,
              g52_result_str(r));
      return false;
   }
   r = g52_submit_wait(g, c->cmd, name);
   if (r != VK_SUCCESS)
      return false;

   g52_image_sync_for_read(g, &c->img);
   uint32_t bad = 0, got_mask = 0, wrong_cells = 0;
   for (unsigned cell = 0; cell < CELLS; cell++) {
      const bool want = expect & (1u << cell);
      const uint8_t *ref = want ? draw_rgba : clear_rgba8;
      uint32_t cell_bad = 0, drawn = 0;
      for (uint32_t y = (cell / 4) * CELL; y < (cell / 4 + 1) * CELL; y++) {
         for (uint32_t x = (cell % 4) * CELL; x < (cell % 4 + 1) * CELL; x++) {
            const uint8_t *px = g52_image_pixel(&c->img, x, y);
            bool okpx = true, isdraw = true;
            for (unsigned k = 0; k < 4; k++) {
               const int d = (int)px[k] - (int)ref[k];
               if (d < -1 || d > 1)
                  okpx = false;
               const int e = (int)px[k] - (int)draw_rgba[k];
               if (e < -1 || e > 1)
                  isdraw = false;
            }
            cell_bad += !okpx;
            drawn += isdraw;
         }
      }
      if (drawn == CELL * CELL)
         got_mask |= 1u << cell;
      if (cell_bad)
         wrong_cells++;
      bad += cell_bad;
   }
   if (bad) {
      printf("  FAIL %s (index memory %s): %u pixels wrong in %u cells; cells drawn 0x%04x, "
             "expected 0x%04x\n", name, ib->coherent ? "coherent" : "non-coherent", bad,
             wrong_cells, got_mask, expect);
      return false;
   }
   printf("  PASS %s (cells 0x%04x)\n", name, expect);
   return true;
}

static uint32_t
mask_of(const unsigned *cells, unsigned n)
{
   uint32_t m = 0;
   for (unsigned i = 0; i < n; i++)
      m |= 1u << cells[i];
   return m;
}

/* The five rounds on an index buffer of one memory type. */
static unsigned
run_type(struct ctx *c, uint32_t type, unsigned *rounds)
{
   struct g52 *g = c->g;
   struct ibuf ib;
   if (!ibuf_create(g, &ib, type))
      return 0;
   const VkMemoryPropertyFlags fl = g->mem_props.memoryTypes[type].propertyFlags;
   printf("  index buffer in memory type %u (%s%s%s)\n", type,
          fl & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? "cached" : "uncached",
          fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? ", coherent" : ", non-coherent",
          fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? ", device local" : "");
   unsigned passed = 0;

   /* 1. cells 5 and 6 */
   static const unsigned r1[2] = {5, 6};
   put16(ib.map, 0, r1, 2);
   ibuf_flush(g, &ib);
   const struct idraw d1 = {0, VK_INDEX_TYPE_UINT16, 12, 0, 0};
   passed += run_round(c, &ib, "round 1 (16-bit, cells 5 and 6)", &d1, 1, mask_of(r1, 2));

   /* 2. rewritten in place: cells 0 and 15, the same draw */
   static const unsigned r2[2] = {0, 15};
   put16(ib.map, 0, r2, 2);
   ibuf_flush(g, &ib);
   passed += run_round(c, &ib, "round 2 (rewritten in place, cells 0 and 15)", &d1, 1,
                       mask_of(r2, 2));

   /* 3. rewritten in place again: cell 7 twice (the range shrinks) */
   static const unsigned r3[2] = {7, 7};
   put16(ib.map, 0, r3, 2);
   ibuf_flush(g, &ib);
   passed += run_round(c, &ib, "round 3 (rewritten in place, cell 7 twice)", &d1, 1,
                       mask_of(r3, 2));

   /* 4. several draws from one buffer in one command buffer */
   static const unsigned c3 = 3, c12 = 12, c0 = 0, c13 = 13;
   put16(ib.map, 64, &c3, 1);
   put16(ib.map, 128, &c12, 1);
   put16(ib.map, 192, &c0, 1);
   /* Cell 13 lives at byte offset 512 (index 256), drawn through a bind at
    * byte offset 384 with firstIndex 64. */
   put16(ib.map, 256, &c13, 1);
   ibuf_flush(g, &ib);
   const struct idraw d4[4] = {
      {0, VK_INDEX_TYPE_UINT16, 6, 64, 0},
      {0, VK_INDEX_TYPE_UINT16, 6, 128, 0},
      {0, VK_INDEX_TYPE_UINT16, 6, 192, 36},
      {384, VK_INDEX_TYPE_UINT16, 6, 64, 0},
   };
   passed += run_round(c, &ib, "round 4 (four draws, two binds, a vertex offset)", d4, 4,
                       (1u << 3) | (1u << 12) | (1u << 9) | (1u << 13));

   /* 5. 32-bit indices */
   static const unsigned r5[3] = {1, 10, 14};
   put32(ib.map, 1024 / 4, r5, 3);
   ibuf_flush(g, &ib);
   const struct idraw d5 = {1024, VK_INDEX_TYPE_UINT32, 18, 0, 0};
   passed += run_round(c, &ib, "round 5 (32-bit, cells 1, 10 and 14)", &d5, 1, mask_of(r5, 3));

   *rounds += 5;
   ibuf_destroy(g, &ib);
   return passed;
}

int
main(int argc, char **argv)
{
   struct g52 g;
   int rc = g52_open(&g, "idx_g52", argc, argv, IDX_G52_DEFAULT_ICD);
   if (rc)
      return rc;
   const struct g52_fns *f = &g.f;
   bool ok = true;
   CmdBindIndexBuffer = (PFN_vkCmdBindIndexBuffer)g52_load_device_fn(&g, "vkCmdBindIndexBuffer",
                                                                     NULL, true, &ok);
   CmdDrawIndexed = (PFN_vkCmdDrawIndexed)g52_load_device_fn(&g, "vkCmdDrawIndexed", NULL, true,
                                                            &ok);
   if (!ok)
      return 2;

   struct ctx c = {.g = &g};
   if (!g52_image_create(&g, &c.img, VK_FORMAT_R8G8B8A8_UNORM, IMG_W, IMG_H,
                         VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, true, SENTINEL))
      return 1;
   float verts[CELLS * 8];
   fill_vertices(verts);
   if (!g52_buffer_create(&g, &c.vbuf, verts, sizeof(verts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT))
      return 1;

   const VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout layout = VK_NULL_HANDLE;
   VkResult r = f->CreatePipelineLayout(g.device, &plci, NULL, &layout);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "idx_g52: vkCreatePipelineLayout failed: %s\n", g52_result_str(r));
      return 1;
   }
   struct g52_pipeline_desc pd = {
      .vs_spv = pos2_vert_spv,
      .vs_size = sizeof(pos2_vert_spv),
      .fs_spv = src_frag_spv,
      .fs_size = sizeof(src_frag_spv),
      .attrib_format = VK_FORMAT_R32G32_SFLOAT,
      .vertex_stride = 2 * sizeof(float),
      .width = IMG_W,
      .height = IMG_H,
      .color_format = VK_FORMAT_R8G8B8A8_UNORM,
      .blend = {.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT},
   };
   c.pipe = g52_pipeline_create(&g, &pd, layout, "pos2 + constant colour");
   if (!c.pipe)
      return 1;

   const VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   r = f->CreateCommandPool(g.device, &cpci, NULL, &c.pool);
   const VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = c.pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (r == VK_SUCCESS)
      r = f->AllocateCommandBuffers(g.device, &cbai, &c.cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "idx_g52: command buffer: %s\n", g52_result_str(r));
      return 1;
   }

   /* Every host-visible memory type an index buffer can live in. */
   const VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = IB_SIZE,
      .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer probe = VK_NULL_HANDLE;
   r = f->CreateBuffer(g.device, &bci, NULL, &probe);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "idx_g52: vkCreateBuffer failed: %s\n", g52_result_str(r));
      return 1;
   }
   VkMemoryRequirements reqs;
   f->GetBufferMemoryRequirements(g.device, probe, &reqs);
   f->DestroyBuffer(g.device, probe, NULL);

   unsigned passed = 0, rounds = 0, types = 0;
   for (uint32_t t = 0; t < g.mem_props.memoryTypeCount; t++) {
      if (!(reqs.memoryTypeBits & (1u << t)) ||
          !(g.mem_props.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         continue;
      types++;
      passed += run_type(&c, t, &rounds);
   }

   const int status = (rounds && passed == rounds) ? 0 : 1;
   printf("%s: %u of %u rounds over %u memory types\n", status ? "FAIL" : "PASS", passed, rounds,
          types);

   f->DeviceWaitIdle(g.device);
   f->DestroyCommandPool(g.device, c.pool, NULL);
   f->DestroyPipeline(g.device, c.pipe, NULL);
   f->DestroyPipelineLayout(g.device, layout, NULL);
   g52_buffer_destroy(&g, &c.vbuf);
   g52_image_destroy(&g, &c.img);
   g52_close(&g);
   return status;
}
