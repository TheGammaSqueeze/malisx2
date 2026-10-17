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
   printf(pass ? "PASS: dynamic stencil reference, rebound uniform offsets and draw order "
                 "over %u draws, two runs\n"
               : "FAIL: see above (%u draws)\n", CELLS + 6);

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
   g52_buffer_destroy(&g, &vbuf);
   g52_image_destroy(&g, &zs);
   g52_image_destroy(&g, &color);
   g52_close(&g);
   return pass ? 0 : 1;
}
