/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Draws on the job manager. Two halves: the Bifrost one (v7, first half of
 * the file: a Vertex job and a Tiler job per draw) and the Valhall one (v9,
 * the rest of this comment and the second half).
 *
 * v9: vkCmdDraw and vkCmdDrawIndexed as one
 * 384-byte Malloc Vertex job (IDVS) each, a tiler-side job in the open
 * batch's vtc chain. The binds, the dirty bits and the descriptors a draw
 * points at are shared with the v11 back half (mali_cmd_state.[ch]); this
 * file turns them into the words of the job.
 *
 * The job is built from a template (struct mali_jm_draw_tmpl): an image of
 * the next job in cached memory, in which each dirty group rewrites only
 * its own words, as the v11 draw rewrites only the staging registers whose
 * value changed. A draw then patches its counts, index pointer and header,
 * copies the 384 bytes into command memory in address order and stores
 * its address into the previous job's Next. Command memory is
 * write-combined and never read: the copy is six 64-byte stores, one per
 * line, in order, plus the 8-byte Next store behind them.
 *
 * The job's words follow Mesa's v9 gallium draw (pan_jm.c:
 * jm_emit_malloc_vertex_job, jm_emit_primitive, jm_emit_tiler_draw,
 * jm_emit_shader_env), "which source wins" for job encoding, with the
 * per-pipeline words our pipeline compile baked for v9
 * (mali_pipeline_state.c). The T820 blob writes the same job (its
 * draw_emit_malloc_vertex_job). Choices and differences:
 *
 *  - Allow rotating primitives is 0 in every draw (the baked Primitive
 *    word clears it; genxml's default is 1). Rotation would move the
 *    provoking vertex of flat-shaded triangles, and ARMSX2 shades flat a
 *    lot. The blob never sets it; Mesa sets it only without flat inputs.
 *    Whichever vertex is provoking (the Tiler Context's First provoking
 *    vertex, mali_jm_cmd_render.c) stays the one the pass's pipelines
 *    asked for, not one the hardware picked on its own;
 *  - the Draw descriptor's Vertex array is Packet = 1, pointer and
 *    strides 0 in every job: the hardware writes the vertex packet's
 *    address and strides there in Malloc Vertex mode (v9.xml), so a
 *    command buffer that runs again gets these words back with the header
 *    (mali_jm_cmd_note_reset); the blob's per-64-draw pre-pass re-arms
 *    the same words on the GPU, and we have no pre-pass;
 *  - the varying shader environment is written only when the Primitive
 *    word's Secondary Shader is set, with the position shader's resources,
 *    FAU and thread storage (Mesa; the blob writes the same), else it is
 *    zero; Allocation is the varying size + 16 and the varying size, or
 *    16 and 0 without a varying shader (Mesa's "no varyings" rule);
 *  - Instance offset is 0: zero-based instance IDs, firstInstance reaches
 *    the shader as a system value, as on v11;
 *  - the FAU block's word count goes into the Shader Environment's own
 *    byte (v9 has no count in the pointer); a zero count would drop the
 *    uniforms silently;
 *  - no occlusion query (ARMSX2 records none).
 *
 * Layered passes. Every layer has its own Tiler Context with layer offset
 * 0 (mali_jm_cmd_render.c). A primitive's layer is 0 unless the vertex
 * shader writes gl_Layer, which no v9 shader can: the device offers no
 * shader-output-layer feature and multiview is refused at compile time on
 * v9 (its tiler has no view mask). So a draw in a layered pass renders
 * into layer 0 only, as Vulkan says, and one job on layer 0's context is
 * the whole draw. The v11 back half runs one RUN_IDVS per 8-layer context
 * for the same result.
 */

#include "mali_cmd_state.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "util/log.h"
#include "vk_log.h"

#if PAN_ARCH != 7 && PAN_ARCH != 9
#error "mali_jm_cmd_draw.c is the job manager back half (PAN_ARCH 7 or 9)"
#endif

#if PAN_ARCH < 9

/*
 * Draws on the Bifrost job manager (v7): vkCmdDraw and vkCmdDrawIndexed as
 * a 192-byte Vertex job (a Compute job running the vertex shader once per
 * vertex of the range) and a 256-byte Tiler job that depends on it, in the
 * open batch's vtc chain. No Malloc Vertex / IDVS job: the vertex shader
 * writes its outputs into varying buffers the draw allocates (position,
 * point size and the generic varyings, one Attribute Buffer each), and
 * the tiler reads them back.
 *
 * Ported from panvk's v7 path (Mesa src/panfrost/vulkan/jm/panvk_vX_cmd_draw.c
 * at PAN_ARCH < 9: panvk_cmd_draw, prepare_draw, prepare_draw_layer,
 * panvk_draw_prepare_vertex_job / tiler_job, panvk_emit_vertex_dcd /
 * tiler_dcd / tiler_primitive / tiler_primitive_size, panvk_draw_prepare_
 * varyings / vs_attribs, panvk_draw_emit_attrib / attrib_buf,
 * panvk_draw_prepare_fs_rsd and panvk_emit_viewport; the varying link from
 * panvk_vX_shader.c link_shaders / emit_varying_attrs) onto this driver's
 * command buffer, dirty groups and pipeline bake. Where panvk computes
 * something at draw time that depends only on the pipeline, the pipeline
 * bakes it (mali_pipeline_state.c: the fragment RSD's state words, the
 * Draw's culling bits, the Primitive flags). Differences and choices:
 *
 *  - the vertex shader's RSD is static per shader (mali_shader.c); the
 *    fragment RSD (shader half from the shader info, state half from the
 *    bake) is rebuilt with its Blend descriptors behind it whenever the
 *    pipeline, blend or depth/stencil group is dirty;
 *  - attribute and varying descriptors and buffers are rebuilt every draw,
 *    as panvk does: they depend on the vertex and instance counts;
 *  - indexed draws: Bifrost shades the vertex range [min, max] of the
 *    indices, and the range must be known when the jobs are written.
 *    panvk computes it with a compute helper (and patches the jobs on the
 *    GPU); compute is not ported, so the indices are read on the CPU from
 *    the index buffer's mapping (mali_cmd_state.c records it). A draw whose
 *    index buffer has no CPU view fails the command buffer;
 *  - no occlusion queries, no multiview, layered passes draw into layer 0
 *    (file comment of the v9 half, which applies);
 *  - descriptor sets: each stage's uniform buffer, image, texture and
 *    sampler tables come from the builder the dispatches use too
 *    (mali_jm_desc_v7.c), rebuilt when the stage's sets or shader change,
 *    and the Draw sections point at them (the vertex shader's images
 *    behind its vertex attributes, the fragment shader's in the Tiler
 *    job's Attribute pointers); the set addresses and the dynamic storage
 *    buffer table go through the stage's FAU block (sysvals).
 *
 * Job dependencies (Mesa pan_jc.h rules): a stage whose tables were
 * rebuilt gets a descriptor copy job (a Compute job) first, the vertex
 * shader's before the fragment shader's; the Vertex job's Dependency 1 is
 * the vertex copy job, the Tiler job's Dependency 1 the Vertex job and its
 * Dependency 2 the previous tiler-side job (mali_jm_cmd_add_job). A
 * pending barrier goes to the draw's first job, and to the Vertex job as
 * well when that first job is the fragment copy.
 */

#include "mali_rsd_v7.h"

#include <inttypes.h>
#include <stdlib.h>

/* ---------------------------------------------------------------------- */
/* Mesa helpers (pan_encoder.h, MIT, Collabora), not includable here       */
/* without the kmod layer                                                  */

/* pan_padded_vertex_count: the instance stride the attribute divisor and
 * varying buffers use, a number of the form (1, 3, 5, 7 or 9) << n. */
static unsigned
padded_vertex_count(unsigned vertex_count)
{
   if (vertex_count < 10)
      return vertex_count;
   if (vertex_count < 20)
      return (vertex_count + 1) & ~1;

   const unsigned highest = 32 - __builtin_clz(vertex_count);
   const unsigned n = highest - 4;
   const unsigned nibble = (vertex_count >> n) & 0xf;
   const unsigned middle_two = (nibble >> 1) & 0x3;
   switch (middle_two) {
   case 0:
      return (nibble & 1) ? (1u << (n + 1)) * 5 : (1u << n) * 9;
   case 1:
      return (1u << (n + 2)) * 3;
   case 2:
      return (1u << (n + 1)) * 7;
   default:
      return 1u << (n + 4);
   }
}

/* pan_compute_npot_divisor: the fixed-point reciprocal of a divisor that
 * is not a power of two. */
static unsigned
compute_npot_divisor(unsigned hw_divisor, unsigned *divisor_r, unsigned *divisor_e)
{
   const unsigned r = util_logbase2(hw_divisor);
   const uint64_t shift_hi = 32 + r;
   const uint64_t t = (uint64_t)1 << shift_hi;
   const uint64_t f0 = t + hw_divisor / 2;
   const uint64_t fi = f0 / hw_divisor;
   const uint64_t ff = f0 - fi * hw_divisor;
   const uint64_t d = fi - (1ull << 31);
   *divisor_r = r;
   *divisor_e = ff > hw_divisor / 2 ? 1 : 0;
   return d;
}

/* pan_pack_work_groups_compute for graphics: the job's Invocation section
 * for (1, vertex_range, instance_count) invocations of workgroup size 1,
 * with the blob's quirks (the z shift 32 when not instanced, the minimum
 * efficient split). */
static void
pack_invocation(struct mali_invocation_packed *out, unsigned vertex_range,
                unsigned instance_count)
{
   const unsigned values[6] = {1, 1, 1, 1, vertex_range, instance_count};
   unsigned shifts[7] = {0};
   uint32_t packed = 0;

   for (unsigned i = 0; i < 6; i++) {
      assert(values[i] >= 1);
      packed |= (values[i] - 1) << shifts[i];
      shifts[i + 1] = shifts[i] + util_logbase2_ceil(values[i]);
   }
   pan_pack(out, INVOCATION, cfg) {
      cfg.invocations = packed;
      cfg.size_y_shift = shifts[1];
      cfg.size_z_shift = shifts[2];
      cfg.workgroups_x_shift = shifts[3];
      cfg.workgroups_y_shift = shifts[4];
      cfg.workgroups_z_shift = instance_count <= 1 ? 32 : shifts[5];
      cfg.thread_group_split = MALI_SPLIT_MIN_EFFICIENT;
   }
}

void
mali_jm_draw_tmpl_init(struct mali_jm_draw_tmpl *t)
{
   memset(t, 0, sizeof(*t));
}

/* The Primitive word's First provoking vertex bit (word 0 only; the count
 * is packed as 1 because the v7 packer encodes it minus one and asserts
 * it is at least 1). */
static inline uint32_t
prim_provoking_bit(void)
{
   struct mali_primitive_packed p;
   pan_pack_nodefaults(&p, PRIMITIVE, cfg) {
      cfg.first_provoking_vertex = true;
      cfg.index_count = 1;
   }
   return p.opaque[0];
}

/* ---------------------------------------------------------------------- */
/* Varyings                                                                */

/* panvk varying_buf_id. */
static inline unsigned
varying_buf(gl_varying_slot loc)
{
   switch (loc) {
   case VARYING_SLOT_POS: return MALI_V7_VARY_BUF_POSITION;
   case VARYING_SLOT_PSIZ: return MALI_V7_VARY_BUF_PSIZ;
   default: return MALI_V7_VARY_BUF_GENERAL;
   }
}

/* panvk varying_format at PAN_ARCH 7. */
static inline uint32_t
varying_format(gl_varying_slot loc, enum pipe_format pfmt)
{
   switch (loc) {
   case VARYING_SLOT_PNTC:
   case VARYING_SLOT_PSIZ:
      return (MALI_R16F << 12) | MALI_RGB_COMPONENT_ORDER_R000;
   case VARYING_SLOT_POS:
      return (MALI_SNAP_4 << 12) | MALI_RGB_COMPONENT_ORDER_RGBA;
   default:
      return GENX(pan_format_from_pipe_format)(pfmt)->hw;
   }
}

/*
 * panvk emit_varying_attrs: one Attribute descriptor per slot of a stage's
 * varying layout (descriptor i is the LEA_ATTR / LD_VAR index i the
 * compiler used). A generic slot the other stage does not have reads or
 * writes a constant zero. bit_sizes and buf_offsets are indexed by
 * location. *out = the descriptors' address, 0 for an empty layout.
 */
static bool
emit_varying_attrs(struct mali_cmd_buffer *cmd, const struct pan_varying_layout *layout,
                   const unsigned *bit_sizes, const unsigned *buf_offsets, uint64_t *out)
{
   *out = 0;
   if (!layout->count)
      return true;

   struct mali_ptr p = mali_cmd_alloc(cmd, layout->count * pan_size(ATTRIBUTE),
                                      pan_alignment(ATTRIBUTE));
   if (!p.cpu)
      return false;

   struct mali_attribute_packed *attrs = p.cpu;
   for (unsigned i = 0; i < layout->count; i++) {
      pan_pack(&attrs[i], ATTRIBUTE, cfg) {
         const struct pan_varying_slot *slot = pan_varying_layout_slot_at(layout, i);
         if (!slot)
            continue;
         const gl_varying_slot loc = slot->location;
         const bool is_hw = loc < VARYING_SLOT_VAR0;
         const bool present = is_hw || bit_sizes[loc] != 0;
         enum pipe_format pfmt = PIPE_FORMAT_NONE;
         if (present && !is_hw) {
            const nir_alu_type base = nir_alu_type_get_base_type(slot->alu_type);
            pfmt = pan_varying_format(base | bit_sizes[loc], slot->ncomps);
         }
         if (!present) {
            cfg.format = (MALI_CONSTANT << 12) | MALI_RGB_COMPONENT_ORDER_0000;
         } else {
            cfg.buffer_index = varying_buf(loc);
            cfg.offset = buf_offsets[loc];
            cfg.format = varying_format(loc, pfmt);
         }
         cfg.offset_enable = false;
      }
   }
   *out = p.gpu;
   return true;
}

/*
 * panvk link_shaders: the varying buffer strides and the two stages'
 * varying descriptors for the bound pipeline. The position and point size
 * have buffers of their own (the tiler reads them through the Draw's
 * position pointer and the Primitive Size section); the generic varyings
 * share the general buffer at the offsets of the vertex shader's layout,
 * which the fragment shader was compiled against. A generic varying is
 * carried only when the fragment shader reads it and the vertex shader
 * writes it.
 */
static bool
link_varyings(struct mali_cmd_buffer *cmd, const struct mali_graphics_pipeline *p,
              struct mali_jm_draw_tmpl *t)
{
   const struct pan_varying_layout *vl = &p->vs->info.varyings.formats;
   const struct pan_varying_layout *fl = p->fs ? &p->fs->info.varyings.formats : NULL;
   unsigned buf_offsets[VARYING_SLOT_MAX] = {0};
   unsigned bit_sizes[VARYING_SLOT_MAX] = {0};

   pan_varying_layout_require_layout(vl);
   memset(t->vary_stride, 0, sizeof(t->vary_stride));

   const struct pan_varying_slot *pos = pan_varying_layout_find_slot(vl, VARYING_SLOT_POS);
   const struct pan_varying_slot *psiz = pan_varying_layout_find_slot(vl, VARYING_SLOT_PSIZ);
   if (pos) {
      t->vary_stride[MALI_V7_VARY_BUF_POSITION] = 4 * sizeof(uint32_t);
      bit_sizes[VARYING_SLOT_POS] = nir_alu_type_get_type_size(pos->alu_type);
   }
   if (psiz) {
      t->vary_stride[MALI_V7_VARY_BUF_PSIZ] = sizeof(uint16_t);
      bit_sizes[VARYING_SLOT_PSIZ] = nir_alu_type_get_type_size(psiz->alu_type);
   }
   t->vary_stride[MALI_V7_VARY_BUF_GENERAL] = vl->generic_size_B;

   const uint32_t fs_count = fl ? fl->count : 0;
   for (uint32_t i = 0; i < fs_count; i++) {
      const struct pan_varying_slot *fslot = pan_varying_layout_slot_at(fl, i);
      if (!fslot || fslot->location < VARYING_SLOT_VAR0)
         continue;
      assert(varying_buf(fslot->location) == MALI_V7_VARY_BUF_GENERAL);
      const struct pan_varying_slot *vslot = pan_varying_layout_find_slot(vl, fslot->location);
      if (vslot) {
         buf_offsets[fslot->location] = vslot->offset;
         bit_sizes[fslot->location] = nir_alu_type_get_type_size(vslot->alu_type);
      }
   }

   if (!emit_varying_attrs(cmd, vl, bit_sizes, buf_offsets, &t->vs_vary))
      return false;
   if (fl) {
      pan_varying_layout_require_format(fl);
      if (!emit_varying_attrs(cmd, fl, bit_sizes, buf_offsets, &t->fs_vary))
         return false;
   } else {
      t->fs_vary = 0;
   }
   return true;
}

/* ---------------------------------------------------------------------- */
/* Per-draw descriptors                                                    */

struct draw_data {
   uint32_t vertex_range;       /* vertices the vertex job shades */
   uint32_t padded_count;       /* padded_vertex_count(vertex_range), or the range when not instanced */
   int32_t raw_vertex_offset;   /* Draw offset_start: firstVertex, or min index + vertexOffset */
   uint64_t indices;            /* GPU address of the first index, 0 for a plain draw */
   /* Outputs of the builders. */
   uint64_t attribs, attrib_bufs;
   uint64_t vary_bufs, position, psiz;
   /* The stages whose descriptor tables were rebuilt for this draw: their
    * copied entries are still to be filled (mali_v7_desc_copy), and the
    * buffer index the vertex shader's image Attributes start at. */
   bool vs_copy, fs_copy;
   uint32_t vs_img_buf_offset;
};

/*
 * panvk_draw_prepare_varyings: the varying buffers of this draw, sized for
 * its vertices and instances (padded count x instances x stride), one
 * Attribute Buffer each plus the empty entry that stops the Bifrost
 * prefetcher, from command memory (GPU read/write).
 */
static bool
emit_varying_bufs(struct mali_cmd_buffer *cmd, const struct mali_jm_draw_tmpl *t,
                  const struct mali_draw_info *di, struct draw_data *dd)
{
   struct mali_ptr bufs = mali_cmd_alloc(cmd, (MALI_V7_VARY_BUF_COUNT + 1) * pan_size(ATTRIBUTE_BUFFER),
                                         pan_alignment(ATTRIBUTE_BUFFER));
   if (!bufs.cpu)
      return false;

   struct mali_attribute_buffer_packed descs[MALI_V7_VARY_BUF_COUNT + 1];
   memset(descs, 0, sizeof(descs));
   uint64_t addrs[MALI_V7_VARY_BUF_COUNT] = {0};
   for (unsigned i = 0; i < MALI_V7_VARY_BUF_COUNT; i++) {
      const uint64_t size = (uint64_t)dd->padded_count * di->instance_count * t->vary_stride[i];
      if (size) {
         if (size > UINT32_MAX)
            return false;
         struct mali_ptr b = mali_cmd_alloc(cmd, size, 64);
         if (!b.cpu)
            return false;
         addrs[i] = b.gpu;
      }
      pan_pack(&descs[i], ATTRIBUTE_BUFFER, cfg) {
         cfg.stride = t->vary_stride[i];
         cfg.size = (uint32_t)size;
         cfg.pointer = addrs[i];
      }
   }
   memcpy(bufs.cpu, descs, sizeof(descs));
   dd->vary_bufs = bufs.gpu;
   dd->position = addrs[MALI_V7_VARY_BUF_POSITION];
   dd->psiz = addrs[MALI_V7_VARY_BUF_PSIZ];
   return true;
}

/* panvk_draw_emit_attrib_buf: a vertex binding as an Attribute Buffer pair
 * (the second entry is the NPOT divisor continuation, else zero). The
 * buffer address is 64-byte aligned down and the remainder moved into the
 * attributes' offsets. */
static void
emit_attrib_buf(const struct vk_vertex_binding_state *b, uint32_t stride, uint64_t addr,
                uint64_t size, uint32_t instance_count, uint32_t padded_count,
                struct mali_attribute_buffer_packed *desc)
{
   const uint64_t base = addr & ~63ull;
   const uint32_t bsize = (uint32_t)MIN2(size + (addr & 63), UINT32_MAX);
   const bool per_instance = b->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   const unsigned divisor = padded_count * b->divisor;
   bool continuation = false;

   if (instance_count <= 1) {
      pan_pack(&desc[0], ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = per_instance ? 0 : stride;
         cfg.pointer = base;
         cfg.size = bsize;
      }
   } else if (!per_instance) {
      pan_pack(&desc[0], ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_MODULUS;
         cfg.divisor = padded_count;
         cfg.stride = stride;
         cfg.pointer = base;
         cfg.size = bsize;
      }
   } else if (!divisor) {
      /* Divisor 0: every instance reads the first element. */
      pan_pack(&desc[0], ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = 0;
         cfg.pointer = base;
         cfg.size = bsize;
      }
   } else if (util_is_power_of_two_or_zero(divisor)) {
      pan_pack(&desc[0], ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.stride = stride;
         cfg.pointer = base;
         cfg.size = bsize;
         cfg.divisor_r = __builtin_ctz(divisor);
      }
   } else {
      unsigned divisor_r = 0, divisor_e = 0;
      const unsigned divisor_d = compute_npot_divisor(divisor, &divisor_r, &divisor_e);
      pan_pack(&desc[0], ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_NPOT_DIVISOR;
         cfg.stride = stride;
         cfg.pointer = base;
         cfg.size = bsize;
         cfg.divisor_r = divisor_r;
         cfg.divisor_e = divisor_e;
      }
      pan_cast_and_pack(&desc[1], ATTRIBUTE_BUFFER_CONTINUATION_NPOT, cfg) {
         cfg.divisor_numerator = divisor_d;
         cfg.divisor = b->divisor;
      }
      continuation = true;
   }
   if (!continuation)
      memset(&desc[1], 0, sizeof(desc[1]));
}

/*
 * panvk_draw_prepare_vs_attribs: the vertex attributes of this draw, one
 * Attribute descriptor per location (unbound locations read a constant
 * zero) and two Attribute Buffer entries per binding plus the empty
 * terminator. Rebuilt every draw: the instance count and base instance
 * are in the descriptors.
 *
 * A vertex shader's storage images and texel buffers are attributes too:
 * their Attribute descriptors follow the 16 vertex attribute locations
 * and their Attribute Buffer pairs follow the bindings', before the
 * terminator. Those entries are the vertex shader's image table
 * (MALI_V7_DESC_TABLE_IMG of its struct mali_v7_desc_tables, pointed here)
 * and are written by its descriptor copy (mali_v7_desc_copy), which runs
 * in front of the Vertex job; the image Attributes name buffer 2 x
 * (binding count + image index), which the image table's size limit keeps
 * within the 9-bit Attribute buffer index (mali_shader.h).
 */
static_assert(2 * (MALI_MAX_VBS + MALI_V7_MAX_STAGE_IMG_ENTRIES) <= MALI_V7_ATTRIB_BUF_RECORDS,
              "a vertex shader's image pairs fit behind the vertex buffers' within the "
              "Attribute's buffer index");

static bool
emit_attributes(struct mali_cmd_buffer *cmd, const struct mali_graphics_pipeline *p,
                const struct mali_draw_info *di, struct draw_data *dd)
{
   const struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct vk_vertex_input_state *vi = d->vi;
   const uint16_t *strides = d->vi_strides;
   struct mali_v7_desc_tables *vt = &cmd->jm.draw.vs_tab;
   const unsigned num_vbs = util_last_bit(vi->bindings_valid);
   const unsigned num_imgs = p->vs->v7.count[MALI_V7_DESC_TABLE_IMG];
   const unsigned num_attribs =
      num_imgs ? MALI_MAX_VS_ATTRIBS
               : MAX2(util_last_bit(vi->attributes_valid), p->vs->info.attribute_count);
   const unsigned buf_count = (num_vbs + num_imgs) * 2;
   assert(buf_count <= MALI_V7_ATTRIB_BUF_RECORDS);

   struct mali_ptr bufs = mali_cmd_alloc(cmd, (buf_count + 1) * pan_size(ATTRIBUTE_BUFFER),
                                         pan_alignment(ATTRIBUTE_BUFFER));
   if (!bufs.cpu)
      return false;
   struct mali_ptr attrs = {0};
   if (num_attribs + num_imgs) {
      attrs = mali_cmd_alloc(cmd, (num_attribs + num_imgs) * pan_size(ATTRIBUTE),
                             pan_alignment(ATTRIBUTE));
      if (!attrs.cpu)
         return false;
   }

   /* Packed in cached memory, copied out once: command memory is uncached. */
   struct mali_attribute_buffer_packed bdescs[2 * MALI_MAX_VBS + 1];
   struct mali_attribute_packed adescs[MALI_MAX_VS_ATTRIBS];
   assert(num_vbs <= MALI_MAX_VBS && num_attribs <= MALI_MAX_VS_ATTRIBS);

   for (unsigned b = 0; b < num_vbs; b++) {
      if ((vi->bindings_valid & BITFIELD_BIT(b)) && d->vb[b].addr) {
         emit_attrib_buf(&vi->bindings[b], strides[b], d->vb[b].addr, d->vb[b].size,
                         di->instance_count, dd->padded_count, &bdescs[2 * b]);
      } else {
         memset(&bdescs[2 * b], 0, 2 * sizeof(bdescs[0]));
      }
   }
   /* The terminator (behind the images when the shader has any). */
   memset(&bdescs[2 * num_vbs], 0, sizeof(bdescs[0]));

   for (unsigned i = 0; i < num_attribs; i++) {
      if (!(vi->attributes_valid & BITFIELD_BIT(i))) {
         pan_pack(&adescs[i], ATTRIBUTE, cfg) {
            cfg.format = (MALI_CONSTANT << 12) | MALI_RGB_COMPONENT_ORDER_0000;
         }
         continue;
      }
      const struct vk_vertex_attribute_state *a = &vi->attributes[i];
      const struct vk_vertex_binding_state *b = &vi->bindings[a->binding];
      const bool per_instance = b->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
      const uint64_t addr = d->vb[a->binding].addr;
      pan_pack(&adescs[i], ATTRIBUTE, cfg) {
         cfg.buffer_index = a->binding * 2;
         cfg.offset_enable = true;
         cfg.format = GENX(pan_format_from_pipe_format)(vk_format_to_pipe_format(a->format))->hw;
         cfg.offset = a->offset + (addr & 63) +
                      (per_instance ? (uint64_t)di->first_instance * strides[a->binding] : 0);
      }
   }

   if (!num_imgs) {
      memcpy(bufs.cpu, bdescs, (buf_count + 1) * sizeof(bdescs[0]));
   } else {
      memcpy(bufs.cpu, bdescs, 2 * num_vbs * sizeof(bdescs[0]));
      memcpy((uint8_t *)bufs.cpu + buf_count * pan_size(ATTRIBUTE_BUFFER), &bdescs[2 * num_vbs],
             sizeof(bdescs[0]));
      /* The vertex shader's image table: the copy fills it. */
      const uint64_t img_bufs = 2ull * num_vbs * pan_size(ATTRIBUTE_BUFFER);
      const uint64_t img_attrs = (uint64_t)MALI_MAX_VS_ATTRIBS * pan_size(ATTRIBUTE);
      vt->tables[MALI_V7_DESC_TABLE_IMG] = bufs.gpu + img_bufs;
      vt->tables_cpu[MALI_V7_DESC_TABLE_IMG] = (uint8_t *)bufs.cpu + img_bufs;
      vt->img_attribs = attrs.gpu + img_attrs;
      vt->img_attribs_cpu = (uint8_t *)attrs.cpu + img_attrs;
      dd->vs_img_buf_offset = 2 * num_vbs;
   }
   if (num_attribs)
      memcpy(attrs.cpu, adescs, num_attribs * sizeof(adescs[0]));
   dd->attrib_bufs = bufs.gpu;
   dd->attribs = attrs.gpu;
   return true;
}

/*
 * The fragment Renderer State Descriptor with the Blend descriptors behind
 * it (panvk_draw_prepare_fs_rsd): the shader half from the fragment
 * shader's info, the state half (words 4 to 11) baked by the pipeline
 * (mali_pipeline_state.c pack_rsd_state, or the command buffer's repack
 * for a pipeline with dynamic state), then one Blend descriptor per
 * render target of the pass. The Blend descriptors come first: whether a
 * target ended up with a blend shader (the dynamic blend constants decide
 * some of them) feeds the RSD under multisampling.
 */
static bool
build_fs_rsd(struct mali_cmd_buffer *cmd, const struct mali_graphics_pipeline *p,
             const struct mali_gfx_baked *bk, struct mali_jm_draw_tmpl *t)
{
   const uint32_t rt_count = cmd->gfx.render.rt_count;
   struct mali_ptr m = mali_cmd_alloc(cmd, pan_size(RENDERER_STATE) + rt_count * pan_size(BLEND),
                                      pan_alignment(RENDERER_STATE));
   if (!m.cpu)
      return false;

   if (!mali_gfx_blend_v7(cmd, (uint8_t *)m.cpu + pan_size(RENDERER_STATE), rt_count, t->fs_code))
      return false;

   /* A blend shader under multisampling runs once per sample and reads the
    * sample ID and coverage (r61): on Bifrost the driver sets the RSD up
    * for it (the compiler preloads those registers on its own on v9+ only;
    * bifrost_compile.c). The bake knows the pipeline's own blend shaders,
    * the dynamic blend constants can add some, so decide here. */
   const struct vk_dynamic_graphics_state *st = p->bind.dynamic ? &cmd->vk.dynamic_graphics_state
                                                                : &p->state;
   const bool per_sample_blend = cmd->gfx.draw.blend_shader_mask &&
                                 st->ms.rasterization_samples > 1;

   struct mali_renderer_state_packed rsd;
   pan_pack(&rsd, RENDERER_STATE, cfg) {
      if (p->fs) {
         mali_v7_rsd_shader(&p->fs->info, t->fs_code, &cfg);
         if (per_sample_blend)
            cfg.preload.fragment.sample_mask_id = true;
      }
   }
   STATIC_ASSERT(sizeof(bk->zsd) == MALI_V7_RSD_STATE_WORDS * sizeof(uint32_t));
   memcpy(&rsd.opaque[MALI_V7_RSD_STATE_WORD_FIRST], bk->zsd, sizeof(bk->zsd));
   if (per_sample_blend) {
      struct mali_multisample_misc_packed ms;
      pan_pack_nodefaults(&ms, MULTISAMPLE_MISC, cfg) {
         cfg.evaluate_per_sample = true;
      }
      rsd.opaque[MALI_V7_RSD_MULTISAMPLE_MISC_WORD] |= ms.opaque[0];
   }
   memcpy(m.cpu, &rsd, sizeof(rsd));
   t->fs_rsd = m.gpu;
   t->fs_rsd_cpu = m.cpu;
   return true;
}

/* ---------------------------------------------------------------------- */
/* Draw state                                                              */

/*
 * The draw's state (mali_cmd_state.h, "A draw's state"), v7 shape: the
 * shared step 1, then each dirty group's descriptors into the template.
 */
static bool
prepare_draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di, struct draw_data *dd)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_jm_draw_tmpl *t = &cmd->jm.draw;
   const struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_gfx_bind *bd = &p->bind;
   const struct mali_gfx_baked *dbk;
   uint64_t v;

   if (!mali_gfx_update_dirty(cmd, di, &dbk))
      return false;
   const struct mali_gfx_baked *bk = dbk ? dbk : &p->baked;

   /* An indexed draw's vertex offset changes the VS FAU block when the
    * shader reads gl_VertexIndex (the raw offset shares an FAU word with
    * the noperspective mask). */
   if (t->raw_vertex_offset != dd->raw_vertex_offset &&
       BITSET_TEST(p->vs->fau.used_sysvals,
                   offsetof(struct mali_graphics_sysvals, vs.raw_vertex_offset) / MALI_FAU_WORD_SIZE))
      d->dirty |= MALI_GFX_DIRTY_VS_FAU;

   if (d->dirty & MALI_GFX_DIRTY_PIPELINE) {
      if (!link_varyings(cmd, p, t))
         return false;
      t->prim0 = bk->tiler_flags;
      t->dcd0 = bk->dcd0[0];
      t->prim = bk->prim;
      t->lines = u_reduced_prim(bk->prim) == MESA_PRIM_LINES;
      t->writes_psiz = bk->prim == MESA_PRIM_POINTS && p->vs->info.vs.writes_point_size;
      /* The bind block's three words are the baked position, varying and
       * fragment entries (mali_pipeline.c fill_bind): on v7 the first is
       * the vertex RSD (one for points and triangles alike), the second is
       * 0 (no varying shader), the third the fragment code address. */
      t->vs_rsd = bd->spd[0];
      t->fs_code = bd->spd[2];
      assert(t->vs_rsd);
   }
   if (d->dirty & (MALI_GFX_DIRTY_PIPELINE | MALI_GFX_DIRTY_BLEND | MALI_GFX_DIRTY_ZSD)) {
      if (!build_fs_rsd(cmd, p, bk, t))
         return false;
   }

   /*
    * The descriptor tables of each stage (mali_jm_desc_v7.c, the builder
    * the dispatches use too; mali_gfx_build_vs_srt's v7 shape). Rebuilt
    * when a set the stage reads was bound (VS_SRT / FS_SRT), when the
    * stage's shader changed (the tables' layout is the shader's own, which
    * the pipeline's table keys do not cover: two pipelines reading the
    * same sets through different bindings share a key), and on every draw
    * for a vertex shader with images, whose entries live in the draw's own
    * attribute tables (emit_attributes). Their copy jobs go into the chain
    * in front of the draw's jobs (draw()). A stage that reads the set
    * addresses gets them through its FAU block, so its FAU group follows.
    */
   if (t->vs_tab.shader != p->vs || p->vs->v7.count[MALI_V7_DESC_TABLE_IMG])
      d->dirty |= MALI_GFX_DIRTY_VS_SRT;
   if (t->fs_tab.shader != p->fs)
      d->dirty |= MALI_GFX_DIRTY_FS_SRT;
   if (d->dirty & MALI_GFX_DIRTY_VS_SRT) {
      if (!mali_v7_desc_prepare(cmd, p->vs, &cmd->gfx.desc, false, &t->vs_tab))
         return false;
      dd->vs_copy = true;
      d->vs_srt_key = bd->vs_srt_key;
      d->vs_sets_dirty = 0;
      if (p->vs->v7.needs_set_addrs)
         d->dirty |= MALI_GFX_DIRTY_VS_FAU;
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_SRT) {
      if (p->fs) {
         if (!mali_v7_desc_prepare(cmd, p->fs, &cmd->gfx.desc, true, &t->fs_tab))
            return false;
         dd->fs_copy = true;
         if (p->fs->v7.needs_set_addrs)
            d->dirty |= MALI_GFX_DIRTY_FS_FAU;
      } else {
         memset(&t->fs_tab, 0, sizeof(t->fs_tab));
      }
      d->fs_srt_key = bd->fs_srt_key;
      d->fs_sets_dirty = 0;
   }

   struct mali_graphics_sysvals sv;
   if (d->dirty & (MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_FS_FAU)) {
      mali_gfx_fill_sysvals(cmd, &sv);
      sv.vs.raw_vertex_offset = dd->raw_vertex_offset;
   }
   if (d->dirty & MALI_GFX_DIRTY_VS_FAU) {
      /* The stage's set addresses and constant data (the v7_desc words
       * take the place of fs.blend_descs, which v7 shaders never read). */
      mali_v7_desc_sysvals(&t->vs_tab, p->vs, sv.v7_desc.sets);
      sv.common.constant_data = p->vs->v7.data_mem.gpu_va;
      if (!mali_gfx_build_vs_fau(cmd, &sv, &v))
         return false;
      /* The RSD's preload.uniform_count is the shader's FAU word count;
       * the block built here has as many words. */
      assert(!v || (v >> 56) == DIV_ROUND_UP(p->vs->info.fau.count, 2));
      t->vs_fau = v & BITFIELD64_MASK(56);
      t->raw_vertex_offset = dd->raw_vertex_offset;
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_FAU) {
      if (p->fs) {
         mali_v7_desc_sysvals(&t->fs_tab, p->fs, sv.v7_desc.sets);
         sv.common.constant_data = p->fs->v7.data_mem.gpu_va;
      }
      if (!mali_gfx_build_fs_fau(cmd, &sv, &v))
         return false;
      assert(!v || !p->fs || (v >> 56) == DIV_ROUND_UP(p->fs->info.fau.count, 2));
      t->fs_fau = v & BITFIELD64_MASK(56);
   }
   if (d->dirty & MALI_GFX_DIRTY_VIEWPORT) {
      t->viewport = mali_gfx_viewport_desc(cmd);
      if (!t->viewport)
         return false;
   }
   if (d->dirty & MALI_GFX_DIRTY_PASS) {
      /* Layer 0's context (the v9 file comment applies). */
      t->tiler = mali_jm_pass_tiler(r, 0);
      t->tsd = r->tsd;
   }

   /* Per draw: the vertex attributes and the varying buffers. */
   if (!emit_attributes(cmd, p, di, dd) || !emit_varying_bufs(cmd, t, di, dd))
      return false;

   d->dirty = 0;
   return true;
}

/* ---------------------------------------------------------------------- */
/* Indexed draws: the vertex range                                         */

/*
 * The smallest and largest index of the draw, read from the index
 * buffer's CPU view (restart indices skipped when primitive restart is
 * on). False when there is no CPU view, or no index is in bounds; *empty
 * when every index is a restart index (nothing to shade).
 */
static bool
index_range(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di, bool restart,
            uint32_t *min_out, uint32_t *max_out, bool *empty)
{
   const struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_jm_draw_tmpl *t = &cmd->jm.draw;
   const uint32_t size = d->ib.index_size;

   *empty = false;
   if (!t->ib_cpu || !size) {
      static bool warned;
      if (!warned) {
         warned = true;
         vk_logw(VK_LOG_OBJS(cmd),
                 "malisx2: an indexed draw on the Mali-G52 needs the index buffer's CPU "
                 "mapping to find its vertex range (the GPU helper is not ported); this "
                 "buffer has none");
      }
      return false;
   }
   const uint64_t first = (uint64_t)di->first_index * size;
   if (first >= d->ib.size) {
      *empty = true;
      return true;
   }
   /* Out-of-bounds indices are dropped (robustBufferAccess semantics are
    * the hardware's business for the in-range ones). */
   const uint32_t count = (uint32_t)MIN2(di->count, (d->ib.size - first) / size);
   const uint8_t *base = t->ib_cpu + first;
   uint32_t min = UINT32_MAX, max = 0;

#define SCAN(T, RESTART)                                                       \
   do {                                                                        \
      const T *idx = (const T *)base;                                          \
      for (uint32_t i = 0; i < count; i++) {                                   \
         const uint32_t v = idx[i];                                            \
         if (restart && v == (RESTART))                                        \
            continue;                                                          \
         min = MIN2(min, v);                                                   \
         max = MAX2(max, v);                                                   \
      }                                                                        \
   } while (0)
   switch (size) {
   case 1: SCAN(uint8_t, 0xff); break;
   case 2: SCAN(uint16_t, 0xffff); break;
   default: SCAN(uint32_t, 0xffffffffu); break;
   }
#undef SCAN
   if (min > max) {
      *empty = true;
      return true;
   }
   *min_out = min;
   *max_out = max;
   return true;
}

/* ---------------------------------------------------------------------- */
/* Debug dump                                                              */

/* MALISX2_DUMP_DRAW=1 or MESA_LOG_LEVEL=debug: every GPU address a v7 draw
 * hands the hardware, one info-level block per draw, so a translation
 * fault's address can be matched against them without a second run. */
static bool
dump_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0) {
      const char *d = getenv("MALISX2_DUMP_DRAW");
      const char *l = getenv("MESA_LOG_LEVEL");
      enabled = (d && *d && strcmp(d, "0") != 0) || (l && strcmp(l, "debug") == 0);
   }
   return enabled;
}

static void
dump_draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di, const struct draw_data *dd,
          struct mali_ptr vj, uint16_t vindex, struct mali_ptr tj, uint16_t tindex,
          uint32_t instance_size, uint16_t vs_copy, uint16_t fs_copy)
{
   const struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_render_state *r = &cmd->gfx.render;
   const struct mali_jm_draw_tmpl *t = &cmd->jm.draw;
   const struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_jm_batch *b = &cmd->jm.cur;
   struct mali_jm_device *jd = cmd->dev->jm;
   uint64_t heap_base = 0, heap_size = 0;

   if (b->heap_slot >= 0) {
      heap_base = jd->heap.bo[b->heap_slot].gpu_va;
      heap_size = jd->heap.bo[b->heap_slot].size;
   }

   mesa_logi("malisx2 v7 draw %u: %u vertices (range %u, padded %u) x %u instances, "
             "%s, offset_start %d, instance_size %u, indices 0x%" PRIx64,
             cmd->draws, di->count, dd->vertex_range, dd->padded_count, di->instance_count,
             di->indexed ? "indexed" : "plain", dd->raw_vertex_offset, instance_size, dd->indices);
   mesa_logi("  vertex job 0x%" PRIx64 " (index %u), tiler job 0x%" PRIx64 " (index %u), "
             "vtc chain first 0x%" PRIx64 ", %u jobs",
             vj.gpu, vindex, tj.gpu, tindex, b->vtc.first, b->vtc.jobs);
   mesa_logi("  vertex Draw: state (VS RSD) 0x%" PRIx64 ", attributes 0x%" PRIx64
             ", attribute_buffers 0x%" PRIx64 ", varyings 0x%" PRIx64
             ", varying_buffers 0x%" PRIx64 ", push_uniforms 0x%" PRIx64 " (%u words), "
             "thread_storage 0x%" PRIx64,
             t->vs_rsd, dd->attribs, dd->attrib_bufs, t->vs_vary, dd->vary_bufs, t->vs_fau,
             p->vs->fau.total_count, t->tsd);
   mesa_logi("  tiler Draw: state (FS RSD + %u blend) 0x%" PRIx64 " (fs code 0x%" PRIx64
             "), position 0x%" PRIx64 ", varyings 0x%" PRIx64 ", varying_buffers 0x%" PRIx64
             ", viewport 0x%" PRIx64 ", push_uniforms 0x%" PRIx64 " (%u words), "
             "thread_storage 0x%" PRIx64 ", psiz 0x%" PRIx64,
             r->rt_count, t->fs_rsd, t->fs_code, dd->position, t->fs_vary,
             t->fs_vary ? dd->vary_bufs : 0, t->viewport, t->fs_fau,
             p->fs ? p->fs->fau.total_count : 0, t->tsd, dd->psiz);
   for (unsigned st = 0; st < 2; st++) {
      const struct mali_shader *s = st ? p->fs : p->vs;
      const struct mali_v7_desc_tables *tab = st ? &t->fs_tab : &t->vs_tab;
      const uint16_t copy = st ? fs_copy : vs_copy;
      if (!s)
         continue;
      mesa_logi("  %s tables: uniform_buffers 0x%" PRIx64 " (%u + %u dynamic), textures 0x%"
                PRIx64 " (%u), samplers 0x%" PRIx64 " (%u), images 0x%" PRIx64
                " (%u, attributes 0x%" PRIx64 "), dynamic ssbos 0x%" PRIx64
                ", set mask 0x%x, constant data 0x%" PRIx64 ", copy map 0x%" PRIx64
                " (%u entries), copy job %s (index %u)",
                st ? "fragment" : "vertex", tab->tables[MALI_V7_DESC_TABLE_UBO],
                s->v7.count[MALI_V7_DESC_TABLE_UBO], s->v7.dyn_ubo_count,
                tab->tables[MALI_V7_DESC_TABLE_TEXTURE], s->info.texture_count,
                tab->tables[MALI_V7_DESC_TABLE_SAMPLER], s->info.sampler_count,
                tab->tables[MALI_V7_DESC_TABLE_IMG], s->v7.count[MALI_V7_DESC_TABLE_IMG],
                tab->img_attribs, tab->dyn_ssbos, s->desc.used_set_mask, s->v7.data_mem.gpu_va,
                s->v7.map_mem.gpu_va, s->v7.map_count,
                copy ? "in this draw" : "none in this draw", copy);
   }
   for (unsigned i = 0; i < MALI_MAX_SETS; i++) {
      const struct mali_descriptor_set *set = cmd->gfx.desc.sets[i];
      if (set)
         mesa_logi("  bound set %u: 0x%" PRIx64 " (%u slots, %u dynamic buffers)", i, set->gpu,
                   set->layout->desc_count, set->layout->dyn_buf_count);
   }
   const struct vk_vertex_input_state *vi = d->vi;
   for (unsigned i = 0; i < util_last_bit(vi->bindings_valid); i++) {
      if (vi->bindings_valid & BITFIELD_BIT(i))
         mesa_logi("  vertex binding %u: buffer 0x%" PRIx64 " size %" PRIu64 " stride %u "
                   "(attribute buffer entry %u, pointer aligned down to 0x%" PRIx64 ")",
                   i, d->vb[i].addr, d->vb[i].size, d->vi_strides[i], 2 * i,
                   (uint64_t)(d->vb[i].addr & ~63ull));
   }
   for (unsigned i = 0; i < MALI_V7_VARY_BUF_COUNT; i++)
      mesa_logi("  varying buffer %u (%s): stride %u, size %" PRIu64 "%s", i,
                i == MALI_V7_VARY_BUF_GENERAL ? "general" :
                i == MALI_V7_VARY_BUF_POSITION ? "position" : "point size",
                t->vary_stride[i], (uint64_t)dd->padded_count * di->instance_count * t->vary_stride[i],
                i == MALI_V7_VARY_BUF_POSITION ? "" : ", pointer in the buffer table");
   mesa_logi("  tiler context 0x%" PRIx64 " (%u layers, fb %ux%u), heap descriptor 0x%" PRIx64
             ", heap slot %d: base 0x%" PRIx64 " size 0x%" PRIx64 " (bottom = base, top = base + "
             "0x%llx), TSD 0x%" PRIx64 ", layer 0 context used by this draw 0x%" PRIx64,
             r->tiler, r->td_count, r->desc.width, r->desc.height, b->heap_desc, b->heap_slot,
             heap_base, heap_size, (unsigned long long)MIN2(MALI_JM_HEAP_CHUNK, heap_size),
             r->tsd, t->tiler);
   mesa_logi("  primitive word 0 0x%08x (prim %u, provoking first %u), draw word 0 bits 0x%04x, "
             "vs attribute_count %u, vs varyings %u, fs varyings %u",
             t->prim0, t->prim, r->first_provoking_vertex, t->dcd0, p->vs->info.attribute_count,
             p->vs->info.varyings.formats.count, p->fs ? p->fs->info.varyings.formats.count : 0);
   /* The RSD state words and the Blend descriptors, as the hardware reads
    * them (the RSD block is command memory: read back once for the dump). */
   const uint32_t *rsd = t->fs_rsd_cpu;
   if (rsd) {
      mesa_logi("  fs RSD words 4-11: properties 0x%08x, depth units %f factor %f clamp %f, "
                "multisample_misc 0x%08x, stencil_mask_misc 0x%08x, stencil front 0x%08x back 0x%08x, "
                "preload 0x%08x",
                rsd[4], uif(rsd[5]), uif(rsd[6]), uif(rsd[7]), rsd[8], rsd[9], rsd[10], rsd[11],
                rsd[12]);
      for (unsigned i = 0; i < r->rt_count; i++) {
         const uint32_t *bd = rsd + pan_size(RENDERER_STATE) / 4 + i * pan_size(BLEND) / 4;
         static const char *const modes[4] = {"shader", "opaque", "fixed-function", "off"};
         mesa_logi("  blend %u: flags 0x%08x equation 0x%08x internal 0x%08x 0x%08x (%s%s)", i,
                   bd[0], bd[1], bd[2], bd[3], modes[bd[2] & 3],
                   (d->blend_shader_mask & BITFIELD_BIT(i)) ? ", blend shader" : "");
      }
   }
}

/* ---------------------------------------------------------------------- */
/* The jobs                                                                */

static void
draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_jm_draw_tmpl *t = &cmd->jm.draw;

   if (!di->count || !di->instance_count)
      return;
   if (!r->active || !d->pipeline || !d->pipeline->vs) {
      vk_logw(VK_LOG_OBJS(cmd), "malisx2: draw outside a render pass or without a pipeline");
      return;
   }
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   const struct mali_graphics_pipeline *p = d->pipeline;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   /* States the bind does not copy into the command buffer come from the
    * pipeline unless it left them dynamic (mali_cmd_bind_graphics). */
   const struct vk_dynamic_graphics_state *st = p->bind.dynamic ? dyn : &p->state;

   struct draw_data dd;
   memset(&dd, 0, sizeof(dd));
   dd.vertex_range = di->count;
   dd.raw_vertex_offset = di->vertex_offset;
   if (di->indexed) {
      uint32_t min, max;
      bool empty;
      if (!index_range(cmd, di, st->ia.primitive_restart_enable, &min, &max, &empty)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      if (empty)
         return;
      dd.vertex_range = max - min + 1;
      dd.raw_vertex_offset = (int32_t)min + di->vertex_offset;
      dd.indices = d->ib.addr + (uint64_t)di->first_index * d->ib.index_size;
   }
   dd.padded_count = di->instance_count > 1 ? padded_vertex_count(dd.vertex_range) : dd.vertex_range;

   if (!prepare_draw(cmd, di, &dd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   /* Rasterizer discard: the vertex job alone, as panvk (no occlusion
    * queries to keep the tiler job for). */
   const bool tiling = !st->rs.rasterizer_discard_enable;

   struct mali_invocation_packed invocation;
   pack_invocation(&invocation, dd.vertex_range, di->instance_count);
   const uint32_t instance_size = di->instance_count > 1 ? dd.padded_count : 1;

   struct mali_jm_chain *c = mali_jm_cmd_pass_vtc(cmd);

   /* The descriptor copies of the stages whose tables were rebuilt, in
    * front of the Vertex job (panvk_cmd_draw): the Vertex job depends on
    * the vertex shader's; the fragment shader's tables are read by the
    * fragment job, which follows the whole vtc atom. A barrier pending on
    * the chain goes to the first of these jobs; when that is the fragment
    * copy, the Vertex job, which does not depend on it, carries the
    * barrier too. */
   const bool pending = c->pending != 0;
   uint16_t vs_copy = 0, fs_copy = 0;
   if (dd.vs_copy &&
       !mali_v7_desc_copy(cmd, c, p->vs, &cmd->gfx.desc, &t->vs_tab, dd.vs_img_buf_offset,
                          &vs_copy))
      return;
   if (dd.fs_copy && !mali_v7_desc_copy(cmd, c, p->fs, &cmd->gfx.desc, &t->fs_tab, 0, &fs_copy))
      return;

   struct mali_ptr vj = mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_VERTEX, pan_size(COMPUTE_JOB),
                                            pending && !vs_copy && fs_copy, vs_copy);
   if (!vj.cpu)
      return;
   const uint16_t vertex_index = c->index;

   /* The Vertex job (panvk_draw_prepare_vertex_job, panvk_emit_vertex_dcd). */
   memcpy(pan_section_ptr(vj.cpu, COMPUTE_JOB, INVOCATION), &invocation, sizeof(invocation));
   pan_section_pack(vj.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = 5;
   }
   pan_section_pack(vj.cpu, COMPUTE_JOB, DRAW, cfg) {
      cfg.state = t->vs_rsd;
      cfg.attributes = dd.attribs;
      cfg.attribute_buffers = dd.attrib_bufs;
      cfg.varyings = t->vs_vary;
      cfg.varying_buffers = dd.vary_bufs;
      cfg.thread_storage = t->tsd;
      cfg.offset_start = dd.raw_vertex_offset;
      cfg.instance_size = instance_size;
      cfg.push_uniforms = t->vs_fau;
      cfg.uniform_buffers = t->vs_tab.tables[MALI_V7_DESC_TABLE_UBO];
      cfg.textures = t->vs_tab.tables[MALI_V7_DESC_TABLE_TEXTURE];
      cfg.samplers = t->vs_tab.tables[MALI_V7_DESC_TABLE_SAMPLER];
   }

   if (tiling) {
      struct mali_ptr tj = mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_TILER, pan_size(TILER_JOB),
                                               false, vertex_index);
      if (!tj.cpu)
         return;

      memcpy(pan_section_ptr(tj.cpu, TILER_JOB, INVOCATION), &invocation, sizeof(invocation));

      /* panvk_emit_tiler_primitive: the baked flags, this draw's index
       * type, the pass's provoking vertex, and the counts. */
      struct mali_primitive_packed prim;
      pan_pack(&prim, PRIMITIVE, cfg) {
         cfg.index_count = di->count;
         if (di->indexed) {
            cfg.indices = dd.indices;
            /* Index + base_vertex_offset selects the shaded vertex:
             * vertexOffset minus the shading range's start (libpan
             * draw_helper.cl). */
            cfg.base_vertex_offset = (int64_t)di->vertex_offset - dd.raw_vertex_offset;
         }
      }
      prim.opaque[0] = (t->prim0 & ~prim_provoking_bit()) |
                       (di->indexed ? (uint32_t)mali_gfx_index_type(d->ib.index_size) << 8 : 0) |
                       (r->first_provoking_vertex ? prim_provoking_bit() : 0);
      memcpy(pan_section_ptr(tj.cpu, TILER_JOB, PRIMITIVE), &prim, sizeof(prim));

      pan_section_pack(tj.cpu, TILER_JOB, PRIMITIVE_SIZE, cfg) {
         if (t->writes_psiz)
            cfg.size_array = dd.psiz;
         else
            cfg.fixed_sized = t->lines ? dyn->rs.line.width : 1.0f;
      }
      pan_section_pack(tj.cpu, TILER_JOB, TILER, cfg) {
         cfg.address = t->tiler;
      }
      pan_section_pack(tj.cpu, TILER_JOB, PADDING, cfg);

      /* panvk_emit_tiler_dcd, with the baked culling bits ORed into word 0. */
      const uint32_t prims_per_instance =
         DIV_ROUND_UP(dd.padded_count, mesa_vertices_per_prim(t->prim));
      struct mali_draw_packed dcd;
      pan_pack(&dcd, DRAW, cfg) {
         cfg.position = dd.position;
         cfg.state = t->fs_rsd;
         /* The fragment shader's images (panvk_emit_tiler_dcd). */
         cfg.attributes = t->fs_tab.img_attribs;
         cfg.attribute_buffers = t->fs_tab.tables[MALI_V7_DESC_TABLE_IMG];
         cfg.viewport = t->viewport;
         cfg.varyings = t->fs_vary;
         cfg.varying_buffers = t->fs_vary ? dd.vary_bufs : 0;
         cfg.thread_storage = t->tsd;
         cfg.offset_start = dd.raw_vertex_offset;
         cfg.instance_size = instance_size;
         cfg.instance_primitive_size = padded_vertex_count(MAX2(prims_per_instance, 1));
         cfg.push_uniforms = t->fs_fau;
         cfg.uniform_buffers = t->fs_tab.tables[MALI_V7_DESC_TABLE_UBO];
         cfg.textures = t->fs_tab.tables[MALI_V7_DESC_TABLE_TEXTURE];
         cfg.samplers = t->fs_tab.tables[MALI_V7_DESC_TABLE_SAMPLER];
         cfg.occlusion_query = MALI_OCCLUSION_MODE_DISABLED;
      }
      dcd.opaque[0] |= t->dcd0;
      memcpy(pan_section_ptr(tj.cpu, TILER_JOB, DRAW), &dcd, sizeof(dcd));

      if (unlikely(dump_enabled()))
         dump_draw(cmd, di, &dd, vj, vertex_index, tj, c->index, instance_size, vs_copy,
                   fs_copy);
   } else if (unlikely(dump_enabled())) {
      dump_draw(cmd, di, &dd, vj, vertex_index, (struct mali_ptr){0}, 0, instance_size,
                vs_copy, fs_copy);
   }

   /* Heap use: polygon-list entries for the primitives, as the v9 guess. */
   mali_jm_cmd_heap_use(cmd, (uint64_t)di->count * di->instance_count * 32);
   cmd->jm.cur.draws++;
   r->draws++;
   cmd->draws++;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDraw)(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                       uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const struct mali_draw_info di = {
      .count = vertexCount,
      .instance_count = instanceCount,
      .vertex_offset = (int32_t)firstVertex,
      .first_instance = firstInstance,
   };
   draw(cmd, &di);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDrawIndexed)(VkCommandBuffer commandBuffer, uint32_t indexCount,
                              uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
                              uint32_t firstInstance)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const struct mali_draw_info di = {
      .count = indexCount,
      .instance_count = instanceCount,
      .first_index = firstIndex,
      .vertex_offset = vertexOffset,
      .first_instance = firstInstance,
      .indexed = true,
   };
   draw(cmd, &di);
}

#else /* PAN_ARCH >= 9 */

/* ---------------------------------------------------------------------- */
/* The template's layout                                                   */

/* 32-bit word indices into the Malloc Vertex job (v9.xml). */
#define MV_WORD(section) (MALI_MALLOC_VERTEX_JOB_SECTION_##section##_OFFSET / 4)
enum {
   TW_HDR_TYPE_INDEX = 4,              /* Type, Barrier, Index */
   TW_HDR_DEPS = 5,                    /* Dependency 1, Dependency 2 */
   TW_PRIM = MV_WORD(PRIMITIVE),       /* flags; +1 base vertex offset;
                                        * +2 instance offset; +3 index count */
   TW_INSTANCES = MV_WORD(INSTANCE_COUNT),
   TW_ALLOC = MV_WORD(ALLOCATION),
   TW_TILER = MV_WORD(TILER),          /* 64-bit */
   TW_SCISSOR = MV_WORD(SCISSOR),      /* 64-bit */
   TW_PSIZE = MV_WORD(PRIMITIVE_SIZE), /* the fixed size, a float */
   TW_INDICES = MV_WORD(INDICES),      /* 64-bit */
   TW_DRAW = MV_WORD(DRAW),
   /* The Draw descriptor's words. */
   TW_DCD0 = TW_DRAW + 0,
   TW_DCD1 = TW_DRAW + 1,
   TW_VARRAY = TW_DRAW + 2,            /* 3 words, GPU-written */
   TW_MIN_Z = TW_DRAW + 6,
   TW_MAX_Z = TW_DRAW + 7,
   TW_ZSD = TW_DRAW + 10,              /* 64-bit */
   TW_BLEND = TW_DRAW + 12,            /* 64-bit: address | count */
   TW_FS_ENV = TW_DRAW + 16,
   TW_POS_ENV = MV_WORD(POSITION),
   TW_VAR_ENV = MV_WORD(VARYING),
};

/* Shader Environment words (16 words). */
enum {
   ENV_FAU_COUNT = 1,
   ENV_RESOURCES = 8,                  /* 64-bit */
   ENV_SHADER = 10,                    /* 64-bit */
   ENV_TSD = 12,                       /* 64-bit */
   ENV_FAU = 14,                       /* 64-bit */
   ENV_WORDS = 16,
};

_Static_assert(MALI_MALLOC_VERTEX_JOB_LENGTH == sizeof(((struct mali_jm_draw_tmpl *)0)->w),
               "the template is one Malloc Vertex job");
_Static_assert(TW_PRIM == 8 && TW_INSTANCES == 12 && TW_ALLOC == 13 && TW_TILER == 14 &&
                  TW_SCISSOR == 26 && TW_PSIZE == 28 && TW_INDICES == 30 && TW_DRAW == 32 &&
                  TW_POS_ENV == 64 && TW_VAR_ENV == 80,
               "v9.xml Malloc Vertex Job layout");

static inline void
tset64(uint32_t *w, unsigned i, uint64_t v)
{
   memcpy(&w[i], &v, sizeof(v));
}

/* The FAU word of the builders (address | count << 56) into a Shader
 * Environment's FAU pointer and FAU count. */
static inline void
env_set_fau(uint32_t *env, uint64_t fau)
{
   env[ENV_FAU_COUNT] = (uint32_t)(fau >> 56);
   tset64(env, ENV_FAU, fau & BITFIELD64_MASK(56));
}

void
mali_jm_draw_tmpl_init(struct mali_jm_draw_tmpl *t)
{
   memset(t, 0, sizeof(*t));

   struct mali_job_header_packed h;
   pan_pack(&h, JOB_HEADER, cfg) {
      cfg.type = MALI_JOB_TYPE_MALLOC_VERTEX;
   }
   t->hdr4 = h.opaque[TW_HDR_TYPE_INDEX];

   /* The Vertex array words the hardware overwrites (file comment). */
   struct mali_vertex_array_packed va;
   pan_pack(&va, VERTEX_ARRAY, cfg) {
      cfg.packet = true;
   }
   memcpy(&t->w[TW_VARRAY], &va, sizeof(va));
}

/* The Primitive word's Secondary Shader bit. */
static inline uint32_t
prim_secondary_bit(void)
{
   struct mali_primitive_packed p;
   pan_pack_nodefaults(&p, PRIMITIVE, cfg) {
      cfg.secondary_shader = true;
   }
   return p.opaque[0];
}

/* The Primitive word's Point Size Array Format field: when it is not
 * NONE, the shader writes the point size and the job's Primitive Size
 * section holds the size array pointer instead of a fixed size. */
static inline uint32_t
prim_point_size_array_mask(void)
{
   struct mali_primitive_packed p;
   pan_pack_nodefaults(&p, PRIMITIVE, cfg) {
      cfg.point_size_array_format = MALI_POINT_SIZE_ARRAY_FORMAT_FP32;   /* both bits */
   }
   return p.opaque[0];
}

/* ---------------------------------------------------------------------- */
/* Draw state into the template                                            */

/*
 * The draw's state (mali_cmd_state.h, "A draw's state"): the shared
 * builders for every dirty group, each result stored into the template
 * right after it is built. Each group owns its words; words no group owns
 * are constant (mali_jm_draw_tmpl_init) or per draw.
 */
static bool
prepare_draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_jm_draw_tmpl *t = &cmd->jm.draw;
   uint32_t *w = t->w;
   uint32_t *const fs_env = &w[TW_FS_ENV];
   uint32_t *const pos_env = &w[TW_POS_ENV];
   const struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_gfx_bind *bd = &p->bind;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct mali_gfx_baked *dbk;
   uint64_t v;

   if (!mali_gfx_update_dirty(cmd, di, &dbk))
      return false;

   if (d->dirty & MALI_GFX_DIRTY_BLEND) {
      if (!mali_gfx_build_blend(cmd, &v))
         return false;
      tset64(w, TW_BLEND, v);
   }

   struct mali_graphics_sysvals sv;
   if (d->dirty & (MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_FS_FAU))
      mali_gfx_fill_sysvals(cmd, &sv);

   if (d->dirty & MALI_GFX_DIRTY_VS_FAU) {
      if (!mali_gfx_build_vs_fau(cmd, &sv, &v))
         return false;
      env_set_fau(pos_env, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_FAU) {
      if (!mali_gfx_build_fs_fau(cmd, &sv, &v))
         return false;
      /* v9 shaders keep no uniform-buffer words in FAU (no stream to
       * copy them with). */
      assert(!p->fs || !p->fs->fau.ubo_push_count);
      env_set_fau(fs_env, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_VS_SRT) {
      if (!mali_gfx_build_vs_srt(cmd, &v))
         return false;
      tset64(pos_env, ENV_RESOURCES, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_SRT) {
      if (!mali_gfx_build_fs_srt(cmd, &v))
         return false;
      tset64(fs_env, ENV_RESOURCES, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_ZSD) {
      if (!mali_gfx_build_zsd(cmd, dbk, &v))
         return false;
      tset64(w, TW_ZSD, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_PIPELINE) {
      bool lines;
      uint64_t spd_var;
      if (unlikely(dbk)) {
         tset64(pos_env, ENV_SHADER, dbk->vs_pos_spd);
         tset64(fs_env, ENV_SHADER, dbk->fs_spd);
         spd_var = dbk->vs_var_spd;
         t->prim0 = dbk->tiler_flags;
         w[TW_DCD0] = dbk->dcd0[0];
         w[TW_DCD1] = dbk->dcd1;
         lines = u_reduced_prim(dbk->prim) == MESA_PRIM_LINES;
      } else {
         tset64(pos_env, ENV_SHADER, bd->spd[0]);
         tset64(fs_env, ENV_SHADER, bd->spd[2]);
         spd_var = bd->spd[1];
         t->prim0 = bd->tiler_flags;
         w[TW_DCD0] = bd->dcd0;
         w[TW_DCD1] = bd->dcd1;
         lines = bd->lines;
      }
      t->secondary = t->prim0 & prim_secondary_bit();
      /* The vertex packet: the varyings after a 16-byte position, or
       * position only. */
      t->packet_stride = t->secondary ? bd->vary_size + 16 : 16;
      w[TW_ALLOC] = t->packet_stride | (t->secondary ? bd->vary_size : 0) << 16;
      /* Lines: the line width. Points whose shader writes the size: the
       * field is the size array pointer, not used when the sizes come in
       * the vertex packets; 0, as Mesa writes, not a float's bits. Other
       * points: 1.0. */
      if (t->prim0 & prim_point_size_array_mask())
         w[TW_PSIZE] = 0;
      else
         w[TW_PSIZE] = fui(lines ? dyn->rs.line.width : 1.0f);
      tset64(&w[TW_VAR_ENV], ENV_SHADER, t->secondary ? spd_var : 0);
   }
   if (d->dirty & MALI_GFX_DIRTY_VIEWPORT) {
      const VkViewport *vp = &dyn->vp.viewports[0];
      tset64(w, TW_SCISSOR, mali_gfx_scissor_box(cmd));
      w[TW_MIN_Z] = fui(MIN2(vp->minDepth, vp->maxDepth));
      w[TW_MAX_Z] = fui(MAX2(vp->minDepth, vp->maxDepth));
   }
   if (d->dirty & MALI_GFX_DIRTY_PASS) {
      /* Layer 0's context (file comment). */
      tset64(w, TW_TILER, r->tiler);
      tset64(fs_env, ENV_TSD, r->tsd);
      tset64(pos_env, ENV_TSD, r->tsd);
   }
   /* The varying environment: the position one's resources, FAU and
    * thread storage with the varying shader, or nothing. */
   if (d->dirty & (MALI_GFX_DIRTY_PIPELINE | MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_VS_SRT |
                   MALI_GFX_DIRTY_PASS)) {
      uint32_t *var_env = &w[TW_VAR_ENV];
      if (t->secondary) {
         var_env[ENV_FAU_COUNT] = pos_env[ENV_FAU_COUNT];
         memcpy(&var_env[ENV_RESOURCES], &pos_env[ENV_RESOURCES], 8);
         memcpy(&var_env[ENV_TSD], &pos_env[ENV_TSD], 16);   /* TSD, FAU */
      } else {
         memset(var_env, 0, ENV_WORDS * 4);
      }
   }

   d->dirty = 0;
   return true;
}

/* ---------------------------------------------------------------------- */
/* Writing the job                                                         */

/* The 384 bytes as six 64-byte stores (one ST1 of four registers each, a
 * whole cache line of the 128-byte-aligned job) in address order: the
 * compiler may not reorder or split them (a library memcpy of this size
 * may store its last block first, or twice; clang reorders the stores of
 * an inlined one), so write-combined memory gets full lines, one after
 * the other. */
static inline void
copy_job(void *restrict dst, const void *restrict src)
{
#if defined(__aarch64__)
   for (unsigned i = 0; i < 6; i++) {
      const uint8x16x4_t v = vld1q_u8_x4((const uint8_t *)src + 64 * i);
      vst1q_u8_x4((uint8_t *)dst + 64 * i, v);
      __asm__ volatile("" ::: "memory");
   }
#else
   memcpy(dst, src, 384);
#endif
}

/* The template as the next job of chain c. */
static bool
emit_job(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c)
{
   struct mali_jm_draw_tmpl *t = &cmd->jm.draw;
   struct mali_ptr job;

   if (likely(!c->pending && c->index < UINT16_MAX - 2)) {
      job = mali_cmd_alloc(cmd, pan_size(MALLOC_VERTEX_JOB), 128);
      if (!job.cpu)
         return false;
      const uint16_t index = ++c->index;
      t->w[TW_HDR_TYPE_INDEX] = t->hdr4 | (uint32_t)index << 16;
      t->w[TW_HDR_DEPS] = (uint32_t)c->tiler_dep << 16;   /* Dependency 2 */
      copy_job(job.cpu, t->w);
      mali_jm_chain_link(c, job, index, true);
      if (unlikely(cmd->jm.resubmit))
         mali_jm_cmd_note_reset(cmd, job.cpu, NULL, 16);
   } else {
      /* A barrier owed to this chain (a Cache Flush job first), or a
       * chain out of indices (an error): the general path. */
      job = mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_MALLOC_VERTEX,
                                pan_size(MALLOC_VERTEX_JOB), false, 0);
      if (!job.cpu)
         return false;
      memcpy((uint8_t *)job.cpu + 32, (const uint8_t *)t->w + 32,
             pan_size(MALLOC_VERTEX_JOB) - 32);
   }
   if (unlikely(cmd->jm.resubmit))
      mali_jm_cmd_note_reset(cmd, (uint8_t *)job.cpu + TW_VARRAY * 4, &t->w[TW_VARRAY], 12);
   return true;
}

static void
draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;

   if (!di->count || !di->instance_count)
      return;
   if (!r->active || !d->pipeline || !d->pipeline->vs) {
      vk_logw(VK_LOG_OBJS(cmd), "malisx2: draw outside a render pass or without a pipeline");
      return;
   }
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   if (!prepare_draw(cmd, di)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   /* The draw's own words. */
   struct mali_jm_draw_tmpl *t = &cmd->jm.draw;
   uint32_t *w = t->w;
   uint64_t indices = 0;
   uint32_t index_type = MALI_INDEX_TYPE_NONE;
   if (di->indexed) {
      index_type = mali_gfx_index_type(d->ib.index_size);
      indices = d->ib.addr + (uint64_t)di->first_index * d->ib.index_size;
   }
   w[TW_PRIM] = t->prim0 | index_type << 8;   /* Index type, bits 8-10 */
   w[TW_PRIM + 1] = (uint32_t)di->vertex_offset;
   w[TW_PRIM + 3] = di->count;
   w[TW_INSTANCES] = di->instance_count;
   tset64(w, TW_INDICES, indices);

   if (!emit_job(cmd, mali_jm_cmd_pass_vtc(cmd)))
      return;

   /* Heap use: every vertex's packet (with an index count, an upper
    * bound) and about as much again for polygon-list entries. A guess
    * for closing batches early enough, tuned on the device. */
   mali_jm_cmd_heap_use(cmd, (uint64_t)di->count * di->instance_count *
                                (t->packet_stride + 16));
   cmd->jm.cur.draws++;
   r->draws++;
   cmd->draws++;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDraw)(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                       uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const struct mali_draw_info di = {
      .count = vertexCount,
      .instance_count = instanceCount,
      .vertex_offset = (int32_t)firstVertex,
      .first_instance = firstInstance,
   };
   draw(cmd, &di);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDrawIndexed)(VkCommandBuffer commandBuffer, uint32_t indexCount,
                              uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
                              uint32_t firstInstance)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const struct mali_draw_info di = {
      .count = indexCount,
      .instance_count = instanceCount,
      .first_index = firstIndex,
      .vertex_offset = vertexOffset,
      .first_instance = firstInstance,
      .indexed = true,
   };
   draw(cmd, &di);
}

#endif /* PAN_ARCH >= 9 */
