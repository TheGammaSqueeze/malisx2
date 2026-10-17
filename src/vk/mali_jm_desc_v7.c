/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * The Bifrost (v7) descriptor tables of a draw or dispatch, built by one
 * builder for both (mali_jm.h, struct mali_v7_desc_tables), and the
 * compute job sizing helpers. Built only at PAN_ARCH 7: src/meson.build
 * adds this file to the v7 job-manager library alone.
 *
 * Ported from panvk's Bifrost job-manager path in the pinned Mesa:
 * src/panfrost/vulkan/panvk_vX_cmd_desc_state.c
 * (cmd_prepare_shader_desc_tables, cmd_prepare_dyn_ssbos,
 * panvk_cmd_fill_dyn_ubos), vulkan/bifrost/panvk_vX_meta_desc_copy.c (the
 * copy shader and its job), vulkan/panvk_vX_cmd_dispatch.c and
 * panvk_vX_cmd_draw.c (the desc.sets system values) and lib/pan_encoder.h
 * (pan_pack_work_groups_compute), onto this driver's command memory,
 * shader objects and job chains.
 *
 * The model, the same for the vertex and fragment stage of a draw and for
 * a dispatch:
 *
 *  - mali_v7_desc_prepare allocates the stage's tables in command memory
 *    and writes what only the CPU knows: the dynamic uniform buffers
 *    (behind the regular ones in the Uniform Buffer table) and the dynamic
 *    storage buffer table, both with the bound dynamic offsets applied, the
 *    dummy sampler, and the bound sets' addresses;
 *  - mali_v7_desc_copy fills the entries that come from the sets (uniform
 *    buffers, textures, samplers, images and texel buffers, in the order of
 *    the shader's copy map) with one small Compute job in the same chain.
 *    The job that reads the tables depends on it: the Vertex job on the
 *    vertex shader's, the dispatch on the compute shader's. A fragment
 *    shader's tables are read by the fragment job, which starts only once
 *    the whole vtc atom is done, so the Tiler job needs no dependency on
 *    that copy job (panvk adds none either);
 *  - storage buffers and size queries are not copied: the shader reads the
 *    set itself through the set addresses in its system values
 *    (mali_v7_desc_sysvals).
 *
 * The copy runs on the GPU, as panvk's does, so the tables hold the sets as
 * they are when the command buffer executes. A copy on the CPU when the
 * command is recorded would be as correct (no update-after-bind feature is
 * exposed, so a set must not change while a command buffer that bound it
 * is pending), but set memory is CPU-uncached and each entry would be an
 * uncached read. MALISX2_V7_CPU_DESC_COPY=1 does exactly that, as a
 * debugging switch that takes the copy shader out of a failing run.
 */

#include "mali_cmd_state.h"

#if PAN_ARCH >= 9
#error "mali_jm_desc_v7.c is the Bifrost (PAN_ARCH 7) descriptor model"
#endif

#include <stdlib.h>
#include <string.h>

#include "nir_builder.h"
#include "util/log.h"
#include "util/mesa-blake3.h"
#include "vk_log.h"
#include "vk_pipeline.h"

#include "mali_arch.h"
#include "mali_memory.h"

/* The address of a dynamic storage buffer bound to nothing: the GPU's
 * memory sink (panvk). */
#define NULL_BUFFER_ADDR (0x8ull << 60)

/* A storage buffer slot of a set, and an entry of the dynamic storage
 * buffer table (panvk_ssbo_addr): what the shader's load_vulkan_descriptor
 * reads (mali_nir_lower_descriptors_v7.c). */
struct ssbo_addr {
   uint64_t base_addr;
   uint32_t size;
   uint32_t zero[5];
};
static_assert(sizeof(struct ssbo_addr) == MALI_DESCRIPTOR_SIZE, "one slot");

/* ---------------------------------------------------------------------- */
/* Compute job sizing                                                      */

/* pan_pack_work_groups_compute for a dispatch: no graphics quirks, and the
 * thread group split equals the workgroup X shift so barriers work. */
void
mali_v7_compute_invocation(struct mali_invocation_packed *out, const uint32_t local[3],
                           const uint32_t groups[3])
{
   const unsigned values[6] = {local[0], local[1], local[2], groups[0], groups[1], groups[2]};
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
      cfg.workgroups_z_shift = shifts[5];
      cfg.thread_group_split = cfg.workgroups_x_shift;
   }
}

/* panvk's job_task_split for a dispatch. */
unsigned
mali_v7_compute_task_split(const uint32_t local[3])
{
   return util_logbase2_ceil(local[0] + 1) + util_logbase2_ceil(local[1] + 1) +
          util_logbase2_ceil(local[2] + 1);
}

/* ---------------------------------------------------------------------- */
/* The bound sets                                                          */

/* The slot a copy handle names, NULL when its set is not bound or is
 * smaller than the layout the shader was compiled against (the CPU fill;
 * the copy shader makes the same test with the set's slot count). */
static const uint8_t *
set_slot(const struct mali_desc_state *desc, uint32_t handle)
{
   const uint32_t set = MALI_COPY_DESC_HANDLE_SET(handle);
   const uint32_t idx = MALI_COPY_DESC_HANDLE_INDEX(handle);
   const struct mali_descriptor_set *s = desc && set < MALI_MAX_SETS ? desc->sets[set] : NULL;
   if (!s || !s->cpu || idx >= s->layout->desc_count)
      return NULL;
   return (const uint8_t *)s->cpu + (size_t)idx * MALI_DESCRIPTOR_SIZE;
}

/* The dynamic buffer a copy handle names: its address with the bound
 * dynamic offset applied, and its range. False when its set is not
 * bound. */
static bool
dyn_buf(const struct mali_desc_state *desc, uint32_t handle, uint64_t *addr, uint64_t *range)
{
   const uint32_t set = MALI_COPY_DESC_HANDLE_SET(handle);
   const uint32_t idx = MALI_COPY_DESC_HANDLE_INDEX(handle);
   const struct mali_descriptor_set *s = desc && set < MALI_MAX_SETS ? desc->sets[set] : NULL;
   if (!s || idx >= s->layout->dyn_buf_count)
      return false;
   *addr = s->dyn_bufs[idx].addr + desc->dyn_offsets[set][idx];
   *range = s->dyn_bufs[idx].range;
   return true;
}

/* ---------------------------------------------------------------------- */
/* The tables                                                              */

static unsigned
table_entry_size(unsigned t)
{
   return t == MALI_V7_DESC_TABLE_UBO ? pan_size(UNIFORM_BUFFER) : MALI_DESCRIPTOR_SIZE;
}

/* The dynamic uniform buffers' entries, offsets applied
 * (panvk_cmd_fill_dyn_ubos), packed into dst. One bound to nothing (or to
 * the null buffer, whose sink address does not fit the descriptor) reads
 * as one zero entry. */
static void
pack_dyn_ubos(const struct mali_shader *s, const struct mali_desc_state *desc, void *dst)
{
   const uint32_t dyn_ubo_count = s->v7.dyn_ubo_count;
   const uint32_t *dyn_ubo_map = s->desc.dyn_bufs.map;
   /* Packed in cached memory and copied out once: command memory is
    * uncached. */
   struct mali_uniform_buffer_packed dyn[MALI_MAX_DYNAMIC_UNIFORM_BUFFERS];
   assert(dyn_ubo_count <= ARRAY_SIZE(dyn));
   for (uint32_t i = 0; i < dyn_ubo_count; i++) {
      uint64_t addr = 0, range = 0;
      if (!dyn_buf(desc, dyn_ubo_map[i], &addr, &range) || (addr >> 56))
         addr = range = 0;
      pan_pack(&dyn[i], UNIFORM_BUFFER, cfg) {
         cfg.pointer = addr;
         /* 16-byte entries, at least one (the packer encodes the count
          * minus one), at most the 12-bit field: 64 KiB, the
          * maxUniformBufferRange of this arch. */
         cfg.entries = CLAMP(DIV_ROUND_UP(range, 16), 1, 1u << 12);
      }
   }
   memcpy(dst, dyn, dyn_ubo_count * sizeof(dyn[0]));
}

/* The dynamic storage buffer table: address and size, offset applied
 * (cmd_prepare_dyn_ssbos). 0 on failure. */
static uint64_t
build_dyn_ssbos(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                const struct mali_desc_state *desc)
{
   const uint32_t dyn_ubo_count = s->v7.dyn_ubo_count;
   const uint32_t dyn_ssbo_count = s->desc.dyn_bufs.count - dyn_ubo_count;
   const uint32_t *dyn_ssbo_map = s->desc.dyn_bufs.map + dyn_ubo_count;
   struct mali_ptr p =
      mali_cmd_alloc(cmd, (uint64_t)dyn_ssbo_count * sizeof(struct ssbo_addr), 64);
   if (!p.cpu)
      return 0;
   struct ssbo_addr entries[MALI_MAX_DYNAMIC_STORAGE_BUFFERS];
   assert(dyn_ssbo_count <= ARRAY_SIZE(entries));
   memset(entries, 0, sizeof(entries));
   for (uint32_t i = 0; i < dyn_ssbo_count; i++) {
      uint64_t addr = 0, range = 0;
      entries[i].base_addr = NULL_BUFFER_ADDR;
      if (dyn_buf(desc, dyn_ssbo_map[i], &addr, &range) && range) {
         entries[i].base_addr = addr;
         entries[i].size = (uint32_t)MIN2(align64(range, 4), UINT32_MAX & ~3ull);
      }
   }
   memcpy(p.cpu, entries, dyn_ssbo_count * sizeof(entries[0]));
   return p.gpu;
}

static bool cpu_copy_mode(void);

bool
mali_v7_desc_refresh(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                     const struct mali_desc_state *desc, bool img,
                     struct mali_v7_desc_tables *t)
{
   const struct mali_v7_shader_desc *v7 = &s->v7;
   if (t->shader != s || v7->count[MALI_V7_DESC_TABLE_UBO] ||
       (!img && v7->count[MALI_V7_DESC_TABLE_IMG]) || cpu_copy_mode())
      return false;
   for (unsigned i = 0; i < MALI_MAX_SETS; i++) {
      const struct mali_descriptor_set *set = desc ? desc->sets[i] : NULL;
      if (t->sets[i] != (set ? set->gpu : 0) ||
          t->set_desc_counts[i] != (set ? set->layout->desc_count : 0))
         return false;
   }

   const uint32_t dyn_ubo_count = v7->dyn_ubo_count;
   const uint32_t dyn_ssbo_count = s->desc.dyn_bufs.count - dyn_ubo_count;
   uint64_t ubo = t->tables[MALI_V7_DESC_TABLE_UBO];
   void *ubo_cpu = t->tables_cpu[MALI_V7_DESC_TABLE_UBO];
   if (dyn_ubo_count) {
      struct mali_ptr p = mali_cmd_alloc(cmd, (uint64_t)dyn_ubo_count * pan_size(UNIFORM_BUFFER), 64);
      if (!p.cpu)
         return false;
      pack_dyn_ubos(s, desc, p.cpu);
      ubo = p.gpu;
      ubo_cpu = p.cpu;
   }
   uint64_t dyn_ssbos = t->dyn_ssbos;
   if (dyn_ssbo_count) {
      dyn_ssbos = build_dyn_ssbos(cmd, s, desc);
      if (!dyn_ssbos)
         return false;
   }
   t->tables[MALI_V7_DESC_TABLE_UBO] = ubo;
   t->tables_cpu[MALI_V7_DESC_TABLE_UBO] = ubo_cpu;
   t->dyn_ssbos = dyn_ssbos;
   return true;
}

bool
mali_v7_desc_prepare(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                     const struct mali_desc_state *desc, bool img,
                     struct mali_v7_desc_tables *out)
{
   const struct mali_v7_shader_desc *v7 = &s->v7;
   /* desc.dyn_bufs.map: the dynamic uniform buffers first, then the
    * dynamic storage buffers (mali_nir_lower_descriptors_v7.c). */
   const uint32_t dyn_ubo_count = v7->dyn_ubo_count;
   const uint32_t dyn_ssbo_count = s->desc.dyn_bufs.count - dyn_ubo_count;

   memset(out, 0, sizeof(*out));
   out->shader = s;
   if (!s->desc.used_set_mask && !v7->map_count && !s->desc.dyn_bufs.count)
      return true;

   /* Every bound set: the copy shader's sources (panvk passes them all). */
   for (unsigned i = 0; i < MALI_MAX_SETS; i++) {
      const struct mali_descriptor_set *set = desc ? desc->sets[i] : NULL;
      out->sets[i] = set ? set->gpu : 0;
      out->set_desc_counts[i] = set ? set->layout->desc_count : 0;
   }

   for (unsigned t = 0; t < MALI_V7_DESC_TABLE_COUNT; t++) {
      const uint32_t regular = v7->count[t];
      const uint32_t count = regular + (t == MALI_V7_DESC_TABLE_UBO ? dyn_ubo_count : 0);
      const unsigned entry = table_entry_size(t);
      if (!count || (t == MALI_V7_DESC_TABLE_IMG && !img))
         continue;

      /* The image table is an Attribute Buffer table: an empty entry after
       * the pairs stops the Bifrost prefetcher, as behind the vertex and
       * varying buffers of a draw. */
      const uint64_t tail = t == MALI_V7_DESC_TABLE_IMG ? pan_size(ATTRIBUTE_BUFFER) : 0;
      struct mali_ptr p = mali_cmd_alloc(cmd, (uint64_t)count * entry + tail, 64);
      if (!p.cpu)
         return false;
      out->tables[t] = p.gpu;
      out->tables_cpu[t] = p.cpu;

      /* The dynamic uniform buffers behind the regular ones. */
      if (t == MALI_V7_DESC_TABLE_UBO && dyn_ubo_count)
         pack_dyn_ubos(s, desc, (uint8_t *)p.cpu + (uint64_t)regular * entry);

      if (t == MALI_V7_DESC_TABLE_IMG) {
         memset((uint8_t *)p.cpu + (uint64_t)count * entry, 0, tail);
         /* One Attribute descriptor per image, written by the copy next to
          * the Attribute Buffer pairs. */
         struct mali_ptr a = mali_cmd_alloc(cmd, (uint64_t)count * pan_size(ATTRIBUTE), 64);
         if (!a.cpu)
            return false;
         out->img_attribs = a.gpu;
         out->img_attribs_cpu = a.cpu;
      }
   }

   /* A shader with textures and no sampler of its own gets one: texel
    * fetches still name sampler 0 (panvk). */
   if (v7->count[MALI_V7_DESC_TABLE_TEXTURE] && !v7->count[MALI_V7_DESC_TABLE_SAMPLER]) {
      struct mali_ptr p = mali_cmd_alloc(cmd, pan_size(SAMPLER), 64);
      if (!p.cpu)
         return false;
      struct mali_sampler_packed sampler;
      pan_pack(&sampler, SAMPLER, cfg) {
         cfg.clamp_integer_array_indices = false;
      }
      memcpy(p.cpu, &sampler, sizeof(sampler));
      out->tables[MALI_V7_DESC_TABLE_SAMPLER] = p.gpu;
   }

   /* The dynamic storage buffers. */
   if (dyn_ssbo_count) {
      out->dyn_ssbos = build_dyn_ssbos(cmd, s, desc);
      if (!out->dyn_ssbos)
         return false;
   }
   return true;
}

void
mali_v7_desc_sysvals(const struct mali_v7_desc_tables *t, const struct mali_shader *s,
                     uint64_t sets[MALI_MAX_SETS + 1])
{
   for (unsigned i = 0; i < MALI_MAX_SETS; i++)
      sets[i] = (s->desc.used_set_mask & BITFIELD_BIT(i)) ? t->sets[i] : 0;
   sets[MALI_V7_DESC_TABLE_DYN_SSBOS] = t->dyn_ssbos;
}

/* ---------------------------------------------------------------------- */
/* The FAU block of a dispatch                                             */

/* mali_cmd_state.h mali_compute_fau, with the Bifrost system values added:
 * the set addresses and the dynamic storage buffer table, and the
 * constant data buffer. */
uint64_t
mali_v7_compute_fau(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                    const struct mali_v7_desc_tables *t, const void *push, uint32_t push_size,
                    const uint32_t base[3], const uint32_t groups[3])
{
   if (!cs->fau.total_count)
      return 0;

   struct mali_ptr p = mali_cmd_alloc(cmd, cs->fau.total_count * 8, 16);
   if (!p.cpu)
      return 0;

   struct mali_compute_sysvals sv;
   memset(&sv, 0, sizeof(sv));
   sv.base.x = base[0];
   sv.base.y = base[1];
   sv.base.z = base[2];
   sv.num_work_groups.x = groups[0];
   sv.num_work_groups.y = groups[1];
   sv.num_work_groups.z = groups[2];
   sv.local_group_size.x = cs->cs.local_size[0];
   sv.local_group_size.y = cs->cs.local_size[1];
   sv.local_group_size.z = cs->cs.local_size[2];
   sv.common.push_uniforms = p.gpu;
   sv.common.constant_data = cs->v7.data_mem.gpu_va;
   if (t)
      mali_v7_desc_sysvals(t, cs, sv.v7_desc.sets);

   uint64_t pc[MALI_MAX_PUSH_CONST_FAUS];
   memset(pc, 0, sizeof(pc));
   memcpy(pc, push, MIN2(push_size, sizeof(pc)));

   uint64_t *out = p.cpu;
   uint64_t tmp[MALI_FAU_WORD_COUNT];
   unsigned n = 0, w;
   BITSET_FOREACH_SET(w, cs->fau.used_sysvals, MALI_MAX_SYSVAL_FAUS) {
      tmp[n++] = mali_sysval_word(&sv, sizeof(sv), w);
   }
   BITSET_FOREACH_SET(w, cs->fau.used_push_consts, MALI_MAX_PUSH_CONST_FAUS)
      tmp[n++] = pc[w];
   n += mali_shader_fau_consts(&cs->fau, &tmp[n]);
   for (unsigned i = n; i < cs->fau.total_count; i++)
      tmp[i] = 0;

   /* Constants the backend promoted into the FAU, in 32-bit halves. */
   pan_fau_foreach_imm(&cs->info.fau, i) {
      bool hi = i & 1;
      unsigned idx = i / 2;
      assert(idx < cs->fau.total_count);
      tmp[idx] = (tmp[idx] & ((uint64_t)UINT32_MAX << (32 * !hi))) |
                 ((uint64_t)cs->info.fau.words[i].constant << (32 * hi));
   }

   for (unsigned i = 0; i < cs->fau.total_count; i++)
      out[i] = tmp[i];
   return p.gpu;
}

/* ---------------------------------------------------------------------- */
/* The copy shader (panvk_vX_meta_desc_copy.c)                             */

/* Its push constants (FAU). */
struct desc_copy_info {
   uint64_t sets[MALI_MAX_SETS];
   uint64_t tables[MALI_V7_DESC_TABLE_COUNT];
   uint64_t img_attrib_table;
   struct {
      uint64_t table;                             /* the shader's copy map */
      uint32_t limits[MALI_V7_DESC_TABLE_COUNT];  /* cumulative entry counts */
      uint32_t attrib_buf_idx_offset;
   } desc_copy;
   uint32_t set_desc_counts[MALI_MAX_SETS];
};
static_assert(sizeof(struct desc_copy_info) <= MALI_MAX_PUSH_CONSTANTS_SIZE,
              "the copy shader's inputs fit the push constant block");

#define get_input_field(b, name)                                               \
   nir_load_push_constant(b, 1, sizeof(((struct desc_copy_info *)0)->name) * 8,   \
                          nir_imm_int(b, offsetof(struct desc_copy_info, name)))

/* Entry idx of a push-constant array, as a lookup over all its entries
 * (push constants are not indexable). */
static nir_def *
get_array_entry(nir_builder *b, unsigned array_offset, unsigned array_size,
                unsigned array_stride, nir_def *idx)
{
   assert(array_size > 0);
   assert(array_stride == 4 || array_stride == 8);

   nir_def *lut[MALI_MAX_SETS];
   assert(array_size <= ARRAY_SIZE(lut));
   for (unsigned i = 0; i < array_size; i++) {
      lut[i] = nir_load_push_constant(b, 1, array_stride * 8,
                                      nir_imm_int(b, i * array_stride + array_offset));
   }
   for (unsigned lut_stride = BITFIELD_BIT(util_last_bit(array_size - 1) - 1);
        lut_stride > 0; lut_stride >>= 1) {
      nir_def *bit_is_set = nir_i2b(b, nir_iand_imm(b, idx, lut_stride));
      for (unsigned i = 0; i < lut_stride && i + lut_stride < array_size; i++)
         lut[i] = nir_bcsel(b, bit_is_set, lut[i + lut_stride], lut[i]);
   }
   return lut[0];
}

#define get_input_array_slot(b, name, index)                                   \
   get_array_entry(b, offsetof(struct desc_copy_info, name),                   \
                   ARRAY_SIZE(((struct desc_copy_info *)0)->name),             \
                   sizeof(((struct desc_copy_info *)0)->name[0]), index)

static nir_def *
zero_vec(nir_builder *b, unsigned comps)
{
   nir_const_value v[8];
   for (unsigned i = 0; i < 8; i++)
      v[i] = nir_const_value_for_uint(0, 32);
   return nir_build_imm(b, comps, 32, v);
}

/* A 32-byte (or, for the UBO table, 8-byte) slot of a set to table entry
 * dst_desc_idx; zero when the slot is past the set's end (a set smaller
 * than the layout the shader was compiled against). */
static void
set_to_table_copy(nir_builder *b, nir_def *set_ptr, nir_def *set_desc_count,
                  nir_def *src_desc_idx, nir_def *table_ptr, nir_def *dst_desc_idx,
                  unsigned element_size)
{
   nir_def *dst_offset = nir_u2u64(b, nir_imul_imm(b, dst_desc_idx, element_size));

   nir_push_if(b, nir_ult(b, src_desc_idx, set_desc_count));
   {
      nir_def *src_offset = nir_u2u64(b, nir_imul_imm(b, src_desc_idx, MALI_DESCRIPTOR_SIZE));
      nir_def *desc = nir_load_global(b, element_size / 4, 32, nir_iadd(b, set_ptr, src_offset),
                                      .align_mul = element_size);
      nir_store_global(b, desc, nir_iadd(b, table_ptr, dst_offset), .align_mul = element_size);
   }
   nir_push_else(b, NULL);
   {
      nir_store_global(b, zero_vec(b, element_size / 4), nir_iadd(b, table_ptr, dst_offset),
                       .align_mul = element_size);
   }
   nir_pop_if(b, NULL);
}

/*
 * An image slot (an Attribute Buffer and its 3D continuation, the format
 * folded into the stride word: mali_image_view.c) or a texel buffer slot (a
 * 1D Attribute Buffer and an Attribute: mali_descriptor_set.c) to image
 * entry dst_desc_idx: the pair into the attribute buffer table, and an
 * Attribute descriptor naming buffer 2 * dst_desc_idx + attrib_buf_idx_offset
 * into the attribute table.
 */
static void
set_to_table_img_copy(nir_builder *b, nir_def *set_ptr, nir_def *set_desc_count,
                      nir_def *src_desc_idx, nir_def *attrib_table_ptr,
                      nir_def *attrib_buf_table_ptr, nir_def *dst_desc_idx)
{
   const unsigned element_size = pan_size(ATTRIBUTE_BUFFER) * 2;
   const unsigned attrib_buf_comps = element_size / 4;
   const unsigned attrib_comps = pan_size(ATTRIBUTE) / 4;
   nir_def *attrib_offset = nir_u2u64(b, nir_imul_imm(b, dst_desc_idx, pan_size(ATTRIBUTE)));
   nir_def *attrib_buf_offset = nir_u2u64(b, nir_imul_imm(b, dst_desc_idx, element_size));

   nir_push_if(b, nir_ult(b, src_desc_idx, set_desc_count));
   {
      nir_def *attr_buf_idx_offset = get_input_field(b, desc_copy.attrib_buf_idx_offset);
      nir_def *src_offset = nir_u2u64(b, nir_imul_imm(b, src_desc_idx, MALI_DESCRIPTOR_SIZE));
      nir_def *src_desc = nir_load_global(b, element_size / 4, 32, nir_iadd(b, set_ptr, src_offset),
                                          .align_mul = element_size);
      nir_def *attr_buf_type = nir_iand_imm(b, nir_channel(b, src_desc, 0), BITFIELD_MASK(5));
      nir_def *buf_idx = nir_iadd(b, nir_imul_imm(b, dst_desc_idx, 2), attr_buf_idx_offset);

      /* Texel buffer. */
      nir_push_if(b, nir_ieq_imm(b, attr_buf_type, MALI_ATTRIBUTE_TYPE_1D));
      {
         nir_def *fmt_offsetenable =
            nir_iand_imm(b, nir_channel(b, src_desc, 4), ~BITFIELD_MASK(9));
         nir_def *attrib_desc = nir_vec2(b, nir_ior(b, fmt_offsetenable, buf_idx),
                                         nir_channel(b, src_desc, 5));
         nir_store_global(b, attrib_desc, nir_iadd(b, attrib_table_ptr, attrib_offset),
                          .align_mul = pan_size(ATTRIBUTE),
                          .write_mask = nir_component_mask(attrib_comps));

         nir_def *null_desc = zero_vec(b, attrib_buf_comps / 2);
         nir_def *attrib_buf_desc = nir_vec8(
            b, nir_channel(b, src_desc, 0), nir_channel(b, src_desc, 1),
            nir_iand_imm(b, nir_channel(b, src_desc, 2), BITFIELD_MASK(7)),
            nir_channel(b, src_desc, 3), nir_channel(b, null_desc, 0),
            nir_channel(b, null_desc, 1), nir_channel(b, null_desc, 2),
            nir_channel(b, null_desc, 3));
         nir_store_global(b, attrib_buf_desc, nir_iadd(b, attrib_buf_table_ptr, attrib_buf_offset),
                          .align_mul = element_size,
                          .write_mask = nir_component_mask(attrib_buf_comps));
      }
      /* Image. */
      nir_push_else(b, NULL);
      {
         nir_def *fmt = nir_iand_imm(b, nir_channel(b, src_desc, 2), 0xfffffc00);
         nir_def *attrib_desc = nir_vec2(b, nir_ior(b, buf_idx, fmt), nir_imm_int(b, 0));
         nir_store_global(b, attrib_desc, nir_iadd(b, attrib_table_ptr, attrib_offset),
                          .align_mul = pan_size(ATTRIBUTE),
                          .write_mask = nir_component_mask(attrib_comps));

         nir_def *attrib_buf_desc = nir_vec8(
            b, nir_channel(b, src_desc, 0), nir_channel(b, src_desc, 1),
            nir_iand_imm(b, nir_channel(b, src_desc, 2), BITFIELD_MASK(7)),
            nir_channel(b, src_desc, 3), nir_channel(b, src_desc, 4),
            nir_channel(b, src_desc, 5), nir_channel(b, src_desc, 6),
            nir_channel(b, src_desc, 7));
         nir_store_global(b, attrib_buf_desc, nir_iadd(b, attrib_buf_table_ptr, attrib_buf_offset),
                          .align_mul = element_size,
                          .write_mask = nir_component_mask(attrib_buf_comps));
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      /* (panvk swaps the two alignments here; an Attribute entry is only
       * 8-byte aligned.) */
      nir_def *desc = zero_vec(b, MAX2(attrib_buf_comps, attrib_comps));
      nir_store_global(b, desc, nir_iadd(b, attrib_buf_table_ptr, attrib_buf_offset),
                       .align_mul = element_size,
                       .write_mask = nir_component_mask(attrib_buf_comps));
      nir_store_global(b, desc, nir_iadd(b, attrib_table_ptr, attrib_offset),
                       .align_mul = pan_size(ATTRIBUTE),
                       .write_mask = nir_component_mask(attrib_comps));
   }
   nir_pop_if(b, NULL);
}

/* One invocation: copy map entry desc_copy_idx (a MALI_COPY_DESC_HANDLE)
 * into whichever table the cumulative limits place it in. */
static void
single_desc_copy(nir_builder *b, nir_def *desc_copy_idx)
{
   nir_def *desc_copy_offset = nir_imul_imm(b, desc_copy_idx, sizeof(uint32_t));
   nir_def *desc_copy_ptr = nir_iadd(b, get_input_field(b, desc_copy.table),
                                     nir_u2u64(b, desc_copy_offset));
   nir_def *src_copy_handle = nir_load_global(b, 1, 32, desc_copy_ptr, .align_mul = 4);

   nir_def *set_idx = nir_ushr_imm(b, src_copy_handle, 28);
   nir_def *src_desc_idx = nir_iand_imm(b, src_copy_handle, BITFIELD_MASK(28));

   nir_def *set_ptr = get_input_array_slot(b, sets, set_idx);
   nir_def *set_desc_count = get_input_array_slot(b, set_desc_counts, set_idx);
   nir_def *ubo_end = get_input_field(b, desc_copy.limits[MALI_V7_DESC_TABLE_UBO]);
   nir_def *img_end = get_input_field(b, desc_copy.limits[MALI_V7_DESC_TABLE_IMG]);
   nir_def *tex_end = get_input_field(b, desc_copy.limits[MALI_V7_DESC_TABLE_TEXTURE]);
   nir_def *sampler_end = get_input_field(b, desc_copy.limits[MALI_V7_DESC_TABLE_SAMPLER]);

   nir_push_if(b, nir_ult(b, desc_copy_idx, ubo_end));
   {
      nir_def *table_ptr = get_input_field(b, tables[MALI_V7_DESC_TABLE_UBO]);
      set_to_table_copy(b, set_ptr, set_desc_count, src_desc_idx, table_ptr, desc_copy_idx,
                        pan_size(UNIFORM_BUFFER));
   }
   nir_push_else(b, NULL);
   {
      nir_push_if(b, nir_ult(b, desc_copy_idx, img_end));
      {
         nir_def *table_ptr = get_input_field(b, tables[MALI_V7_DESC_TABLE_IMG]);
         nir_def *attrib_table_ptr = get_input_field(b, img_attrib_table);
         set_to_table_img_copy(b, set_ptr, set_desc_count, src_desc_idx, attrib_table_ptr,
                               table_ptr, nir_isub(b, desc_copy_idx, ubo_end));
      }
      nir_push_else(b, NULL);
      {
         nir_push_if(b, nir_ult(b, desc_copy_idx, tex_end));
         {
            nir_def *table_ptr = get_input_field(b, tables[MALI_V7_DESC_TABLE_TEXTURE]);
            set_to_table_copy(b, set_ptr, set_desc_count, src_desc_idx, table_ptr,
                              nir_isub(b, desc_copy_idx, img_end), pan_size(TEXTURE));
         }
         nir_push_else(b, NULL);
         {
            nir_push_if(b, nir_ult(b, desc_copy_idx, sampler_end));
            {
               nir_def *table_ptr = get_input_field(b, tables[MALI_V7_DESC_TABLE_SAMPLER]);
               set_to_table_copy(b, set_ptr, set_desc_count, src_desc_idx, table_ptr,
                                 nir_isub(b, desc_copy_idx, tex_end), pan_size(SAMPLER));
            }
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

static nir_shader *
build_desc_copy_nir(struct mali_device *dev)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE), "%s",
      "mali_v7_desc_copy");
   /* One invocation per workgroup: the job dispatches as many workgroups
    * as the map has entries (mali_v7_desc_copy). */
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   single_desc_copy(&b, id);
   return b.shader;
}

/* The copy shader, compiled through the normal path on first use and kept
 * for the device's lifetime (as the copy/fill shaders of mali_cmd_copy.c). */
static const struct mali_shader *
get_desc_copy(struct mali_cmd_buffer *cmd)
{
   struct mali_device *dev = cmd->dev;
   struct mali_jm_device *jd = dev->jm;

   simple_mtx_lock(&dev->meta_lock);
   struct mali_shader *s = jd->v7_desc_copy;
   if (!s) {
      nir_shader *nir = build_desc_copy_nir(dev);
      MALI_PER_ARCH(shader_preprocess)(dev, nir);

      const struct vk_pipeline_robustness_state rs = {
         .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      };
      const struct mali_shader_compile_info info = {.nir = nir, .rs = &rs};
      blake3_hash key;
      struct mesa_blake3 h;
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, "mali-v7-desc-copy", 17);
      _mesa_blake3_final(&h, key);
      if (MALI_PER_ARCH(shader_compile)(dev, &info, key, &s) != VK_SUCCESS)
         s = NULL;
      jd->v7_desc_copy = s;
   }
   simple_mtx_unlock(&dev->meta_lock);

   if (!s)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return s;
}

/* ---------------------------------------------------------------------- */
/* Uniform buffer words in FAU                                             */

/*
 * The copy job of a fragment shader whose dynamic uniform buffer reads
 * became FAU reads (mali_shader.c lower_ubo_push, fau.ubo_push): one
 * invocation per 32-bit half of a pushed word, dst[i] = *table[i], where
 * table[i] is the address of that half in the bound buffer, offset
 * applied, or of a zero word for a half past the binding's range. The
 * fragment job reads the FAU block after the whole vtc atom, so the job
 * needs no dependency and nothing waits for it.
 */
struct ubo_copy_info {
   uint64_t table;
   uint64_t dst;
};

static nir_shader *
build_ubo_copy_nir(struct mali_device *dev)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE), "%s",
      "mali_v7_ubo_copy");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *table = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, offsetof(struct ubo_copy_info, table)));
   nir_def *dst = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, offsetof(struct ubo_copy_info, dst)));
   nir_def *src = nir_load_global(&b, 1, 64, nir_iadd(&b, table, nir_u2u64(&b, nir_imul_imm(&b, id, 8))),
                                  .align_mul = 8);
   nir_def *v = nir_load_global(&b, 1, 32, src, .align_mul = 4);
   nir_store_global(&b, v, nir_iadd(&b, dst, nir_u2u64(&b, nir_imul_imm(&b, id, 4))), .align_mul = 4);
   return b.shader;
}

static const struct mali_shader *
get_ubo_copy(struct mali_cmd_buffer *cmd)
{
   struct mali_device *dev = cmd->dev;
   struct mali_jm_device *jd = dev->jm;

   simple_mtx_lock(&dev->meta_lock);
   struct mali_shader *s = jd->v7_ubo_copy;
   if (!s) {
      nir_shader *nir = build_ubo_copy_nir(dev);
      MALI_PER_ARCH(shader_preprocess)(dev, nir);
      const struct vk_pipeline_robustness_state rs = {
         .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      };
      const struct mali_shader_compile_info info = {.nir = nir, .rs = &rs};
      blake3_hash key;
      struct mesa_blake3 h;
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, "mali-v7-ubo-copy", 16);
      _mesa_blake3_final(&h, key);
      if (MALI_PER_ARCH(shader_compile)(dev, &info, key, &s) != VK_SUCCESS)
         s = NULL;
      jd->v7_ubo_copy = s;
   }
   simple_mtx_unlock(&dev->meta_lock);

   if (!s)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return s;
}

bool
mali_v7_ubo_copy(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c, const struct mali_shader *s,
                 const struct mali_desc_state *desc, uint64_t fau_va, uint16_t dep, uint16_t *index)
{
   const uint32_t n = s->fau.ubo_push_count;
   if (index)
      *index = dep;
   if (!n || !fau_va)
      return true;
   const struct mali_shader *copy = get_ubo_copy(cmd);
   if (!copy || !copy->spd[MALI_SPD_MAIN])
      return false;

   /* 2n source addresses, then the zero word they fall back to. */
   struct mali_ptr p = mali_cmd_alloc(cmd, (2 * n + 1) * sizeof(uint64_t), 16);
   if (!p.cpu)
      return false;
   const uint64_t zero = p.gpu + 2 * n * sizeof(uint64_t);
   uint64_t table[2 * MALI_MAX_UBO_PUSH_FAUS + 1];
   uint32_t dyn_cached = UINT32_MAX;
   uint64_t addr = 0, range = 0;
   for (uint32_t i = 0; i < n; i++) {
      const struct mali_ubo_push_word *w = &s->fau.ubo_push[i];
      if (w->dyn != dyn_cached) {
         dyn_cached = w->dyn;
         if (w->dyn >= s->v7.dyn_ubo_count ||
             !dyn_buf(desc, s->desc.dyn_bufs.map[w->dyn], &addr, &range) || (addr >> 56))
            addr = range = 0;
      }
      for (unsigned h = 0; h < 2; h++) {
         const uint64_t off = w->offset + 4 * h;
         table[2 * i + h] = addr && off + 4 <= range ? addr + off : zero;
      }
   }
   table[2 * n] = 0;
   memcpy(p.cpu, table, (2 * n + 1) * sizeof(uint64_t));

   const struct ubo_copy_info info = {
      .table = p.gpu,
      .dst = fau_va + (uint64_t)s->fau.ubo_push_start * MALI_FAU_WORD_SIZE,
   };
   const uint32_t base[3] = {0, 0, 0};
   const uint32_t groups[3] = {2 * n, 1, 1};
   const uint32_t local[3] = {1, 1, 1};
   const struct mali_thread_props tp = mali_thread_props(cmd->dev);
   const uint64_t fau = mali_v7_compute_fau(cmd, copy, NULL, &info, sizeof(info), base, groups);
   const uint64_t tsd = mali_compute_tsd(cmd, copy, &tp, groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !tsd || (copy->fau.total_count && !fau))
      return false;

   struct mali_ptr job =
      mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_COMPUTE, pan_size(COMPUTE_JOB), false, dep);
   if (!job.cpu)
      return false;
   if (index)
      *index = c->index;
   mali_v7_compute_invocation(pan_section_ptr(job.cpu, COMPUTE_JOB, INVOCATION), local, groups);
   pan_section_pack(job.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = mali_v7_compute_task_split(local);
   }
   pan_section_pack(job.cpu, COMPUTE_JOB, DRAW, cfg) {
      cfg.state = copy->spd[MALI_SPD_MAIN];
      cfg.push_uniforms = fau;
      cfg.thread_storage = tsd;
   }
   return true;
}

void
mali_v7_desc_finish(struct mali_device *dev)
{
   struct mali_jm_device *jd = dev->jm;
   MALI_PER_ARCH(shader_unref)(dev, jd->v7_desc_copy);
   jd->v7_desc_copy = NULL;
   MALI_PER_ARCH(shader_unref)(dev, jd->v7_ubo_copy);
   jd->v7_ubo_copy = NULL;
}

/* ---------------------------------------------------------------------- */
/* Filling the copied entries                                              */

/*
 * MALISX2_V7_CPU_DESC_COPY=1: the CPU copies the entries when the command
 * is recorded, from the sets' CPU mappings, instead of the copy job (file
 * comment).
 */
static bool
cpu_copy_mode(void)
{
   static int mode = -1;
   if (mode < 0) {
      const char *e = getenv("MALISX2_V7_CPU_DESC_COPY");
      mode = e && *e && strcmp(e, "0") != 0;
      if (mode)
         mesa_logi("malisx2: MALISX2_V7_CPU_DESC_COPY is set: Bifrost descriptor tables are "
                   "filled on the CPU at record time, without the copy job");
   }
   return mode;
}

/* The CPU form of the copy shader's set_to_table_img_copy: one image or
 * texel buffer slot (zero when unbound) into the Attribute Buffer table
 * (the pair) and the Attribute table (an Attribute naming buffer
 * 2 * idx + offset). */
static void
cpu_img_copy(const uint8_t *slot, uint32_t idx, uint32_t attrib_buf_idx_offset,
             uint8_t *attrib_bufs, uint8_t *attribs)
{
   uint32_t src[8], buf[8] = {0}, attr[2] = {0};
   if (slot) {
      memcpy(src, slot, sizeof(src));
      const uint32_t buf_idx = idx * 2 + attrib_buf_idx_offset;
      /* Both records of the pair within the 9-bit index (mali_shader.h). */
      assert(buf_idx + 1 < MALI_V7_ATTRIB_BUF_RECORDS);
      if ((src[0] & BITFIELD_MASK(5)) == MALI_ATTRIBUTE_TYPE_1D) {
         /* Texel buffer: its own Attribute (format, offset enable, offset),
          * the Attribute Buffer alone. */
         attr[0] = (src[4] & ~BITFIELD_MASK(9)) | buf_idx;
         attr[1] = src[5];
         buf[0] = src[0];
         buf[1] = src[1];
         buf[2] = src[2] & BITFIELD_MASK(7);
         buf[3] = src[3];
      } else {
         /* Image: the format from the stride word, the pair as it is. */
         attr[0] = (src[2] & 0xfffffc00u) | buf_idx;
         memcpy(buf, src, sizeof(buf));
         buf[2] = src[2] & BITFIELD_MASK(7);
      }
   }
   memcpy(attrib_bufs + (size_t)idx * 2 * pan_size(ATTRIBUTE_BUFFER), buf, sizeof(buf));
   memcpy(attribs + (size_t)idx * pan_size(ATTRIBUTE), attr, sizeof(attr));
}

/* The debugging fill: table by table in map order, from the sets' CPU
 * mappings (an unbound or short set's entry zero, as the copy shader). */
static void
cpu_fill(const struct mali_shader *s, const struct mali_desc_state *desc,
         const struct mali_v7_desc_tables *t, uint32_t attrib_buf_idx_offset)
{
   const uint32_t *map = s->v7.map;
   for (unsigned tab = 0; tab < MALI_V7_DESC_TABLE_COUNT; tab++) {
      const unsigned entry = table_entry_size(tab);
      for (uint32_t i = 0; i < s->v7.count[tab]; i++) {
         const uint8_t *slot = set_slot(desc, *map++);
         if (tab == MALI_V7_DESC_TABLE_IMG) {
            cpu_img_copy(slot, i, attrib_buf_idx_offset, t->tables_cpu[tab], t->img_attribs_cpu);
         } else {
            uint8_t *dst = (uint8_t *)t->tables_cpu[tab] + (size_t)i * entry;
            if (slot)
               memcpy(dst, slot, entry);
            else
               memset(dst, 0, entry);
         }
      }
   }
}

bool
mali_v7_desc_copy(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                  const struct mali_shader *s, const struct mali_desc_state *desc,
                  const struct mali_v7_desc_tables *t, uint32_t attrib_buf_idx_offset,
                  uint16_t *index)
{
   *index = 0;
   assert(t->shader == s);
   if (!s->v7.map_count)
      return true;
   /* Every table the map fills exists (a vertex shader's image tables are
    * the draw's, mali_jm_cmd_draw.c). */
   for (unsigned tab = 0; tab < MALI_V7_DESC_TABLE_COUNT; tab++)
      assert(!s->v7.count[tab] || (t->tables[tab] && t->tables_cpu[tab]));
   assert(!s->v7.count[MALI_V7_DESC_TABLE_IMG] || (t->img_attribs && t->img_attribs_cpu));
   /* The last image pair's records within the 9-bit Attribute buffer
    * index (mali_shader.h; v7_fixup_info refuses a bigger table). */
   assert(attrib_buf_idx_offset + 2 * s->v7.count[MALI_V7_DESC_TABLE_IMG] <=
          MALI_V7_ATTRIB_BUF_RECORDS);

   if (cpu_copy_mode()) {
      cpu_fill(s, desc, t, attrib_buf_idx_offset);
      return true;
   }

   const struct mali_shader *copy = get_desc_copy(cmd);
   if (!copy || !copy->spd[MALI_SPD_MAIN])
      return false;
   if (!s->v7.map_mem.gpu_va) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }

   struct desc_copy_info info;
   memset(&info, 0, sizeof(info));
   for (unsigned i = 0; i < MALI_MAX_SETS; i++) {
      info.sets[i] = t->sets[i];
      info.set_desc_counts[i] = t->set_desc_counts[i];
   }
   for (unsigned i = 0; i < MALI_V7_DESC_TABLE_COUNT; i++) {
      info.tables[i] = s->v7.count[i] ? t->tables[i] : 0;
      info.desc_copy.limits[i] = s->v7.count[i] + (i ? info.desc_copy.limits[i - 1] : 0);
   }
   info.img_attrib_table = t->img_attribs;
   info.desc_copy.table = s->v7.map_mem.gpu_va;
   info.desc_copy.attrib_buf_idx_offset = attrib_buf_idx_offset;
   const uint32_t copy_count = info.desc_copy.limits[MALI_V7_DESC_TABLE_COUNT - 1];
   assert(copy_count == s->v7.map_count && copy_count > 0);

   /* copy_count workgroups of one invocation (panvk runs one workgroup of
    * copy_count; this way the workgroup size the shader was compiled with
    * is the one it runs with). */
   const uint32_t base[3] = {0, 0, 0};
   const uint32_t groups[3] = {copy_count, 1, 1};
   const uint32_t local[3] = {1, 1, 1};
   const struct mali_thread_props tp = mali_thread_props(cmd->dev);
   const uint64_t fau = mali_v7_compute_fau(cmd, copy, NULL, &info, sizeof(info), base, groups);
   const uint64_t tsd = mali_compute_tsd(cmd, copy, &tp, groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !tsd || (copy->fau.total_count && !fau))
      return false;

   struct mali_ptr job =
      mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_COMPUTE, pan_size(COMPUTE_JOB), false, 0);
   if (!job.cpu)
      return false;

   mali_v7_compute_invocation(pan_section_ptr(job.cpu, COMPUTE_JOB, INVOCATION), local, groups);
   pan_section_pack(job.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = mali_v7_compute_task_split(local);
   }
   pan_section_pack(job.cpu, COMPUTE_JOB, DRAW, cfg) {
      cfg.state = copy->spd[MALI_SPD_MAIN];
      cfg.push_uniforms = fau;
      cfg.thread_storage = tsd;
   }
   *index = c->index;
   return true;
}
