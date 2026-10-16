/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Descriptor accesses in NIR to the hardware's resource model, following
 * panvk's lowering (Mesa src/panfrost/vulkan/panvk_vX_nir_lower_descriptors.c,
 * MIT), because the descriptor layout is panvk's (mali_descriptor_set_layout.h).
 * Built per arch.
 *
 * Valhall (v9, v11), resource tables:
 *
 *  - set N is resource table N + 1; a binding element's slot comes from the
 *    set layout (mali_desc_index);
 *  - table 0 is the driver's: vertex attributes (vertex shaders, 16
 *    entries) or varying descriptors (fragment shaders, one per input
 *    slot), then a dummy sampler, then copies of the dynamic buffers the
 *    shader uses. The caller sets desc_info->driver_table_prefix to the
 *    first part's size before calling. panvk puts a fragment shader's dummy
 *    sampler at 0 when it compiles and after the varying descriptors when
 *    it draws; we reserve the varying slots at compile time instead, so the
 *    two always agree;
 *  - uniform and storage buffers use the vec2-index + 32-bit offset address
 *    format through the resource handle; images and textures get their
 *    resource handle as texture/sampler index or offset source;
 *  - descriptor reads (buffer sizes) load the 32-byte descriptor through
 *    the special resource table 62, which addresses resource tables as
 *    buffers.
 *
 * Bifrost (v7), per-stage descriptor tables (mali_shader.h, enum
 * mali_v7_desc_table; panvk's PAN_ARCH < 9 branches):
 *
 *  - every regular uniform buffer, texture, sampler, storage image and
 *    texel buffer the shader can reach is given an index in its table, in
 *    the order the accesses are recorded, and the copy map remembers which
 *    set slot fills it; the shader's uniform-buffer index, texture and
 *    sampler index, and image (attribute) index are those table indices
 *    (images in a vertex shader come after its 16 vertex attributes);
 *  - dynamic uniform buffers follow the regular ones in the UBO table;
 *    dynamic storage buffers have a table of address/size pairs of their
 *    own, both filled by the dispatch or draw with the dynamic offsets;
 *  - storage buffers are read from the set itself: the set's base address
 *    is a system value (desc.sets), the slot holds the 64-bit address and
 *    32-bit size (struct mali_v7_ssbo_addr), and the addressing is 64-bit
 *    global (bounded when robustBufferAccess is on);
 *  - size queries read the descriptor from the set the same way;
 *  - constant data lives in a buffer whose address is the common.constant_data
 *    system value (no PC-relative constant pool on Bifrost).
 *
 * YCbCr conversion and immutable-sampler folding of the original are left
 * out (no YCbCr).
 */

#ifndef PAN_ARCH
#error "mali_nir_lower_descriptors.c is built per arch: PAN_ARCH must be set"
#endif

#include "mali_shader.h"

#include "nir_builder.h"
#include "panfrost/compiler/pan_nir.h"
#include "util/hash_table.h"

#include "vk_pipeline.h"

#include "mali_arch.h"

#define RESOURCE_TABLE_IDX 62

struct desc_map {
   /* Entry i: MALI_COPY_DESC_HANDLE(set, slot or dynamic buffer index). */
   uint32_t *map;
   uint32_t count;
};

struct lower_desc_ctx {
   mesa_shader_stage stage;
   const struct mali_descriptor_set_layout *set_layouts[MALI_MAX_SETS];
   struct hash_table_u64 *ht;
   bool add_bounds_checks;
   uint32_t used_set_mask;
#if PAN_ARCH >= 9
   uint32_t dummy_sampler_handle;
   uint32_t dyn_bufs_start;
   struct desc_map dyn_bufs;
   uint32_t driver_table_prefix;
#else
   struct desc_map dyn_ubos;
   struct desc_map dyn_ssbos;
   struct desc_map tables[MALI_V7_TABLE_COUNT];
   nir_address_format ssbo_addr_format;
#endif
};

struct desc_id {
   union {
      struct {
         uint32_t binding;
         uint32_t set : 4;
         uint32_t sampler_subdesc : 1;
         uint32_t pad : 27;
      };
      uint64_t ht_key;
   };
};

static const struct mali_descriptor_set_binding_layout *
binding_layout(const struct lower_desc_ctx *ctx, uint32_t set, uint32_t binding)
{
   assert(set < MALI_MAX_SETS && ctx->set_layouts[set]);
   assert(binding < ctx->set_layouts[set]->binding_count);
   return &ctx->set_layouts[set]->bindings[binding];
}

#if PAN_ARCH < 9

/* Which Bifrost table a binding's descriptors are copied into (panvk
 * desc_type_to_table_type); -1 for the types that have none (storage
 * buffers are read from the set, dynamic buffers have their own maps). */
static int
v7_table_of(const struct mali_descriptor_set_binding_layout *bl, bool sampler_subdesc)
{
   switch (bl->type) {
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return sampler_subdesc ? MALI_V7_TABLE_SAMPLER : MALI_V7_TABLE_TEXTURE;
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      return MALI_V7_TABLE_TEXTURE;
   case VK_DESCRIPTOR_TYPE_SAMPLER:
      return MALI_V7_TABLE_SAMPLER;
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      return MALI_V7_TABLE_IMG;
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
      return MALI_V7_TABLE_UBO;
   default:
      return -1;
   }
}

/* The desc.sets entry holding a stage's dynamic storage buffer table. */
static unsigned
v7_dyn_ssbo_table(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX: return MALI_V7_DESC_SETS_VS_DYN_SSBOS;
   case MESA_SHADER_FRAGMENT: return MALI_V7_DESC_SETS_FS_DYN_SSBOS;
   default: return MALI_V7_DESC_SETS_CS_DYN_SSBOS;
   }
}

/* A 64-bit desc.sets[idx] system value (the load is through the push
 * constant mechanism of mali_shader.c, SYSVALS_PUSH_CONST_BASE), with a
 * dynamic or constant index. */
#define V7_SYSVALS_PUSH_CONST_BASE MALI_MAX_PUSH_CONSTANTS_SIZE

static nir_def *
v7_load_set_base(nir_builder *b, nir_def *idx)
{
   const bool compute = b->shader->info.stage == MESA_SHADER_COMPUTE;
   const unsigned off = compute ? offsetof(struct mali_compute_sysvals, desc.sets)
                                : offsetof(struct mali_graphics_sysvals, desc.sets);
   const unsigned size = compute ? sizeof(((struct mali_compute_sysvals *)0)->desc.sets)
                                 : sizeof(((struct mali_graphics_sysvals *)0)->desc.sets);
   return nir_load_push_constant(b, 1, 64, nir_imul_imm(b, idx, 8),
                                 .base = V7_SYSVALS_PUSH_CONST_BASE + off, .range = size);
}

#endif /* PAN_ARCH < 9 */

/* Resource handle (v9+) or table index (v7) of (set, binding, sub-descriptor),
 * element 0. */
static uint32_t
shader_desc_idx(const struct lower_desc_ctx *ctx, uint32_t set, uint32_t binding,
                enum mali_subdesc sub)
{
   const struct mali_descriptor_set_binding_layout *bl =
      binding_layout(ctx, set, binding);

#if PAN_ARCH >= 9
   /* Non-dynamic descriptors are read straight from their set's table. */
   if (!vk_descriptor_type_is_dynamic(bl->type))
      return pan_res_handle(set + 1, mali_desc_index(bl, 0, sub));
#else
   /* Storage buffers are read from the set: their slot index. */
   if (bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      return bl->desc_idx;
#endif

   struct desc_id id = {
      .set = set,
      .sampler_subdesc = sub == MALI_SUBDESC_SAMPLER,
      .binding = binding,
   };
   uint32_t *entry = _mesa_hash_table_u64_search(ctx->ht, id.ht_key);

#if PAN_ARCH >= 9
   /* Dynamic buffers: their copies in the driver table. */
   assert(entry && entry >= ctx->dyn_bufs.map &&
          entry < ctx->dyn_bufs.map + ctx->dyn_bufs.count);
   return pan_res_handle(0, ctx->dyn_bufs_start + (entry - ctx->dyn_bufs.map));
#else
   const struct desc_map *map;
   int table = -1;
   if (bl->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) {
      map = &ctx->dyn_ubos;
   } else if (bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
      map = &ctx->dyn_ssbos;
   } else {
      table = v7_table_of(bl, id.sampler_subdesc);
      assert(table >= 0);
      map = &ctx->tables[table];
   }
   assert(entry && entry >= map->map && entry < map->map + map->count);
   uint32_t idx = entry - map->map;
   if (bl->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) {
      /* Dynamic uniform buffers follow the regular ones in the UBO table. */
      idx += ctx->tables[MALI_V7_TABLE_UBO].count;
   } else if (table == MALI_V7_TABLE_IMG && ctx->stage == MESA_SHADER_VERTEX) {
      /* Images follow the vertex attributes in the attribute table. */
      idx += MALI_MAX_VS_ATTRIBS;
   }
   return idx;
#endif
}

/* Vulkan resource index. v9+: vec3(first handle, array index, array size - 1).
 * v7 uniform buffers: vec2((array size - 1) << 16 | first index, array index);
 * v7 storage buffers: vec4(desc.sets entry, first slot, array index,
 * array size - 1). */
static nir_def *
build_res_index(nir_builder *b, const struct lower_desc_ctx *ctx, uint32_t set,
                uint32_t binding, nir_def *array_index)
{
   const struct mali_descriptor_set_binding_layout *bl =
      binding_layout(ctx, set, binding);
   uint32_t desc_idx = shader_desc_idx(ctx, set, binding, MALI_SUBDESC_NONE);

#if PAN_ARCH >= 9
   return nir_vec3(b, nir_imm_int(b, desc_idx), array_index,
                   nir_imm_int(b, bl->desc_count - 1));
#else
   switch (bl->type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
      return nir_vec2(b, nir_imm_int(b, ((bl->desc_count - 1) << 16) | desc_idx),
                      array_index);
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      return nir_vec4(b, nir_imm_int(b, set), nir_imm_int(b, desc_idx), array_index,
                      nir_imm_int(b, bl->desc_count - 1));
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      return nir_vec4(b, nir_imm_int(b, v7_dyn_ssbo_table(ctx->stage)),
                      nir_imm_int(b, desc_idx), array_index,
                      nir_imm_int(b, bl->desc_count - 1));
   default:
      UNREACHABLE("not a buffer descriptor");
   }
#endif
}

static nir_def *
build_res_reindex(nir_builder *b, nir_def *orig, nir_def *delta)
{
#if PAN_ARCH >= 9
   return nir_vec3(b, nir_channel(b, orig, 0),
                   nir_iadd(b, nir_channel(b, orig, 1), delta),
                   nir_channel(b, orig, 2));
#else
   if (orig->num_components == 2) {
      return nir_vec2(b, nir_channel(b, orig, 0),
                      nir_iadd(b, nir_channel(b, orig, 1), delta));
   }
   return nir_vec4(b, nir_channel(b, orig, 0), nir_channel(b, orig, 1),
                   nir_iadd(b, nir_channel(b, orig, 2), delta), nir_channel(b, orig, 3));
#endif
}

static nir_def *
build_buffer_addr_for_res_index(nir_builder *b, nir_def *res_index,
                                const struct lower_desc_ctx *ctx)
{
#if PAN_ARCH >= 9
   nir_def *first_desc_index = nir_channel(b, res_index, 0);
   nir_def *array_index = nir_channel(b, res_index, 1);
   nir_def *array_max = nir_channel(b, res_index, 2);

   if (ctx->add_bounds_checks)
      array_index = nir_umin(b, array_index, array_max);

   return nir_vec3(b, first_desc_index, array_index, nir_imm_int(b, 0));
#else
   if (res_index->num_components == 2) {
      /* Uniform buffer: (table index, 0) for the 32-bit index + offset
       * address format. */
      nir_def *packed = nir_channel(b, res_index, 0);
      nir_def *array_index = nir_channel(b, res_index, 1);
      nir_def *first = nir_extract_u16(b, packed, nir_imm_int(b, 0));
      nir_def *array_max = nir_extract_u16(b, packed, nir_imm_int(b, 1));
      if (ctx->add_bounds_checks)
         array_index = nir_umin(b, array_index, array_max);
      return nir_vec2(b, nir_iadd(b, first, array_index), nir_imm_int(b, 0));
   }

   /* Storage buffer: the address/size pair from the set (or the dynamic
    * table), as (address lo, address hi, size, offset 0). */
   nir_def *table = nir_channel(b, res_index, 0);
   nir_def *first = nir_channel(b, res_index, 1);
   nir_def *array_index = nir_channel(b, res_index, 2);
   nir_def *array_max = nir_channel(b, res_index, 3);
   if (ctx->add_bounds_checks)
      array_index = nir_umin(b, array_index, array_max);
   nir_def *off = nir_imul_imm(b, nir_iadd(b, array_index, first), MALI_DESCRIPTOR_SIZE);
   nir_def *base = v7_load_set_base(b, table);
   nir_def *desc = nir_load_global(b, 4, 32, nir_iadd(b, base, nir_u2u64(b, off)),
                                   .align_mul = MALI_DESCRIPTOR_SIZE);
   return nir_vec4(b, nir_channel(b, desc, 0), nir_channel(b, desc, 1),
                   nir_channel(b, desc, 2), nir_imm_int(b, 0));
#endif
}

static bool
lower_res_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin,
                    const struct lower_desc_ctx *ctx)
{
   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *res;
   switch (intrin->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      res = build_res_index(b, ctx, nir_intrinsic_desc_set(intrin),
                            nir_intrinsic_binding(intrin), intrin->src[0].ssa);
      break;
   case nir_intrinsic_vulkan_resource_reindex:
      res = build_res_reindex(b, intrin->src[0].ssa, intrin->src[1].ssa);
      break;
   case nir_intrinsic_load_vulkan_descriptor:
      res = build_buffer_addr_for_res_index(b, intrin->src[0].ssa, ctx);
      break;
   default:
      UNREACHABLE("unhandled resource intrinsic");
   }

   assert(intrin->def.bit_size == res->bit_size);
   assert(intrin->def.num_components == res->num_components);
   nir_def_replace(&intrin->def, res);
   return true;
}

static void
get_resource_deref_binding(nir_deref_instr *deref, uint32_t *set,
                           uint32_t *binding, uint32_t *index_imm,
                           nir_def **index_ssa, uint32_t *max_idx)
{
   *index_imm = 0;
   *max_idx = 0;
   *index_ssa = NULL;

   if (deref->deref_type == nir_deref_type_array) {
      if (nir_src_is_const(deref->arr.index)) {
         *index_imm = nir_src_as_uint(deref->arr.index);
         *max_idx = *index_imm;
      } else {
         *index_ssa = deref->arr.index.ssa;
         /* Zero means a runtime-sized array; minus one gives UINT32_MAX. */
         *max_idx = ((uint32_t)glsl_array_size(nir_deref_instr_parent(deref)->type)) - 1;
      }
      deref = nir_deref_instr_parent(deref);
   }

   assert(deref->deref_type == nir_deref_type_var);
   *set = deref->var->data.descriptor_set;
   *binding = deref->var->data.binding;
}

#if PAN_ARCH < 9

/* ---------------------------------------------------------------------- */
/* Bifrost: descriptor words read from the set (size queries)              */

/* num_components x bit_size bits at byte desc_offset of the descriptor
 * `deref` names, read from the set's memory (panvk load_resource_deref_desc
 * at PAN_ARCH < 9). sub picks the half of a combined image/sampler. */
static nir_def *
v7_load_desc_words(nir_builder *b, nir_deref_instr *deref, enum mali_subdesc sub,
                   unsigned desc_offset, unsigned num_components, unsigned bit_size,
                   const struct lower_desc_ctx *ctx)
{
   uint32_t set, binding, index_imm, max_idx;
   nir_def *index_ssa;
   get_resource_deref_binding(deref, &set, &binding, &index_imm, &index_ssa, &max_idx);

   const struct mali_descriptor_set_binding_layout *bl = binding_layout(ctx, set, binding);
   const uint32_t stride = mali_desc_stride(bl);
   const uint32_t first = bl->desc_idx + mali_subdesc_offset(bl, sub);

   assert(index_ssa == NULL || index_imm == 0);
   if (!index_ssa)
      index_ssa = nir_imm_int(b, index_imm);

   nir_def *slot = nir_iadd_imm(b, nir_imul_imm(b, index_ssa, stride), first);
   nir_def *off = nir_iadd_imm(b, nir_imul_imm(b, slot, MALI_DESCRIPTOR_SIZE), desc_offset);
   nir_def *base = v7_load_set_base(b, nir_imm_int(b, set));
   const unsigned align = 1u << (ffs(MALI_DESCRIPTOR_SIZE + desc_offset) - 1);
   return nir_load_global(b, num_components, bit_size, nir_iadd(b, base, nir_u2u64(b, off)),
                          .align_mul = align);
}

/* Storage image or texel buffer size (panvk load_img_size). The image slot
 * holds an Attribute Buffer (stride word at 8, the texel size in its low 7
 * bits and log2(samples) in bits 7..9) and its 3D continuation (S, T, R
 * minus one, 16 bits each from byte 18); a texel buffer slot holds a 1D
 * Attribute Buffer (stride at 8, size at 12). */
static nir_def *
v7_load_img_size(nir_builder *b, nir_deref_instr *deref, enum glsl_sampler_dim dim,
                 bool is_array, const struct lower_desc_ctx *ctx)
{
   if (dim == GLSL_SAMPLER_DIM_BUF) {
      nir_def *stride_size = v7_load_desc_words(b, deref, MALI_SUBDESC_NONE, 8, 2, 32, ctx);
      return nir_udiv(b, nir_channel(b, stride_size, 1), nir_channel(b, stride_size, 0));
   }

   nir_def *sz = v7_load_desc_words(b, deref, MALI_SUBDESC_NONE, 18, 3, 16, ctx);
   if (is_array && dim == GLSL_SAMPLER_DIM_CUBE)
      sz = nir_vector_insert_imm(b, sz, nir_udiv_imm(b, nir_channel(b, sz, 2), 6), 2);
   if (is_array && dim == GLSL_SAMPLER_DIM_1D)
      sz = nir_vec2(b, nir_channel(b, sz, 0), nir_channel(b, sz, 2));
   sz = nir_pad_vector_imm_int(b, sz, 0, 4);
   /* Stored minus one. */
   sz = nir_iadd_imm(b, nir_u2u32(b, sz), 1);
   if (is_array && dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = v7_load_desc_words(b, deref, MALI_SUBDESC_NONE, 8, 1, 32, ctx);
      samples = nir_iand_imm(b, nir_ushr_imm(b, samples, 7), BITFIELD_MASK(3));
      sz = nir_vector_insert_imm(b, sz, nir_ushr(b, nir_channel(b, sz, 2), samples), 2);
   }
   return sz;
}

static nir_def *
v7_load_img_samples(nir_builder *b, nir_deref_instr *deref, const struct lower_desc_ctx *ctx)
{
   nir_def *samples = v7_load_desc_words(b, deref, MALI_SUBDESC_NONE, 8, 1, 32, ctx);
   samples = nir_iand_imm(b, nir_ushr_imm(b, samples, 7), BITFIELD_MASK(3));
   return nir_ishl(b, nir_imm_int(b, 1), samples);
}

/*
 * Texture queries (panvk load_tex_size / load_tex_levels / load_tex_samples):
 * the word offsets of the Bifrost Texture descriptor (width and height
 * minus one as 16 bits at 4, depth or layer count at 24 or 28, the level
 * count in word 2 bits 16..21, log2 samples in word 3 bits 13..15). The
 * descriptors themselves are the texture family's (mali_image_view.c);
 * only the offsets are relied on here.
 */
static nir_def *
v7_load_tex_size(nir_builder *b, nir_deref_instr *deref, enum glsl_sampler_dim dim,
                 bool is_array, const struct lower_desc_ctx *ctx)
{
   if (dim == GLSL_SAMPLER_DIM_BUF) {
      nir_def *stride_size = v7_load_desc_words(b, deref, MALI_SUBDESC_NONE, 8, 2, 32, ctx);
      return nir_udiv(b, nir_channel(b, stride_size, 1), nir_channel(b, stride_size, 0));
   }
   nir_def *wh = v7_load_desc_words(b, deref, MALI_SUBDESC_TEXTURE, 4, 2, 16, ctx);
   nir_def *depth_or_layers = v7_load_desc_words(b, deref, MALI_SUBDESC_TEXTURE,
                                                 dim == GLSL_SAMPLER_DIM_3D ? 28 : 24, 1, 16,
                                                 ctx);
   nir_def *sz = is_array && dim == GLSL_SAMPLER_DIM_1D
                    ? nir_vec2(b, nir_channel(b, wh, 0), depth_or_layers)
                    : nir_vec3(b, nir_channel(b, wh, 0), nir_channel(b, wh, 1), depth_or_layers);
   sz = nir_pad_vector_imm_int(b, sz, 0, 4);
   return nir_iadd_imm(b, nir_u2u32(b, sz), 1);
}

static nir_def *
v7_load_tex_levels(nir_builder *b, nir_deref_instr *deref, const struct lower_desc_ctx *ctx)
{
   nir_def *w2 = v7_load_desc_words(b, deref, MALI_SUBDESC_TEXTURE, 8, 1, 32, ctx);
   return nir_iadd_imm(b, nir_iand_imm(b, nir_ushr_imm(b, w2, 16), 0x1f), 1);
}

static nir_def *
v7_load_tex_samples(nir_builder *b, nir_deref_instr *deref, const struct lower_desc_ctx *ctx)
{
   nir_def *w3 = v7_load_desc_words(b, deref, MALI_SUBDESC_TEXTURE, 12, 1, 32, ctx);
   return nir_ishl(b, nir_imm_int(b, 1), nir_iand_imm(b, nir_ushr_imm(b, w3, 13), 0x7));
}

/* The index stride of an array element in a Bifrost table: textures and
 * samplers of a combined image/sampler are copied to separate tables, so
 * each advances by one per element (panvk get_desc_array_stride). */
static uint32_t
v7_array_stride(const struct mali_descriptor_set_binding_layout *bl)
{
   (void)bl;
   return 1;
}

#endif /* PAN_ARCH < 9 */

static bool
lower_tex(nir_builder *b, nir_tex_instr *tex, const struct lower_desc_ctx *ctx)
{
   bool progress = false;

   b->cursor = nir_before_instr(&tex->instr);

#if PAN_ARCH < 9
   if (tex->op == nir_texop_txs || tex->op == nir_texop_query_levels ||
       tex->op == nir_texop_texture_samples) {
      int tex_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      assert(tex_src_idx >= 0);
      nir_deref_instr *deref = nir_src_as_deref(tex->src[tex_src_idx].src);
      nir_def *res;
      switch (tex->op) {
      case nir_texop_txs:
         res = nir_channels(b, v7_load_tex_size(b, deref, tex->sampler_dim, tex->is_array, ctx),
                            nir_component_mask(tex->def.num_components));
         break;
      case nir_texop_query_levels:
         res = v7_load_tex_levels(b, deref, ctx);
         break;
      default:
         res = v7_load_tex_samples(b, deref, ctx);
         break;
      }
      nir_def_replace(&tex->def, res);
      return true;
   }
#endif

   int sampler_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   if (sampler_src_idx >= 0) {
      nir_def *plane = nir_steal_tex_src(tex, nir_tex_src_plane);
      assert(!plane || nir_src_as_uint(nir_src_for_ssa(plane)) == 0);
      (void)plane;

      nir_deref_instr *deref = nir_src_as_deref(tex->src[sampler_src_idx].src);
      nir_tex_instr_remove_src(tex, sampler_src_idx);

      uint32_t set, binding, index_imm, max_idx;
      nir_def *index_ssa;
      get_resource_deref_binding(deref, &set, &binding, &index_imm, &index_ssa,
                                 &max_idx);

      const struct mali_descriptor_set_binding_layout *bl =
         binding_layout(ctx, set, binding);
#if PAN_ARCH >= 9
      uint32_t stride = mali_desc_stride(bl);
#else
      uint32_t stride = v7_array_stride(bl);
#endif
      uint32_t sampler_index =
         shader_desc_idx(ctx, set, binding, MALI_SUBDESC_SAMPLER) + index_imm * stride;

      if (index_ssa) {
         nir_def *offset = nir_iadd_imm(b, nir_imul_imm(b, index_ssa, stride),
                                        sampler_index);
         nir_tex_instr_add_src(tex, nir_tex_src_sampler_offset, offset);
      } else {
         tex->sampler_index = sampler_index;
      }
      progress = true;
   } else {
#if PAN_ARCH >= 9
      /* Texel fetches still name a sampler on Valhall. */
      tex->sampler_index = ctx->dummy_sampler_handle;
#endif
   }

   int tex_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (tex_src_idx >= 0) {
      nir_deref_instr *deref = nir_src_as_deref(tex->src[tex_src_idx].src);
      nir_tex_instr_remove_src(tex, tex_src_idx);

      uint32_t set, binding, index_imm, max_idx;
      nir_def *index_ssa;
      get_resource_deref_binding(deref, &set, &binding, &index_imm, &index_ssa,
                                 &max_idx);

      const struct mali_descriptor_set_binding_layout *bl =
         binding_layout(ctx, set, binding);
#if PAN_ARCH >= 9
      uint32_t stride = mali_desc_stride(bl);
#else
      uint32_t stride = v7_array_stride(bl);
#endif
      uint32_t texture_index =
         shader_desc_idx(ctx, set, binding, MALI_SUBDESC_TEXTURE) + index_imm * stride;

      if (index_ssa) {
         nir_def *offset = nir_iadd_imm(b, nir_imul_imm(b, index_ssa, stride),
                                        texture_index);
         nir_tex_instr_add_src(tex, nir_tex_src_texture_offset, offset);
      } else {
         tex->texture_index = texture_index;
      }
      progress = true;
   }

   return progress;
}

static nir_def *
get_img_index(nir_builder *b, nir_deref_instr *deref,
              const struct lower_desc_ctx *ctx)
{
   uint32_t set, binding, index_imm, max_idx;
   nir_def *index_ssa;
   get_resource_deref_binding(deref, &set, &binding, &index_imm, &index_ssa,
                              &max_idx);

   ASSERTED const struct mali_descriptor_set_binding_layout *bl =
      binding_layout(ctx, set, binding);
   assert(bl->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
          bl->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
          bl->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER ||
          bl->type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);

   unsigned img_offset = shader_desc_idx(ctx, set, binding, MALI_SUBDESC_NONE);

   if (!index_ssa)
      return nir_imm_int(b, img_offset + index_imm);
   assert(index_imm == 0);
   return nir_iadd_imm(b, index_ssa, img_offset);
}

static bool
lower_img_intrinsic(nir_builder *b, nir_intrinsic_instr *intr,
                    const struct lower_desc_ctx *ctx)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
#if PAN_ARCH < 9
   if (intr->intrinsic == nir_intrinsic_image_deref_size ||
       intr->intrinsic == nir_intrinsic_image_deref_samples) {
      nir_def *res;
      if (intr->intrinsic == nir_intrinsic_image_deref_size)
         res = nir_channels(b,
                            v7_load_img_size(b, deref, nir_intrinsic_image_dim(intr),
                                             nir_intrinsic_image_array(intr), ctx),
                            nir_component_mask(intr->def.num_components));
      else
         res = v7_load_img_samples(b, deref, ctx);
      nir_def_replace(&intr->def, res);
      return true;
   }
#endif
   nir_rewrite_image_intrinsic(intr, get_img_index(b, deref, ctx),
                               nir_image_intrinsic_type_default);
   return true;
}

/* Constant data: v9+ in the binary's inline constant pool, addressed from
 * the program counter; v7 in a buffer the driver uploads, whose address is
 * the common.constant_data system value. */
static bool
lower_load_constant(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_constant)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *offset = nir_iadd_imm(b, intr->src[0].ssa, nir_intrinsic_base(intr));
#if PAN_ARCH >= 9
   nir_def *addr = nir_iadd(b, nir_load_constant_base_ptr(b, 1, 64),
                            nir_u2u64(b, offset));
#else
   nir_def *base = nir_load_push_constant(
      b, 1, 64, nir_imm_int(b, offsetof(struct mali_graphics_sysvals, common.constant_data)),
      .base = V7_SYSVALS_PUSH_CONST_BASE);
   nir_def *addr = nir_iadd(b, base, nir_u2u64(b, offset));
#endif
   nir_def *val = nir_load_global_constant(
      b, intr->def.num_components, intr->def.bit_size, addr,
      .access = ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE,
      .align_mul = nir_intrinsic_align_mul(intr),
      .align_offset = nir_intrinsic_align_offset(intr));

   nir_def_replace(&intr->def, val);
   return true;
}

static bool
lower_descriptors_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct lower_desc_ctx *ctx = data;

   if (instr->type == nir_instr_type_tex)
      return lower_tex(b, nir_instr_as_tex(instr), ctx);

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_vulkan_resource_reindex:
   case nir_intrinsic_load_vulkan_descriptor:
      return lower_res_intrinsic(b, intr, ctx);
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
      return lower_img_intrinsic(b, intr, ctx);
   default:
      return false;
   }
}

/* First pass: which sets the shader uses, and how many elements of each
 * binding it can reach (v9+: dynamic buffers only; v7: everything with a
 * table). */
static void
record_binding(struct lower_desc_ctx *ctx, uint32_t set, uint32_t binding,
               enum mali_subdesc sub, uint32_t max_idx)
{
   const struct mali_descriptor_set_binding_layout *bl =
      binding_layout(ctx, set, binding);

   ctx->used_set_mask |= BITFIELD_BIT(set);

#if PAN_ARCH >= 9
   if (!vk_descriptor_type_is_dynamic(bl->type))
      return;
#else
   /* Storage buffers are read from the set. */
   if (bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      return;
#endif

   struct desc_id id = {
      .set = set,
      .sampler_subdesc = sub == MALI_SUBDESC_SAMPLER,
      .binding = binding,
   };
   uint32_t old_count =
      (uintptr_t)_mesa_hash_table_u64_search(ctx->ht, id.ht_key);
   uint32_t new_count = max_idx == UINT32_MAX ? bl->desc_count : max_idx + 1;
   assert(new_count <= bl->desc_count);

   if (old_count >= new_count)
      return;

   _mesa_hash_table_u64_insert(ctx->ht, id.ht_key, (void *)(uintptr_t)new_count);
#if PAN_ARCH >= 9
   ctx->dyn_bufs.count += new_count - old_count;
#else
   if (bl->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) {
      ctx->dyn_ubos.count += new_count - old_count;
   } else if (bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
      ctx->dyn_ssbos.count += new_count - old_count;
   } else {
      int table = v7_table_of(bl, id.sampler_subdesc);
      assert(table >= 0);
      ctx->tables[table].count += (new_count - old_count) * v7_array_stride(bl);
   }
#endif
}

static bool
collect_desc_access(nir_builder *b, nir_instr *instr, void *data)
{
   struct lower_desc_ctx *ctx = data;
   uint32_t set, binding, index_imm, max_idx;
   nir_def *index_ssa;

   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      bool recorded = false;

      int idx = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
      if (idx >= 0) {
         get_resource_deref_binding(nir_src_as_deref(tex->src[idx].src), &set,
                                    &binding, &index_imm, &index_ssa, &max_idx);
         record_binding(ctx, set, binding, MALI_SUBDESC_SAMPLER, max_idx);
         recorded = true;
      }
      idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      if (idx >= 0) {
         get_resource_deref_binding(nir_src_as_deref(tex->src[idx].src), &set,
                                    &binding, &index_imm, &index_ssa, &max_idx);
         record_binding(ctx, set, binding, MALI_SUBDESC_TEXTURE, max_idx);
         recorded = true;
      }
      return recorded;
   }

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      record_binding(ctx, nir_intrinsic_desc_set(intr), nir_intrinsic_binding(intr),
                     MALI_SUBDESC_NONE, UINT32_MAX);
      return true;
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
      get_resource_deref_binding(nir_src_as_deref(intr->src[0]), &set, &binding,
                                 &index_imm, &index_ssa, &max_idx);
      record_binding(ctx, set, binding, MALI_SUBDESC_NONE, max_idx);
      return true;
   default:
      return false;
   }
}

#if PAN_ARCH >= 9

/* Lay out the driver table (mali_shader.h, struct mali_shader_desc_info):
 * the dummy sampler after the attributes or varyings, the dynamic buffer
 * copies after it. */
static void
create_copy_table(nir_shader *nir, struct lower_desc_ctx *ctx)
{
   uint32_t dummy_sampler_idx = ctx->driver_table_prefix;
   ctx->dummy_sampler_handle = pan_res_handle(0, dummy_sampler_idx);

   uint32_t copy_count = ctx->dyn_bufs.count;
   if (copy_count == 0)
      return;

   ctx->dyn_bufs_start = dummy_sampler_idx + 1;
   ctx->dyn_bufs.map = rzalloc_array(ctx->ht, uint32_t, copy_count);
   ctx->dyn_bufs.count = 0;

   hash_table_u64_foreach(ctx->ht, he) {
      uint32_t count = (uintptr_t)he.data;
      struct desc_id id = { .ht_key = he.key };
      const struct mali_descriptor_set_binding_layout *bl =
         binding_layout(ctx, id.set, id.binding);

      uint32_t *first = &ctx->dyn_bufs.map[ctx->dyn_bufs.count];
      for (uint32_t i = 0; i < count; i++)
         ctx->dyn_bufs.map[ctx->dyn_bufs.count++] =
            MALI_COPY_DESC_HANDLE(id.set, bl->desc_idx + i);

      /* From here on the table maps (set, binding) to its first copy. */
      _mesa_hash_table_u64_replace(ctx->ht, &he, first);
   }
}

#else /* PAN_ARCH < 9 */

/*
 * Lay out the Bifrost tables (panvk create_copy_table at PAN_ARCH < 9): one
 * map per table and per dynamic buffer kind, all carved from one array in
 * the order dyn_ubos, dyn_ssbos, then the four tables, so that the table
 * maps are contiguous and become the shader's copy map.
 */
static void
v7_create_copy_table(nir_shader *nir, struct lower_desc_ctx *ctx)
{
   uint32_t total = ctx->dyn_ubos.count + ctx->dyn_ssbos.count;
   for (unsigned t = 0; t < MALI_V7_TABLE_COUNT; t++)
      total += ctx->tables[t].count;
   if (!total)
      return;

   uint32_t *arr = rzalloc_array(ctx->ht, uint32_t, total);
   ctx->dyn_ubos.map = arr;
   arr += ctx->dyn_ubos.count;
   ctx->dyn_ubos.count = 0;
   ctx->dyn_ssbos.map = arr;
   arr += ctx->dyn_ssbos.count;
   ctx->dyn_ssbos.count = 0;
   for (unsigned t = 0; t < MALI_V7_TABLE_COUNT; t++) {
      ctx->tables[t].map = arr;
      arr += ctx->tables[t].count;
      ctx->tables[t].count = 0;
   }

   hash_table_u64_foreach(ctx->ht, he) {
      uint32_t count = (uintptr_t)he.data;
      struct desc_id id = { .ht_key = he.key };
      const struct mali_descriptor_set_binding_layout *bl =
         binding_layout(ctx, id.set, id.binding);
      struct desc_map *map;
      if (bl->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
         map = &ctx->dyn_ubos;
      else if (bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
         map = &ctx->dyn_ssbos;
      else
         map = &ctx->tables[v7_table_of(bl, id.sampler_subdesc)];

      const uint32_t stride = mali_desc_stride(bl);
      const uint32_t sub_off = id.sampler_subdesc ? bl->textures_per_desc : 0;
      uint32_t *first = &map->map[map->count];
      for (uint32_t i = 0; i < count; i++) {
         const uint32_t src = vk_descriptor_type_is_dynamic(bl->type)
                                 ? bl->desc_idx + i
                                 : bl->desc_idx + i * stride + sub_off;
         for (uint32_t j = 0; j < v7_array_stride(bl); j++)
            map->map[map->count++] = MALI_COPY_DESC_HANDLE(id.set, src + j);
      }
      _mesa_hash_table_u64_replace(ctx->ht, &he, first);
   }
}

#endif /* PAN_ARCH < 9 */

void
MALI_PER_ARCH(nir_lower_descriptors)(nir_shader *nir,
                                     const struct vk_pipeline_robustness_state *rs,
                                     const struct vk_pipeline_layout *layout,
                                     struct mali_shader_desc_info *desc_info)
{
   struct lower_desc_ctx ctx = {
      .stage = nir->info.stage,
      .add_bounds_checks =
         rs->storage_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT ||
         rs->uniform_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT ||
         rs->images != VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
#if PAN_ARCH >= 9
      .driver_table_prefix = desc_info->driver_table_prefix,
#endif
   };
   bool progress = false;

   ctx.ht = _mesa_hash_table_u64_create(NULL);

   for (uint32_t i = 0; layout && i < layout->set_count && i < MALI_MAX_SETS; i++) {
      if (layout->set_layouts[i])
         ctx.set_layouts[i] = mali_descriptor_set_layout(layout->set_layouts[i]);
   }

   NIR_PASS(progress, nir, nir_shader_instructions_pass, collect_desc_access,
            nir_metadata_all, &ctx);
   if (progress) {
#if PAN_ARCH >= 9
      create_copy_table(nir, &ctx);

      assert(ctx.dyn_bufs.count <= ARRAY_SIZE(desc_info->dyn_bufs.map));
      desc_info->dyn_bufs.count = ctx.dyn_bufs.count;
      memcpy(desc_info->dyn_bufs.map, ctx.dyn_bufs.map,
             ctx.dyn_bufs.count * sizeof(*desc_info->dyn_bufs.map));
#else
      v7_create_copy_table(nir, &ctx);

      assert(ctx.dyn_ubos.count <= ARRAY_SIZE(desc_info->v7.dyn_ubos.map));
      assert(ctx.dyn_ssbos.count <= ARRAY_SIZE(desc_info->v7.dyn_ssbos.map));
      desc_info->v7.dyn_ubos.count = ctx.dyn_ubos.count;
      memcpy(desc_info->v7.dyn_ubos.map, ctx.dyn_ubos.map,
             ctx.dyn_ubos.count * sizeof(uint32_t));
      desc_info->v7.dyn_ssbos.count = ctx.dyn_ssbos.count;
      memcpy(desc_info->v7.dyn_ssbos.map, ctx.dyn_ssbos.map,
             ctx.dyn_ssbos.count * sizeof(uint32_t));
      desc_info->v7.copy_count = 0;
      for (unsigned t = 0; t < MALI_V7_TABLE_COUNT; t++) {
         desc_info->v7.count[t] = ctx.tables[t].count;
         desc_info->v7.copy_count += ctx.tables[t].count;
      }
      /* The copy map (owned by the shader, mali_shader.c frees it): the
       * table maps are contiguous, from the first table's. */
      free(desc_info->v7.copy_map);
      desc_info->v7.copy_map = NULL;
      if (desc_info->v7.copy_count) {
         uint32_t *map = malloc(desc_info->v7.copy_count * sizeof(uint32_t));
         if (map)
            memcpy(map, ctx.tables[0].map, desc_info->v7.copy_count * sizeof(uint32_t));
         else
            desc_info->v7.copy_count = 0;
         desc_info->v7.copy_map = map;
      }
#endif
      desc_info->used_set_mask = ctx.used_set_mask;

      NIR_PASS(_, nir, nir_shader_instructions_pass, lower_descriptors_instr,
               nir_metadata_control_flow, &ctx);
   }

   _mesa_hash_table_u64_destroy(ctx.ht);

   if (nir->constant_data_size)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_load_constant,
               nir_metadata_control_flow, NULL);
}
