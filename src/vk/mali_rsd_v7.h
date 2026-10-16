/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * The Bifrost (v7) Renderer State Descriptor: the parts that come from a
 * compiled shader, shared by the shader upload (the vertex RSD is static
 * per shader, mali_shader.c) and the draw (the fragment RSD is built per
 * pipeline and blend state, mali_pipeline_state.c and mali_jm_cmd_draw.c).
 *
 * Ported from Mesa's src/panfrost/lib/pan_shader.h at PAN_ARCH 7:
 * pan_shader_prepare_rsd, pan_shader_prepare_bifrost_rsd, pan_make_preload
 * and pan_pack_message_preload (MIT, Collabora). That header is not
 * included directly because it pulls in pan_props.h and the kmod layer.
 *
 * Word layout of the 64-byte descriptor (v7.xml, see v7_pack.h):
 *   0-1  shader pointer           (shader)
 *   2    sampler and texture counts, 3 attribute and varying counts
 *   4    properties               (shader side and draw side bits mixed)
 *   5-7  depth units, factor, bias clamp
 *   8    multisample misc, 9 stencil mask misc, 10-11 stencil front/back
 *   12   preload, 13 secondary preload, 14-15 secondary shader and the
 *        message preloads
 * The pipeline bakes words 4 to 11 (MALI_V7_RSD_STATE_WORD_FIRST ..
 * +MALI_V7_RSD_STATE_WORDS) from its static state and the fragment shader;
 * the draw packs the rest from the shader and overlays the baked words.
 */

#ifndef MALI_RSD_V7_H
#define MALI_RSD_V7_H

#if !defined(PAN_ARCH) || PAN_ARCH >= 9
#error "mali_rsd_v7.h is for the Bifrost (PAN_ARCH 7) build only"
#endif

#include "genxml/gen_macros.h"

#include "panfrost/compiler/pan_compiler.h"
#include "util/macros.h"

#define MALI_V7_RSD_STATE_WORD_FIRST 4
#define MALI_V7_RSD_STATE_WORDS 8

static inline enum mali_shader_register_allocation
mali_v7_register_allocation(unsigned work_reg_count)
{
   return work_reg_count <= 32 ? MALI_SHADER_REGISTER_ALLOCATION_32_PER_THREAD
                               : MALI_SHADER_REGISTER_ALLOCATION_64_PER_THREAD;
}

/* pan_make_preload: which of r55..r62 the shader expects preloaded. */
static inline void
mali_v7_preload(mesa_shader_stage stage, uint64_t preload, struct MALI_PRELOAD *out)
{
#define PRELOADS(reg) ((preload & BITFIELD64_BIT(reg)) != 0)
   switch (stage) {
   case MESA_SHADER_VERTEX:
      out->vertex.position_result_address_lo = PRELOADS(58);
      out->vertex.position_result_address_hi = PRELOADS(59);
      out->vertex.vertex_id = PRELOADS(61);
      out->vertex.instance_id = PRELOADS(62);
      break;
   case MESA_SHADER_FRAGMENT:
      out->fragment.primitive_id = PRELOADS(57);
      out->fragment.primitive_flags = PRELOADS(58);
      out->fragment.fragment_position = PRELOADS(59);
      out->fragment.sample_mask_id = PRELOADS(61);
      out->fragment.coverage = true;
      break;
   default:
      out->compute.local_invocation_xy = PRELOADS(55);
      out->compute.local_invocation_z = PRELOADS(56);
      out->compute.work_group_x = PRELOADS(57);
      out->compute.work_group_y = PRELOADS(58);
      out->compute.work_group_z = PRELOADS(59);
      out->compute.global_invocation_x = PRELOADS(60);
      out->compute.global_invocation_y = PRELOADS(61);
      out->compute.global_invocation_z = PRELOADS(62);
      break;
   }
#undef PRELOADS
}

/* pan_pack_message_preload: the LD_VAR / VAR_TEX the hardware issues before
 * the fragment shader starts (the compiler's message preload optimisation). */
static inline void
mali_v7_message_preload(struct MALI_MESSAGE_PRELOAD *cfg, const struct bifrost_message_preload *msg)
{
   const enum mali_message_preload_register_format regfmt =
      msg->fp16 ? MALI_MESSAGE_PRELOAD_REGISTER_FORMAT_F16
                : MALI_MESSAGE_PRELOAD_REGISTER_FORMAT_F32;

   if (msg->enabled && msg->texture) {
      cfg->type = MALI_MESSAGE_TYPE_VAR_TEX;
      cfg->var_tex.varying_index = msg->varying_index;
      cfg->var_tex.texture_index = msg->texture_index;
      cfg->var_tex.register_format = regfmt;
      cfg->var_tex.skip = msg->skip;
      cfg->var_tex.zero_lod = msg->zero_lod;
   } else if (msg->enabled) {
      cfg->type = MALI_MESSAGE_TYPE_LD_VAR;
      cfg->ld_var.varying_index = msg->varying_index;
      cfg->ld_var.register_format = regfmt;
      cfg->ld_var.num_components = msg->num_components;
   } else {
      cfg->type = MALI_MESSAGE_TYPE_DISABLED;
   }
}

/*
 * pan_shader_prepare_rsd for Bifrost: the shader's own fields. `code` is
 * the GPU address of the machine code (0 for a draw without a fragment
 * shader, whose RSD is then shader-less). The vertex shaders of this driver
 * are compiled without IDVS (mali_shader.c), so there is no secondary
 * shader; the counts come from the shader info the compile patched
 * (attribute_count, texture/sampler/ubo counts, mali_shader.c).
 */
static inline void
mali_v7_rsd_shader(const struct pan_shader_info *info, uint64_t code,
                   struct MALI_RENDERER_STATE *rsd)
{
   rsd->shader.shader = code;
   rsd->shader.attribute_count = info->attribute_count;
   rsd->shader.varying_count = info->varyings.formats.count;
   rsd->shader.texture_count = info->texture_count;
   rsd->shader.sampler_count = info->sampler_count;
   rsd->properties.shader_contains_barrier = info->contains_barrier;
   rsd->properties.uniform_buffer_count = info->ubo_count;

   if (info->stage == MESA_SHADER_FRAGMENT) {
      rsd->properties.stencil_from_shader = info->fs.writes_stencil;
      rsd->properties.depth_source = info->fs.writes_depth ? MALI_DEPTH_SOURCE_SHADER
                                                           : MALI_DEPTH_SOURCE_FIXED_FUNCTION;
      /* The draw ORs in the API's per-sample shading. */
      rsd->multisample_misc.evaluate_per_sample = info->fs.sample_shading;
   }

   /* pan_shader_prepare_bifrost_rsd. */
   rsd->preload.uniform_count = DIV_ROUND_UP(info->fau.count, 2);
   rsd->properties.shader_register_allocation = mali_v7_register_allocation(info->work_reg_count);
   mali_v7_preload(info->stage, info->preload, &rsd->preload);

   if (info->stage == MESA_SHADER_FRAGMENT) {
      rsd->properties.shader_modifies_coverage = info->fs.writes_coverage || info->fs.can_discard;
      rsd->properties.allow_forward_pixel_to_be_killed = !info->writes_global;
      rsd->properties.shader_wait_dependency_6 = info->bifrost.wait_6;
      rsd->properties.shader_wait_dependency_7 = info->bifrost.wait_7;
      mali_v7_message_preload(&rsd->message_preload_1, &info->bifrost.messages[0]);
      mali_v7_message_preload(&rsd->message_preload_2, &info->bifrost.messages[1]);
   } else if (info->stage == MESA_SHADER_VERTEX) {
      /* No IDVS on this driver's v7 path: one vertex shader entry. */
      assert(!info->vs.secondary_enable);
   }
}

#endif
