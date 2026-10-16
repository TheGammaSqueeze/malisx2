/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Per-arch symbol naming, so a function compiled once per GPU architecture
 * gets a distinct linker symbol instead of colliding with its other-arch
 * copy in the same .so. Mirrors panvk's panvk_per_arch()/panvk_macros.h.
 *
 * MALI_PER_ARCH() needs PAN_ARCH (the same precondition genxml/gen_macros.h
 * has); the declaration and dispatch helpers below do not, so headers that
 * code compiled once also includes can use them.
 */

/* Outside the include guard: a translation unit compiled once may get
 * PAN_ARCH only after a first include (from mali_cs.h), and a later
 * include then defines MALI_PER_ARCH. */
#if defined(PAN_ARCH) && !defined(MALI_PER_ARCH)
#if PAN_ARCH == 7
#define MALI_PER_ARCH(name) mali_v7_##name
#elif PAN_ARCH == 9
#define MALI_PER_ARCH(name) mali_v9_##name
#elif PAN_ARCH == 11
#define MALI_PER_ARCH(name) mali_v11_##name
#else
#error "no per-arch naming for this PAN_ARCH"
#endif
#endif

/*
 * The job-manager back half's internal symbols (mali_jm.h and the JM
 * section of mali_measure.h): the same files are built at PAN_ARCH 7 and
 * 9 into one driver, so these too get the arch's prefix (mali_v7_jm_*,
 * mali_v9_jm_*). Code built once reaches the one it needs through
 * mali_jm_arch_dispatch (mali_jm_measure_reached, measurement/timing.c).
 */
#if defined(PAN_ARCH) && PAN_ARCH < 10 && !defined(MALI_JM_NAMES)
#define MALI_JM_NAMES
#define mali_jm_build_abort MALI_PER_ARCH(jm_build_abort)
#define mali_jm_build_atom MALI_PER_ARCH(jm_build_atom)
#define mali_jm_build_begin MALI_PER_ARCH(jm_build_begin)
#define mali_jm_build_end MALI_PER_ARCH(jm_build_end)
#define mali_jm_build_join_all MALI_PER_ARCH(jm_build_join_all)
#define mali_jm_cmd_add_batch MALI_PER_ARCH(jm_cmd_add_batch)
#define mali_jm_cmd_add_job MALI_PER_ARCH(jm_cmd_add_job)
#define mali_jm_cmd_barrier MALI_PER_ARCH(jm_cmd_barrier)
#define mali_jm_cmd_batch MALI_PER_ARCH(jm_cmd_batch)
#define mali_jm_cmd_batch_close MALI_PER_ARCH(jm_cmd_batch_close)
#define mali_jm_cmd_frag MALI_PER_ARCH(jm_cmd_frag)
#define mali_jm_cmd_mark_frag_transfer MALI_PER_ARCH(jm_cmd_mark_frag_transfer)
#define mali_jm_cmd_note_reset MALI_PER_ARCH(jm_cmd_note_reset)
#define mali_jm_cmd_pass_split MALI_PER_ARCH(jm_cmd_pass_split)
#define mali_jm_cmd_prepare_submit MALI_PER_ARCH(jm_cmd_prepare_submit)
#define mali_jm_cmd_transfer_needs_frag MALI_PER_ARCH(jm_cmd_transfer_needs_frag)
#define mali_jm_cmd_vtc MALI_PER_ARCH(jm_cmd_vtc)
#define mali_jm_cmd_vtc_transfer MALI_PER_ARCH(jm_cmd_vtc_transfer)
#define mali_jm_cmd_write_value MALI_PER_ARCH(jm_cmd_write_value)
#define mali_jm_drain_locked MALI_PER_ARCH(jm_drain_locked)
#define mali_jm_draw_tmpl_init MALI_PER_ARCH(jm_draw_tmpl_init)
#define mali_jm_heap_finish MALI_PER_ARCH(jm_heap_finish)
#define mali_jm_heap_init MALI_PER_ARCH(jm_heap_init)
#define mali_jm_heap_take MALI_PER_ARCH(jm_heap_take)
#define mali_jm_measure_begin MALI_PER_ARCH(jm_measure_begin)
#define mali_jm_measure_capture_atoms MALI_PER_ARCH(jm_measure_capture_atoms)
#define mali_jm_measure_capture_begin MALI_PER_ARCH(jm_measure_capture_begin)
#define mali_jm_measure_capture_end MALI_PER_ARCH(jm_measure_capture_end)
#define mali_jm_measure_cmd_create MALI_PER_ARCH(jm_measure_cmd_create)
#define mali_jm_measure_cmd_destroy MALI_PER_ARCH(jm_measure_cmd_destroy)
#define mali_jm_measure_cmd_reset MALI_PER_ARCH(jm_measure_cmd_reset)
#define mali_jm_measure_end MALI_PER_ARCH(jm_measure_end)
#define mali_jm_measure_info MALI_PER_ARCH(jm_measure_info)
#define mali_jm_measure_reached MALI_PER_ARCH(jm_measure_reached)
#define mali_jm_measure_shader MALI_PER_ARCH(jm_measure_shader)
#define mali_jm_measure_submit MALI_PER_ARCH(jm_measure_submit)
#define mali_jm_sync_file_finish MALI_PER_ARCH(jm_sync_file_finish)
#define mali_jm_wait_locked MALI_PER_ARCH(jm_wait_locked)
#endif

#ifndef MALI_ARCH_H
#define MALI_ARCH_H

#include <stdbool.h>

#include "util/log.h"

/*
 * A v9 construct the Bifrost (v7) build has not ported yet: say so once
 * per site, then the caller fails (VK_ERROR_FEATURE_NOT_PRESENT or a
 * zero result its callers treat as failure), never pretends it worked.
 * Each site names the panvk_v7 function it should be ported from in a
 * TODO v7 comment next to it.
 *
 * Census mode (bring-up only): with MALISX2_V7_CENSUS=1 in the
 * environment, or the Android property debug.malisx2.v7_census=1, each
 * site logs "malisx2 v7 census: <function>: <what>" once instead, and the
 * caller asks mali_v7_census() whether to degrade to a no-op or the
 * closest safe behaviour rather than fail, so that one application run
 * names every unported path it reaches. Off by default: the sites then
 * log and fail exactly as before. Every site and what it does in census
 * mode is listed in MALISX2_G52.md (milestone 5). The two functions are in
 * mali_v7_census.c, which only the v7 library builds: only v7 code
 * expands the macro.
 */
void mali_v7_todo_log(const char *what, const char *func);
bool mali_v7_census(void);

/* A v7 triage switch: the environment variable `env`, else the Android
 * property `prop`, else `def` (mali_v7_census.c). Not cached: callers
 * read it once. */
bool mali_v7_debug_switch(const char *env, const char *prop, bool def);

#define MALI_TODO_V7(what)                                                     \
   do {                                                                        \
      static bool mali_todo_v7_said_;                                          \
      if (!mali_todo_v7_said_) {                                               \
         mali_todo_v7_said_ = true;                                            \
         mali_v7_todo_log(what, __func__);                                     \
      }                                                                        \
   } while (0)

/*
 * Declares every arch's variant of a per-arch function:
 *    MALI_PER_ARCH_DECL(void, pack_dummy_sampler, (void *out));
 * declares mali_v7_pack_dummy_sampler, mali_v9_pack_dummy_sampler and
 * mali_v11_pack_dummy_sampler.
 * Per-arch code calls its own with MALI_PER_ARCH(name)(...), code compiled
 * once picks one with mali_arch_dispatch.
 */
#define MALI_PER_ARCH_DECL(ret, name, params)                                  \
   ret mali_v7_##name params;                                                  \
   ret mali_v9_##name params;                                                  \
   ret mali_v11_##name params

/*
 * The job-manager back half (mali_jm_*.c) is built at PAN_ARCH 7 and 9;
 * declares both variants of one of its functions.
 */
#define MALI_PER_JM_ARCH_DECL(ret, name, params)                               \
   ret mali_v7_##name params;                                                  \
   ret mali_v9_##name params

/*
 * For code compiled once (not per arch): call the variant that matches a
 * device found at runtime to have this arch. Only for cold paths (device
 * creation, pipeline-cache import, sync-type and image-layout choices);
 * never on the per-draw path, which reaches the right variant directly
 * through the device's dispatch table.
 */
#define mali_arch_dispatch(arch, name, ...)                                    \
   ((arch) == 7 ? mali_v7_##name(__VA_ARGS__)                                  \
    : (arch) == 9 ? mali_v9_##name(__VA_ARGS__)                                \
    : mali_v11_##name(__VA_ARGS__))

/* The same for the job-manager back half's own functions (arch 7 or 9). */
#define mali_jm_arch_dispatch(arch, name, ...)                                 \
   ((arch) == 7 ? mali_v7_##name(__VA_ARGS__) : mali_v9_##name(__VA_ARGS__))

/* Whether a device of this arch runs on the job-manager back half. */
#define mali_arch_is_jm(arch) ((arch) == 7 || (arch) == 9)

#endif
