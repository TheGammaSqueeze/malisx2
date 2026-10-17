/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * The census mode of the Bifrost (v7) bring-up (mali_arch.h, MALI_TODO_V7):
 * whether it is on, and the once-per-site log line of an unported path;
 * and the reader of the v7 triage switches it shares with the other ones
 * (mali_v7_debug_switch).
 *
 * A switch is an environment variable, or, when the variable is not set,
 * an Android property (the way debug.malisx2.measure is read,
 * measurement/config.c): an application started by a launcher gets no
 * environment of ours, but setprop reaches it. "0", "false", "off" and
 * "no" turn a switch off, any other non-empty value on; unset or empty
 * leaves its default. Census: MALISX2_V7_CENSUS / debug.malisx2.v7_census,
 * off by default, read once per process.
 *
 * Built into the v7 library only (src/meson.build), so the v9 and v11
 * objects do not change.
 */

#include "mali_arch.h"

#include <stdlib.h>
#include <string.h>

#include "util/log.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

static bool
switch_value_on(const char *s)
{
   return strcmp(s, "0") != 0 && strcmp(s, "false") != 0 && strcmp(s, "off") != 0 &&
          strcmp(s, "no") != 0;
}

bool
mali_v7_debug_switch(const char *env, const char *prop, bool def)
{
   const char *s = getenv(env);
   if (s && *s)
      return switch_value_on(s);
#ifdef __ANDROID__
   char value[PROP_VALUE_MAX] = "";
   if (__system_property_get(prop, value) > 0 && value[0])
      return switch_value_on(value);
#else
   (void)prop;
#endif
   return def;
}

int
mali_v7_debug_int(const char *env, const char *prop, int def, int min, int max)
{
   const char *s = getenv(env);
#ifdef __ANDROID__
   char value[PROP_VALUE_MAX] = "";
   if (!(s && *s) && __system_property_get(prop, value) > 0 && value[0])
      s = value;
#else
   (void)prop;
#endif
   if (!s || !*s)
      return def;
   char *end = NULL;
   const long v = strtol(s, &end, 10);
   return end != s && !*end && v >= min && v <= max ? (int)v : def;
}

bool
mali_v7_census(void)
{
   /* 0 unread, 1 off, 2 on. Every thread computes the same answer, so a
    * race only repeats the read. */
   static int state;
   int s = __atomic_load_n(&state, __ATOMIC_RELAXED);
   if (!s) {
      s = mali_v7_debug_switch("MALISX2_V7_CENSUS", "debug.malisx2.v7_census", false) ? 2 : 1;
      __atomic_store_n(&state, s, __ATOMIC_RELAXED);
      if (s == 2)
         mesa_logw("malisx2 v7 census: on: unported Bifrost paths log once each and degrade "
                   "instead of failing the command buffer (MALISX2_G52.md, milestone 5)");
   }
   return s == 2;
}

void
mali_v7_todo_log(const char *what, const char *func)
{
   if (mali_v7_census())
      mesa_logw("malisx2 v7 census: %s: %s", func, what);
   else
      mesa_loge("malisx2: TODO v7: %s is not ported to Bifrost yet (%s)", what, func);
}
