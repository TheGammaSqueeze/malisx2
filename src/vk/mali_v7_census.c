/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * The census mode of the Bifrost (v7) bring-up (mali_arch.h, MALI_TODO_V7):
 * whether it is on, and the once-per-site log line of an unported path.
 *
 * On: MALISX2_V7_CENSUS set to anything but "" or "0", or, when the
 * variable is not set, the Android property debug.malisx2.v7_census set to
 * "1", "true", "on" or "yes" (the way debug.malisx2.measure is read,
 * measurement/config.c). The answer is read once per process.
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
census_value_on(const char *s)
{
   if (!s || !*s)
      return false;
   return strcmp(s, "0") != 0 && strcmp(s, "false") != 0 && strcmp(s, "off") != 0 &&
          strcmp(s, "no") != 0;
}

static bool
census_read(void)
{
   const char *s = getenv("MALISX2_V7_CENSUS");
   if (s)
      return census_value_on(s);
#ifdef __ANDROID__
   char prop[PROP_VALUE_MAX] = "";
   if (__system_property_get("debug.malisx2.v7_census", prop) > 0)
      return census_value_on(prop);
#endif
   return false;
}

bool
mali_v7_census(void)
{
   /* 0 unread, 1 off, 2 on. Every thread computes the same answer, so a
    * race only repeats the read. */
   static int state;
   int s = __atomic_load_n(&state, __ATOMIC_RELAXED);
   if (!s) {
      s = census_read() ? 2 : 1;
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
