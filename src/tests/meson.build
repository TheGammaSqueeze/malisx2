# SPDX-License-Identifier: MIT
#
# Standalone device tests: plain C programs that dlopen the driver and
# reach it through vk_icdGetInstanceProcAddr, so they need no Vulkan loader
# and no ICD manifest on the device. They are built for the host and in the
# Android cross build alike (binaries under <builddir>/src/tests/), and
# depend only on the Vulkan headers of the Mesa checkout and libdl.

# Where the driver library is looked for when no path is given: the Android
# push location (see MALISX2_G52.md), or this build's own library.
_tests_default_icd = host_machine.system() == 'android' \
  ? '/data/local/tmp/malisx2/libvulkan_malisx2.so' \
  : libvulkan_malisx2.full_path()

# clear_g52: a render pass that clears a 64x64 linear RGBA8 image with no
# draws (one fragment job, one kbase atom), read back and checked on the
# CPU. The first thing to run on a new job-manager kernel.
executable(
  'clear_g52',
  files('clear_g52.c'),
  include_directories : [inc_include],
  c_args : [malisx2_c_args, '-DCLEAR_G52_DEFAULT_ICD="@0@"'.format(_tests_default_icd)],
  dependencies : [dep_dl],
  install : false,
)
