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

# draw_g52: the same image cleared to black, then one full-screen triangle
# drawn into it through a vertex + fragment pipeline (a Vertex job and a
# Tiler job on the Bifrost path), read back and checked for the triangle
# colour. The SPIR-V is embedded (see the file comment).
executable(
  'draw_g52',
  files('draw_g52.c'),
  include_directories : [inc_include],
  c_args : [malisx2_c_args, '-DDRAW_G52_DEFAULT_ICD="@0@"'.format(_tests_default_icd)],
  dependencies : [dep_dl],
  install : false,
)

# zs_g52: depth and stencil. One pass per depth/stencil format the device
# offers (D24S8, D32S8, D32, D16): a near and a far triangle under the depth
# test LESS, then a stencil write and a stencil-tested draw, read back
# through the colour image. g52_harness.h holds the shared setup.
executable(
  'zs_g52',
  files('zs_g52.c'),
  include_directories : [inc_include],
  c_args : [malisx2_c_args, '-DZS_G52_DEFAULT_ICD="@0@"'.format(_tests_default_icd)],
  dependencies : [dep_dl],
  install : false,
)

# blend_g52: three images cleared to grey and drawn with a fixed-function
# equation and a colour write mask, a blend-shader equation (MIN) and a
# blend-constant equation, checked within one LSB.
executable(
  'blend_g52',
  files('blend_g52.c'),
  include_directories : [inc_include],
  c_args : [malisx2_c_args, '-DBLEND_G52_DEFAULT_ICD="@0@"'.format(_tests_default_icd)],
  dependencies : [dep_dl, dep_m],
  install : false,
)
