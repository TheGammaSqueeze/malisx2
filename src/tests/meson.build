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
# tex_g52: a 4x4 RGBA8 texture uploaded through a staging buffer, bound in
# a descriptor set with a nearest sampler and a uniform buffer, sampled by a
# full-screen triangle onto the 64x64 target, read back and checked texel
# block by texel block (the Bifrost Texture, Sampler and Uniform Buffer
# descriptors and the per-stage tables). The SPIR-V is embedded.
executable(
  'tex_g52',
  files('tex_g52.c'),
  include_directories : [inc_include],
  c_args : [malisx2_c_args, '-DTEX_G52_DEFAULT_ICD="@0@"'.format(_tests_default_icd)],
  dependencies : [dep_dl],
  install : false,
)

# compute_g52: one compute dispatch with a descriptor set (two storage
# buffers read from the set, a storage image through the attribute unit),
# push constants and the compute system values; the buffer and the image
# are read back and checked. The SPIR-V is embedded (see the file comment).
executable(
  'compute_g52',
  files('compute_g52.c'),
  include_directories : [inc_include],
  c_args : [malisx2_c_args, '-DCOMPUTE_G52_DEFAULT_ICD="@0@"'.format(_tests_default_icd)],
  dependencies : [dep_dl],
  install : false,
)

# copy_g52: the transfer commands (buffer copy, fill, update, buffer to a
# tiled image with a ragged patch, image to image, a 1:1 blit, image to
# buffer), every result checked against its pattern on the CPU.
executable(
  'copy_g52',
  files('copy_g52.c'),
  include_directories : [inc_include],
  c_args : [malisx2_c_args, '-DCOPY_G52_DEFAULT_ICD="@0@"'.format(_tests_default_icd)],
  dependencies : [dep_dl],
  install : false,
)

# blit_g52: vkCmdBlitImage on the v7 blit pass: a 32x32 RGBA8 pattern
# scaled 2x nearest and linear, mirrored on X and Y, converted to R5G6B5 and
# back and to sRGB, blitted into a 37x29 rectangle of a prefilled image,
# and the depth blits (D32, D24S8 depth and stencil), every image optimally
# tiled and read back through vkCmdCopyImageToBuffer.
executable(
  'blit_g52',
  files('blit_g52.c'),
  include_directories : [inc_include],
  c_args : [malisx2_c_args, '-DBLIT_G52_DEFAULT_ICD="@0@"'.format(_tests_default_icd)],
  dependencies : [dep_dl, dep_m],
  install : false,
)

# load_g52: loadOp LOAD of colour and depth with a render area of 37x29 at
# (5, 3) and one triangle drawn: the frame shader preloads and the border
# loads of partly covered tiles, outside the area untouched.
executable(
  'load_g52',
  files('load_g52.c'),
  include_directories : [inc_include],
  c_args : [malisx2_c_args, '-DLOAD_G52_DEFAULT_ICD="@0@"'.format(_tests_default_icd)],
  dependencies : [dep_dl],
  install : false,
)

# afbc_g52: AFBC colour images on v7, every case run twice in child
# processes, with MALISX2_V7_AFBC=1 and =0: which images are AFBC, buffer
# and image copies into and out of them per format, a blended draw and a
# partial LOAD pass, sampling an AFBC pattern and an AFBC render target,
# blits AFBC -> linear -> AFBC, and eight AFBC targets in one pass.
executable(
  'afbc_g52',
  files('afbc_g52.c'),
  include_directories : [inc_include],
  c_args : [malisx2_c_args, '-DAFBC_G52_DEFAULT_ICD="@0@"'.format(_tests_default_icd)],
  dependencies : [dep_dl, dep_m],
  install : false,
)

# idx_g52: indexed draws whose vertices sit at both ends of the vertex
# buffer (a wrong vertex range drops them), with the index buffer rewritten
# in place between submits, several draws and binds of one buffer in one
# command buffer, a vertex offset and 32-bit indices, once per host-visible
# memory type (cached non-coherent included).
executable(
  'idx_g52',
  files('idx_g52.c'),
  include_directories : [inc_include],
  c_args : [malisx2_c_args, '-DIDX_G52_DEFAULT_ICD="@0@"'.format(_tests_default_icd)],
  dependencies : [dep_dl],
  install : false,
)
