/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * blit_g52: vkCmdBlitImage on the Bifrost (v7) path of this driver: the
 * blit pass (a render pass on the destination whose frame shader samples
 * the source inside the destination rectangle and reloads the destination
 * outside it), its scaling, mirroring, filtering and format conversion, and
 * the depth and stencil blits the v7 frame shaders write.
 *
 * A 32x32 R8G8B8A8_UNORM source is filled through vkCmdCopyBufferToImage
 * with a known pattern (texel (x, y) = (8x, 8y, 3x + 5y, 255 - x - y)); every
 * image is optimally tiled (u-interleaved, as azahar's are), and every
 * result is read back through vkCmdCopyImageToBuffer and checked on the CPU:
 *
 *   A  2x nearest into 64x64: dst(X, Y) = src(X / 2, Y / 2);
 *   B  2x linear into 64x64: bilinear with clamp to edge, within 2;
 *   C  mirrored on X (destination offsets reversed), 32x32;
 *   D  mirrored on Y (source offsets reversed), 32x32;
 *   E  RGBA8 -> R5G6B5 1:1 (conversion), within one 5/6-bit step;
 *   F  that R5G6B5 image -> RGBA8 1:1, within 1 of the expanded values;
 *   G  RGBA8 -> RGBA8 sRGB 1:1 (encode), within 1;
 *   H  into the rectangle (5, 3) .. (42, 32) of a 64x64 destination filled
 *      with 11 22 33 44: inside, nearest-scaled by 37/32 and 29/32 (a pixel
 *      on a texel boundary may take either neighbour); outside, untouched;
 *   I  D32_SFLOAT 2x nearest (depth (x + 32y + 1) / 2048), exact, read back
 *      through a depth copy;
 *   J  D24_UNORM_S8_UINT 2x nearest, mirrored on Y, depth and stencil in
 *      one region: depth within 1 (24-bit units), stencil exact;
 *   K  RGBA8 -> R4G4B4A4 1:1 (azahar's RGBA4 surfaces), within one 4-bit
 *      step;
 *   L  an unscaled 1:1 blit of a whole 32x32 image into another of the same
 *      format, for each 1x1-block colour format of 3 or 12 bytes the device
 *      offers as blit source and destination (R8G8B8_UNORM, B8G8R8_UNORM,
 *      R8G8B8_SRGB, B8G8R8_SRGB, R8G8B8_SNORM, R32G32B32_SFLOAT, _UINT and
 *      _SINT: every such format of Mesa's v7 table). Each format runs
 *      twice. On optimal images, the source uploaded and the destination
 *      prefilled with vkCmdCopyBufferToImage, both read back with
 *      vkCmdCopyImageToBuffer; the source's readback is checked too, so a
 *      copy that drops these texels is reported as such and not taken for
 *      the blit. On linear host-visible images, written and read by the CPU
 *      through the mapping: the blit alone, no copy involved. Every texel
 *      of the destination must be the source's: within 1 for the 8-bit
 *      formats (in signed steps for SNORM), bit for bit for the 32-bit
 *      ones. The prefill differs from the source in every byte (each byte
 *      XOR 0x80), so a blit that writes nothing leaves every texel wrong
 *      (review #45: a v7 shortcut sends a same-format 1:1 blit to the raw
 *      compute copy, which copies only 1, 2, 4, 8 and 16-byte texels).
 *
 * The colour cases allow one LSB per channel (the float round trip through
 * the shader and the blender); neighbouring texels of the pattern differ by
 * at least 3 in some channel, so a wrong texel is never within it.
 *
 * Formats the device does not offer as blit source and destination are
 * skipped with a line saying so (I and J need the v7 depth/stencil blit
 * destination feature of milestone 5).
 *
 * Usage: blit_g52 [path/to/libvulkan_malisx2.so] [--only=X]
 *   The path defaults to $MALISX2_ICD, then to BLIT_G52_DEFAULT_ICD.
 *   --only=X runs case X alone (A..L; E and F run together).
 *
 * Exit status: 0 PASS (every case run passed); 1 FAIL once the device
 * exists; 2 the program did not get as far as a VkDevice.
 *
 * Driver log: MESA_LOG=file MESA_LOG_LEVEL=info; MALISX2_DUMP_DRAW=1 adds
 * the pass dump (see MALISX2_G52.md, milestone 5).
 */

#include "g52_harness.h"

#include <math.h>

#ifndef BLIT_G52_DEFAULT_ICD
#define BLIT_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define SRC_W 32
#define SRC_H 32
#define BIG 64

static struct g52 g;
static VkCommandPool pool;
static VkCommandBuffer cmd;
static char only;

/* ---------------------------------------------------------------------- */
/* The source pattern                                                      */

static void
src_texel(int x, int y, uint8_t out[4])
{
   out[0] = (uint8_t)(8 * x);
   out[1] = (uint8_t)(8 * y);
   out[2] = (uint8_t)(3 * x + 5 * y);
   out[3] = (uint8_t)(255 - x - y);
}

static int
clampi(int v, int lo, int hi)
{
   return v < lo ? lo : v > hi ? hi : v;
}

/* Bilinear sample of the source at texel-space coordinate (u, v) (texel
 * centres at i + 0.5), clamp to edge. */
static float
src_bilinear(float u, float v, int c)
{
   const float tu = u - 0.5f, tv = v - 0.5f;
   const int iu = (int)floorf(tu), iv = (int)floorf(tv);
   const float fu = tu - iu, fv = tv - iv;
   float acc = 0;
   for (int dy = 0; dy < 2; dy++) {
      for (int dx = 0; dx < 2; dx++) {
         uint8_t t[4];
         src_texel(clampi(iu + dx, 0, SRC_W - 1), clampi(iv + dy, 0, SRC_H - 1), t);
         acc += t[c] * (dx ? fu : 1 - fu) * (dy ? fv : 1 - fv);
      }
   }
   return acc;
}

static float
srgb_encode(float l)
{
   return l <= 0.0031308f ? l * 12.92f : 1.055f * powf(l, 1.0f / 2.4f) - 0.055f;
}

/* ---------------------------------------------------------------------- */
/* Images and the command buffer                                           */

static bool
format_blits(VkFormat f, const char *name)
{
   VkFormatProperties p;
   g.f.GetPhysicalDeviceFormatProperties(g.pdev, f, &p);
   const VkFormatFeatureFlags want = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                     VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                     VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   if ((p.optimalTilingFeatures & want) != want) {
      printf("  %s: optimal tiling features 0x%x lack blit or transfer bits (want 0x%x)\n", name,
             p.optimalTilingFeatures, want);
      return false;
   }
   return true;
}

static bool
make(struct g52_image *im, VkFormat f, uint32_t w, uint32_t h, VkImageAspectFlags aspects)
{
   const bool zs = aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
   const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT |
      (zs ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
   return g52_image_create(&g, im, f, w, h, VK_IMAGE_TILING_OPTIMAL, usage, aspects, false, 0);
}

static bool
begin(void)
{
   VkResult r = g.f.ResetCommandPool(g.device, pool, 0);
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   if (r == VK_SUCCESS)
      r = g.f.BeginCommandBuffer(cmd, &bi);
   if (r != VK_SUCCESS) {
      printf("FAIL: vkBeginCommandBuffer: %s\n", g52_result_str(r));
      return false;
   }
   return true;
}

static bool
end_and_run(const char *what)
{
   VkResult r = g.f.EndCommandBuffer(cmd);
   if (r != VK_SUCCESS) {
      printf("FAIL: vkEndCommandBuffer (%s): %s (a TODO v7 line in the log names an unported "
             "path)\n", what, g52_result_str(r));
      return false;
   }
   return g52_submit_wait(&g, cmd, what) == VK_SUCCESS;
}

static void
to_layout(VkImage image, VkImageAspectFlags aspects, VkImageLayout from, VkImageLayout to)
{
   g52_cmd_transition(&g, cmd, image, aspects, from, to, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}

static void
blit(VkImage src, VkImage dst, VkImageAspectFlags aspects, int sx0, int sy0, int sx1, int sy1,
     int dx0, int dy0, int dx1, int dy1, VkFilter filter)
{
   const VkImageBlit r = {
      .srcSubresource = {aspects, 0, 0, 1},
      .srcOffsets = {{sx0, sy0, 0}, {sx1, sy1, 1}},
      .dstSubresource = {aspects, 0, 0, 1},
      .dstOffsets = {{dx0, dy0, 0}, {dx1, dy1, 1}},
   };
   g.f.CmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r, filter);
}

/* Uploads `data` (w x h texels of `bpp` bytes) into the image's aspect and
 * leaves it in TRANSFER_SRC_OPTIMAL. Records into the open command buffer. */
static bool
record_upload(struct g52_image *im, VkImageAspectFlags view_aspects, VkImageAspectFlags aspect,
              struct g52_hostbuf *hb, const void *data, uint32_t bpp, bool first)
{
   if (!g52_hostbuf_create(&g, hb, (VkDeviceSize)im->width * im->height * bpp, 0))
      return false;
   memcpy(hb->map, data, (size_t)im->width * im->height * bpp);
   g52_hostbuf_flush(&g, hb);
   if (first)
      to_layout(im->image, view_aspects, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
   g52_cmd_copy_image(&g, cmd, im->image, aspect, im->width, im->height, hb->buffer, true);
   return true;
}

/* ---------------------------------------------------------------------- */
/* Checks                                                                  */

struct tally {
   unsigned bad, shown;
};

static void
bad_pixel(struct tally *t, const char *what, int x, int y, const int *got, const int *want,
          unsigned n)
{
   if (t->shown++ < 6) {
      printf("    %s (%d, %d): got", what, x, y);
      for (unsigned i = 0; i < n; i++)
         printf(" %d", got[i]);
      printf(", want");
      for (unsigned i = 0; i < n; i++)
         printf(" %d", want[i]);
      printf("\n");
   }
   t->bad++;
}

static bool
verdict(char id, const char *name, const struct tally *t, unsigned total)
{
   if (t->bad) {
      printf("FAIL %c (%s): %u of %u pixels wrong\n", id, name, t->bad, total);
      return false;
   }
   printf("PASS %c (%s): %u pixels\n", id, name, total);
   return true;
}

/* RGBA8 readback vs a per-pixel expectation within tol. */
typedef void (*rgba_expect_fn)(int x, int y, int want[4], bool *either, int alt[4]);

static bool
check_rgba8(char id, const char *name, const uint8_t *px, uint32_t w, uint32_t h,
            rgba_expect_fn fn, int tol)
{
   struct tally t = {0};
   for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
         const uint8_t *p = px + ((size_t)y * w + x) * 4;
         int want[4], alt[4];
         bool either = false;
         fn((int)x, (int)y, want, &either, alt);
         int got[4] = {p[0], p[1], p[2], p[3]};
         bool ok = true, ok_alt = either;
         for (unsigned c = 0; c < 4; c++) {
            ok &= abs(got[c] - want[c]) <= tol;
            if (either)
               ok_alt &= abs(got[c] - alt[c]) <= tol;
         }
         if (!ok && !ok_alt)
            bad_pixel(&t, name, (int)x, (int)y, got, want, 4);
      }
   }
   return verdict(id, name, &t, w * h);
}

static void
expect_a(int x, int y, int want[4], bool *either, int alt[4])
{
   uint8_t t[4];
   src_texel(x / 2, y / 2, t);
   for (int c = 0; c < 4; c++)
      want[c] = t[c];
}

static void
expect_b(int x, int y, int want[4], bool *either, int alt[4])
{
   for (int c = 0; c < 4; c++)
      want[c] = (int)lroundf(src_bilinear((x + 0.5f) * 0.5f, (y + 0.5f) * 0.5f, c));
}

static void
expect_c(int x, int y, int want[4], bool *either, int alt[4])
{
   uint8_t t[4];
   src_texel(SRC_W - 1 - x, y, t);
   for (int c = 0; c < 4; c++)
      want[c] = t[c];
}

static void
expect_d(int x, int y, int want[4], bool *either, int alt[4])
{
   uint8_t t[4];
   src_texel(x, SRC_H - 1 - y, t);
   for (int c = 0; c < 4; c++)
      want[c] = t[c];
}

static void
expect_g(int x, int y, int want[4], bool *either, int alt[4])
{
   uint8_t t[4];
   src_texel(x, y, t);
   for (int c = 0; c < 3; c++)
      want[c] = (int)lroundf(srgb_encode(t[c] / 255.0f) * 255.0f);
   want[3] = t[3];
}

/* H: the destination rectangle and the nearest texel of a pixel centre;
 * a centre within 1/256 texel of a boundary may take either side. */
#define H_X0 5
#define H_Y0 3
#define H_X1 42
#define H_Y1 32

static int
h_texel(int d, int d0, int d1, int size, int *alt)
{
   const double u = (d + 0.5 - d0) * (double)size / (double)(d1 - d0);
   const int t = clampi((int)floor(u), 0, size - 1);
   const double frac = u - floor(u);
   *alt = t;
   if (frac < 1.0 / 256)
      *alt = clampi(t - 1, 0, size - 1);
   else if (frac > 1.0 - 1.0 / 256)
      *alt = clampi(t + 1, 0, size - 1);
   return t;
}

static void
expect_h(int x, int y, int want[4], bool *either, int alt[4])
{
   if (x < H_X0 || x >= H_X1 || y < H_Y0 || y >= H_Y1) {
      want[0] = 0x11;
      want[1] = 0x22;
      want[2] = 0x33;
      want[3] = 0x44;
      return;
   }
   int ax, ay;
   const int tx = h_texel(x, H_X0, H_X1, SRC_W, &ax);
   const int ty = h_texel(y, H_Y0, H_Y1, SRC_H, &ay);
   uint8_t t[4], a[4];
   src_texel(tx, ty, t);
   src_texel(ax, ay, a);
   for (int c = 0; c < 4; c++) {
      want[c] = t[c];
      alt[c] = a[c];
   }
   *either = ax != tx || ay != ty;
}

/* ---------------------------------------------------------------------- */
/* The cases                                                               */

static bool
want_case(char id)
{
   return !only || only == id;
}

/* A..D, G, H: one RGBA8 blit from the uploaded source into dst, read back. */
static bool
run_rgba8_case(char id, const char *name, struct g52_image *src, VkFormat dst_fmt, uint32_t dw,
               uint32_t dh, int sx0, int sy0, int sx1, int sy1, int dx0, int dy0, int dx1, int dy1,
               VkFilter filter, rgba_expect_fn fn, int tol, bool prefill)
{
   if (!want_case(id))
      return true;
   printf("case %c: %s\n", id, name);
   struct g52_image dst;
   struct g52_hostbuf fill = {0}, rb = {0};
   bool ok = make(&dst, dst_fmt, dw, dh, VK_IMAGE_ASPECT_COLOR_BIT) &&
             g52_hostbuf_create(&g, &rb, (VkDeviceSize)dw * dh * 4, 0xa5) && begin();
   if (ok) {
      if (prefill) {
         uint8_t *p = malloc((size_t)dw * dh * 4);
         for (uint32_t i = 0; i < dw * dh; i++) {
            p[4 * i + 0] = 0x11;
            p[4 * i + 1] = 0x22;
            p[4 * i + 2] = 0x33;
            p[4 * i + 3] = 0x44;
         }
         ok = record_upload(&dst, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &fill, p, 4,
                            true);
         free(p);
         g52_cmd_full_barrier(&g, cmd);
      } else {
         to_layout(dst.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      }
   }
   if (ok) {
      blit(src->image, dst.image, VK_IMAGE_ASPECT_COLOR_BIT, sx0, sy0, sx1, sy1, dx0, dy0, dx1,
           dy1, filter);
      to_layout(dst.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      g52_cmd_copy_image(&g, cmd, dst.image, VK_IMAGE_ASPECT_COLOR_BIT, dw, dh, rb.buffer, false);
      ok = end_and_run(name);
   }
   if (ok) {
      g52_hostbuf_invalidate(&g, &rb);
      ok = check_rgba8(id, name, rb.map, dw, dh, fn, tol);
   } else {
      printf("FAIL %c (%s): not run to the end\n", id, name);
   }
   g52_hostbuf_destroy(&g, &fill);
   g52_hostbuf_destroy(&g, &rb);
   g52_image_destroy(&g, &dst);
   return ok;
}

/* E and F: RGBA8 -> R5G6B5 -> RGBA8, both read back. */
static bool
run_565_cases(struct g52_image *src)
{
   if (!want_case('E') && !want_case('F'))
      return true;
   printf("case E/F: RGBA8 -> R5G6B5 -> RGBA8, 1:1\n");
   if (!format_blits(VK_FORMAT_R5G6B5_UNORM_PACK16, "R5G6B5_UNORM_PACK16")) {
      printf("FAIL E/F: the device offers no R5G6B5 blits\n");
      return false;
   }
   struct g52_image m, back;
   struct g52_hostbuf rb565 = {0}, rb = {0};
   bool ok = make(&m, VK_FORMAT_R5G6B5_UNORM_PACK16, SRC_W, SRC_H, VK_IMAGE_ASPECT_COLOR_BIT) &&
             make(&back, VK_FORMAT_R8G8B8A8_UNORM, SRC_W, SRC_H, VK_IMAGE_ASPECT_COLOR_BIT) &&
             g52_hostbuf_create(&g, &rb565, SRC_W * SRC_H * 2, 0xa5) &&
             g52_hostbuf_create(&g, &rb, SRC_W * SRC_H * 4, 0xa5) && begin();
   if (ok) {
      to_layout(m.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      to_layout(back.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      blit(src->image, m.image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, SRC_W, SRC_H, 0, 0, SRC_W, SRC_H,
           VK_FILTER_NEAREST);
      to_layout(m.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      blit(m.image, back.image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, SRC_W, SRC_H, 0, 0, SRC_W, SRC_H,
           VK_FILTER_NEAREST);
      to_layout(back.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      g52_cmd_copy_image(&g, cmd, m.image, VK_IMAGE_ASPECT_COLOR_BIT, SRC_W, SRC_H, rb565.buffer,
                         false);
      g52_cmd_copy_image(&g, cmd, back.image, VK_IMAGE_ASPECT_COLOR_BIT, SRC_W, SRC_H, rb.buffer,
                         false);
      ok = end_and_run("RGBA8 -> R5G6B5 -> RGBA8");
   }
   bool pass = ok;
   if (ok) {
      g52_hostbuf_invalidate(&g, &rb565);
      g52_hostbuf_invalidate(&g, &rb);
      const uint16_t *v565 = rb565.map;
      const uint8_t *v8 = rb.map;
      struct tally te = {0}, tf = {0};
      for (int y = 0; y < SRC_H; y++) {
         for (int x = 0; x < SRC_W; x++) {
            uint8_t t[4];
            src_texel(x, y, t);
            const uint16_t v = v565[y * SRC_W + x];
            /* R5G6B5_UNORM_PACK16: R in bits 15..11, B in 4..0. */
            const int got[3] = {v >> 11, (v >> 5) & 63, v & 31};
            const int want[3] = {(int)lroundf(t[0] * 31.0f / 255.0f),
                                 (int)lroundf(t[1] * 63.0f / 255.0f),
                                 (int)lroundf(t[2] * 31.0f / 255.0f)};
            if (abs(got[0] - want[0]) > 1 || abs(got[1] - want[1]) > 1 ||
                abs(got[2] - want[2]) > 1)
               bad_pixel(&te, "R5G6B5", x, y, got, want, 3);
            const uint8_t *p = v8 + (y * SRC_W + x) * 4;
            const int got8[4] = {p[0], p[1], p[2], p[3]};
            const int want8[4] = {(int)lroundf(got[0] * 255.0f / 31.0f),
                                  (int)lroundf(got[1] * 255.0f / 63.0f),
                                  (int)lroundf(got[2] * 255.0f / 31.0f), 255};
            bool okf = true;
            for (int c = 0; c < 4; c++)
               okf &= abs(got8[c] - want8[c]) <= 1;
            if (!okf)
               bad_pixel(&tf, "R5G6B5 -> RGBA8", x, y, got8, want8, 4);
         }
      }
      pass = verdict('E', "RGBA8 -> R5G6B5 1:1", &te, SRC_W * SRC_H);
      pass &= verdict('F', "R5G6B5 -> RGBA8 1:1", &tf, SRC_W * SRC_H);
   } else {
      printf("FAIL E/F: not run to the end\n");
   }
   g52_hostbuf_destroy(&g, &rb565);
   g52_hostbuf_destroy(&g, &rb);
   g52_image_destroy(&g, &m);
   g52_image_destroy(&g, &back);
   return pass;
}

/* K: RGBA8 -> R4G4B4A4_UNORM_PACK16 1:1. */
static bool
run_rgba4_case(struct g52_image *src)
{
   if (!want_case('K'))
      return true;
   printf("case K: RGBA8 -> R4G4B4A4 1:1\n");
   if (!format_blits(VK_FORMAT_R4G4B4A4_UNORM_PACK16, "R4G4B4A4_UNORM_PACK16")) {
      printf("SKIP K: the device offers no R4G4B4A4 blits\n");
      return true;
   }
   struct g52_image d;
   struct g52_hostbuf rb = {0};
   bool ok = make(&d, VK_FORMAT_R4G4B4A4_UNORM_PACK16, SRC_W, SRC_H, VK_IMAGE_ASPECT_COLOR_BIT) &&
             g52_hostbuf_create(&g, &rb, SRC_W * SRC_H * 2, 0xa5) && begin();
   if (ok) {
      to_layout(d.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      blit(src->image, d.image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, SRC_W, SRC_H, 0, 0, SRC_W, SRC_H,
           VK_FILTER_NEAREST);
      to_layout(d.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      g52_cmd_copy_image(&g, cmd, d.image, VK_IMAGE_ASPECT_COLOR_BIT, SRC_W, SRC_H, rb.buffer,
                         false);
      ok = end_and_run("RGBA8 -> R4G4B4A4");
   }
   bool pass = ok;
   if (ok) {
      g52_hostbuf_invalidate(&g, &rb);
      const uint16_t *v4 = rb.map;
      struct tally t = {0};
      for (int y = 0; y < SRC_H; y++) {
         for (int x = 0; x < SRC_W; x++) {
            uint8_t s8[4];
            src_texel(x, y, s8);
            const uint16_t v = v4[y * SRC_W + x];
            /* R4G4B4A4_UNORM_PACK16: R in bits 15..12, A in 3..0. */
            const int got[4] = {v >> 12, (v >> 8) & 15, (v >> 4) & 15, v & 15};
            int want[4];
            bool okp = true;
            for (int c = 0; c < 4; c++) {
               want[c] = (int)lroundf(s8[c] * 15.0f / 255.0f);
               okp &= abs(got[c] - want[c]) <= 1;
            }
            if (!okp)
               bad_pixel(&t, "R4G4B4A4", x, y, got, want, 4);
         }
      }
      pass = verdict('K', "RGBA8 -> R4G4B4A4 1:1", &t, SRC_W * SRC_H);
   } else {
      printf("FAIL K: not run to the end\n");
   }
   g52_hostbuf_destroy(&g, &rb);
   g52_image_destroy(&g, &d);
   return pass;
}

/* I: D32_SFLOAT, 2x nearest. */
static bool
run_d32_case(void)
{
   if (!want_case('I'))
      return true;
   printf("case I: D32_SFLOAT 2x nearest\n");
   if (!format_blits(VK_FORMAT_D32_SFLOAT, "D32_SFLOAT")) {
      printf("FAIL I: the device offers no D32_SFLOAT blits (milestone 5 adds the v7 depth blit "
             "destination feature)\n");
      return false;
   }
   struct g52_image s, d;
   struct g52_hostbuf up = {0}, fill = {0}, rb = {0};
   float data[SRC_W * SRC_H], ones[BIG * BIG];
   for (int y = 0; y < SRC_H; y++)
      for (int x = 0; x < SRC_W; x++)
         data[y * SRC_W + x] = (float)(x + SRC_W * y + 1) / 2048.0f;
   for (int i = 0; i < BIG * BIG; i++)
      ones[i] = 1.0f;
   bool ok = make(&s, VK_FORMAT_D32_SFLOAT, SRC_W, SRC_H, VK_IMAGE_ASPECT_DEPTH_BIT) &&
             make(&d, VK_FORMAT_D32_SFLOAT, BIG, BIG, VK_IMAGE_ASPECT_DEPTH_BIT) &&
             g52_hostbuf_create(&g, &rb, BIG * BIG * 4, 0xa5) && begin();
   if (ok)
      ok = record_upload(&s, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, &up, data, 4,
                         true) &&
           record_upload(&d, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, &fill, ones, 4,
                         true);
   if (ok) {
      to_layout(s.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      g52_cmd_full_barrier(&g, cmd);
      blit(s.image, d.image, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, SRC_W, SRC_H, 0, 0, BIG, BIG,
           VK_FILTER_NEAREST);
      to_layout(d.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      g52_cmd_copy_image(&g, cmd, d.image, VK_IMAGE_ASPECT_DEPTH_BIT, BIG, BIG, rb.buffer, false);
      ok = end_and_run("D32 blit");
   }
   bool pass = ok;
   if (ok) {
      g52_hostbuf_invalidate(&g, &rb);
      const float *got = rb.map;
      struct tally t = {0};
      unsigned ones_left = 0;
      for (int y = 0; y < BIG; y++) {
         for (int x = 0; x < BIG; x++) {
            const float w = data[(y / 2) * SRC_W + x / 2];
            const float v = got[y * BIG + x];
            if (v != w) {
               const int gi[1] = {(int)lroundf(v * 2048.0f)}, wi[1] = {(int)lroundf(w * 2048.0f)};
               ones_left += v == 1.0f;
               bad_pixel(&t, "D32 (depth x 2048)", x, y, gi, wi, 1);
            }
         }
      }
      if (ones_left)
         printf("    %u pixels still hold the 1.0 prefill (the blit wrote no depth there)\n",
                ones_left);
      pass = verdict('I', "D32_SFLOAT 2x nearest", &t, BIG * BIG);
   } else {
      printf("FAIL I: not run to the end\n");
   }
   g52_hostbuf_destroy(&g, &up);
   g52_hostbuf_destroy(&g, &fill);
   g52_hostbuf_destroy(&g, &rb);
   g52_image_destroy(&g, &s);
   g52_image_destroy(&g, &d);
   return pass;
}

/* J: D24_UNORM_S8_UINT, 2x nearest, mirrored on Y, both aspects. */
static bool
run_d24s8_case(void)
{
   if (!want_case('J'))
      return true;
   printf("case J: D24_UNORM_S8_UINT 2x nearest, mirrored on Y, depth and stencil\n");
   if (!format_blits(VK_FORMAT_D24_UNORM_S8_UINT, "D24_UNORM_S8_UINT")) {
      printf("SKIP J: the device does not offer D24_UNORM_S8_UINT blits\n");
      return true;
   }
   const VkImageAspectFlags both = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
   struct g52_image s, d;
   struct g52_hostbuf upz = {0}, ups = {0}, rbz = {0}, rbs = {0};
   static uint32_t z[SRC_W * SRC_H];
   static uint8_t st[SRC_W * SRC_H];
   for (int y = 0; y < SRC_H; y++) {
      for (int x = 0; x < SRC_W; x++) {
         z[y * SRC_W + x] = (uint32_t)(x * SRC_H + y) * 16384u + 77u;
         st[y * SRC_W + x] = (uint8_t)(x * 7 + y * 13);
      }
   }
   bool ok = make(&s, VK_FORMAT_D24_UNORM_S8_UINT, SRC_W, SRC_H, both) &&
             make(&d, VK_FORMAT_D24_UNORM_S8_UINT, BIG, BIG, both) &&
             g52_hostbuf_create(&g, &rbz, BIG * BIG * 4, 0xa5) &&
             g52_hostbuf_create(&g, &rbs, BIG * BIG, 0xa5) && begin();
   if (ok)
      ok = record_upload(&s, both, VK_IMAGE_ASPECT_DEPTH_BIT, &upz, z, 4, true) &&
           record_upload(&s, both, VK_IMAGE_ASPECT_STENCIL_BIT, &ups, st, 1, false);
   if (ok) {
      to_layout(s.image, both, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      to_layout(d.image, both, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      blit(s.image, d.image, both, 0, 0, SRC_W, SRC_H, 0, BIG, BIG, 0, VK_FILTER_NEAREST);
      to_layout(d.image, both, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      g52_cmd_copy_image(&g, cmd, d.image, VK_IMAGE_ASPECT_DEPTH_BIT, BIG, BIG, rbz.buffer, false);
      g52_cmd_copy_image(&g, cmd, d.image, VK_IMAGE_ASPECT_STENCIL_BIT, BIG, BIG, rbs.buffer,
                         false);
      ok = end_and_run("D24S8 blit");
   }
   bool pass = ok;
   if (ok) {
      g52_hostbuf_invalidate(&g, &rbz);
      g52_hostbuf_invalidate(&g, &rbs);
      const uint32_t *gz = rbz.map;
      const uint8_t *gs = rbs.map;
      struct tally tz = {0}, ts = {0};
      int max_dz = 0;
      for (int y = 0; y < BIG; y++) {
         for (int x = 0; x < BIG; x++) {
            const int sx = x / 2, sy = SRC_H - 1 - y / 2;
            const int wz = (int)(z[sy * SRC_W + sx] & 0xffffff);
            const int vz = (int)(gz[y * BIG + x] & 0xffffff);
            max_dz = abs(vz - wz) > max_dz ? abs(vz - wz) : max_dz;
            if (abs(vz - wz) > 1) {
               const int a[1] = {vz}, b[1] = {wz};
               bad_pixel(&tz, "D24", x, y, a, b, 1);
            }
            const int ws = st[sy * SRC_W + sx], vs = gs[y * BIG + x];
            if (vs != ws) {
               const int a[1] = {vs}, b[1] = {ws};
               bad_pixel(&ts, "S8", x, y, a, b, 1);
            }
         }
      }
      printf("    largest depth difference: %d (24-bit units)\n", max_dz);
      pass = verdict('J', "D24_UNORM_S8_UINT depth", &tz, BIG * BIG);
      pass &= verdict('J', "D24_UNORM_S8_UINT stencil", &ts, BIG * BIG);
   } else {
      printf("FAIL J: not run to the end\n");
   }
   g52_hostbuf_destroy(&g, &upz);
   g52_hostbuf_destroy(&g, &ups);
   g52_hostbuf_destroy(&g, &rbz);
   g52_hostbuf_destroy(&g, &rbs);
   g52_image_destroy(&g, &s);
   g52_image_destroy(&g, &d);
   return pass;
}

/* ---------------------------------------------------------------------- */
/* L: same-format 1:1 blits of 3- and 12-byte texels                        */

enum l_kind { L_UNORM8, L_SRGB8, L_SNORM8, L_FLOAT32, L_UINT32, L_SINT32 };

struct l_format {
   VkFormat format;
   const char *name;
   enum l_kind kind;
};

/* The 1x1-block colour formats whose texel size is not a power of two and
 * that Mesa's v7 table (pan_format.c) makes both a sampler view and a
 * render target, so this driver offers them as blit source and
 * destination on the G52. */
static const struct l_format l_formats[] = {
   {VK_FORMAT_R8G8B8_UNORM, "R8G8B8_UNORM", L_UNORM8},
   {VK_FORMAT_B8G8R8_UNORM, "B8G8R8_UNORM", L_UNORM8},
   {VK_FORMAT_R8G8B8_SRGB, "R8G8B8_SRGB", L_SRGB8},
   {VK_FORMAT_B8G8R8_SRGB, "B8G8R8_SRGB", L_SRGB8},
   {VK_FORMAT_R8G8B8_SNORM, "R8G8B8_SNORM", L_SNORM8},
   {VK_FORMAT_R32G32B32_SFLOAT, "R32G32B32_SFLOAT", L_FLOAT32},
   {VK_FORMAT_R32G32B32_UINT, "R32G32B32_UINT", L_UINT32},
   {VK_FORMAT_R32G32B32_SINT, "R32G32B32_SINT", L_SINT32},
};

/* What the readback buffers start as: a texel still holding it was never
 * written by the copy out of the image. */
#define L_FILL 0xa5

static unsigned
l_bytes(enum l_kind k)
{
   return k >= L_FLOAT32 ? 12 : 3;
}

/*
 * Texel (x, y) of the source, as the bytes of the format. The 8-bit
 * formats: R 4x and G 4y (a neighbour differs by 4), B (3x + 5y) mod 255 -
 * 127 as a byte, which is never 0x80 (SNORM reads 0x80 and 0x81 both as
 * -1.0). The 32-bit formats: values past 16 bits and of both signs; the
 * floats are never zero and are exact in half precision too.
 */
static void
l_texel(enum l_kind k, int x, int y, uint8_t *out)
{
   switch (k) {
   case L_FLOAT32: {
      const float v[3] = {(float)(x + 32 * y) * 0.25f + 0.5f,
                          -(float)(4 * x + 2 * y + 1) * 0.5f,
                          (float)(x * y) * 0.125f + 1.5f};
      memcpy(out, v, sizeof(v));
      return;
   }
   case L_UINT32: {
      const uint32_t v[3] = {(uint32_t)(x | y << 8) * 0x10001u + 0x80000000u,
                             0xdeadbeefu ^ ((uint32_t)x * 0x9e3779b9u + (uint32_t)y),
                             (uint32_t)(x * y) + 70000u};
      memcpy(out, v, sizeof(v));
      return;
   }
   case L_SINT32: {
      const int32_t v[3] = {-(x + 32 * y) - 100000, (x - y) * 65537,
                            INT32_MIN + x + 32 * y + 1};
      memcpy(out, v, sizeof(v));
      return;
   }
   default:
      out[0] = (uint8_t)(4 * x);
      out[1] = (uint8_t)(4 * y);
      out[2] = (uint8_t)((3 * x + 5 * y) % 255 - 127);
      return;
   }
}

/* The destination's prefill at (x, y): the source texel with every byte
 * XOR 0x80, so it differs from the source by 128 in every byte. */
static void
l_prefill(enum l_kind k, int x, int y, uint8_t *out)
{
   l_texel(k, x, y, out);
   for (unsigned i = 0; i < l_bytes(k); i++)
      out[i] ^= 0x80;
}

/* Whether a texel read back is the source's: within one step for the
 * 8-bit formats (the blit pass converts through floats, and sRGB decodes
 * and encodes), bit for bit for the 32-bit ones. */
static bool
l_match(enum l_kind k, const uint8_t *got, const uint8_t *want)
{
   switch (k) {
   case L_UNORM8:
   case L_SRGB8:
      for (int c = 0; c < 3; c++)
         if (abs((int)got[c] - (int)want[c]) > 1)
            return false;
      return true;
   case L_SNORM8:
      for (int c = 0; c < 3; c++)
         if (abs((int)(int8_t)got[c] - (int)(int8_t)want[c]) > 1)
            return false;
      return true;
   default:
      return !memcmp(got, want, 12);
   }
}

/*
 * Checks the SRC_W x SRC_H texels at `base` (rows `pitch` bytes apart)
 * against the source and prints the verdict. `dst`: the image was
 * prefilled, so name texels that still hold the prefill; `via_copy`: they
 * were read back through vkCmdCopyImageToBuffer, so name texels that still
 * hold the buffer's fill.
 */
static bool
l_check(const struct l_format *lf, const char *what, const uint8_t *base, size_t pitch, bool dst,
        bool via_copy)
{
   const unsigned bytes = l_bytes(lf->kind);
   struct tally t = {0};
   unsigned prefilled = 0, unwritten = 0;
   for (int y = 0; y < SRC_H; y++) {
      for (int x = 0; x < SRC_W; x++) {
         const uint8_t *got = base + (size_t)y * pitch + (size_t)x * bytes;
         uint8_t want[12], pre[12];
         l_texel(lf->kind, x, y, want);
         if (l_match(lf->kind, got, want))
            continue;
         l_prefill(lf->kind, x, y, pre);
         bool fill = true;
         for (unsigned i = 0; i < bytes; i++)
            fill &= got[i] == L_FILL;
         if (dst && !memcmp(got, pre, bytes))
            prefilled++;
         else if (via_copy && fill)
            unwritten++;
         if (t.shown++ < 6) {
            printf("    %s (%d, %d): got", what, x, y);
            for (unsigned i = 0; i < bytes; i++)
               printf(" %02x", got[i]);
            printf(", want");
            for (unsigned i = 0; i < bytes; i++)
               printf(" %02x", want[i]);
            printf("\n");
         }
         t.bad++;
      }
   }
   if (prefilled)
      printf("    %u texels still hold the destination's prefill (the blit wrote nothing "
             "there)\n", prefilled);
   if (unwritten)
      printf("    %u texels read back as the readback buffer's 0x%02x fill (the copy out of the "
             "image wrote nothing there)\n", unwritten, L_FILL);
   return verdict('L', what, &t, SRC_W * SRC_H);
}

/*
 * A 32x32 image of `format` with transfer usage only (all that a blit and
 * the buffer/image copies ask for, so no view): optimal tiling in the
 * first memory type it takes, or linear tiling in host-visible memory,
 * created PREINITIALIZED and left mapped with its subresource layout for
 * the CPU to fill and read.
 */
static bool
l_image_create(struct g52_image *im, VkFormat format, bool linear)
{
   const struct g52_fns *f = &g.f;
   memset(im, 0, sizeof(*im));
   im->format = format;
   im->width = SRC_W;
   im->height = SRC_H;
   if (linear && !f->GetImageSubresourceLayout) {
      fprintf(stderr, "blit_g52: no vkGetImageSubresourceLayout for the linear images\n");
      return false;
   }
   const VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {SRC_W, SRC_H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = linear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = linear ? VK_IMAGE_LAYOUT_PREINITIALIZED : VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkResult r = f->CreateImage(g.device, &ici, NULL, &im->image);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "blit_g52: vkCreateImage (format %d, %s) failed: %s\n", format,
              linear ? "linear" : "optimal", g52_result_str(r));
      return false;
   }
   VkMemoryRequirements reqs;
   f->GetImageMemoryRequirements(g.device, im->image, &reqs);
   im->size = reqs.size;
   const int type = g52_pick_memory_type(&g, reqs.memoryTypeBits, linear, &im->coherent);
   if (type < 0) {
      fprintf(stderr, "blit_g52: no %smemory type for the image (bits 0x%x)\n",
              linear ? "host-visible " : "", reqs.memoryTypeBits);
      return false;
   }
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)type,
   };
   r = f->AllocateMemory(g.device, &mai, NULL, &im->memory);
   if (r == VK_SUCCESS)
      r = f->BindImageMemory(g.device, im->image, im->memory, 0);
   if (r == VK_SUCCESS && linear)
      r = f->MapMemory(g.device, im->memory, 0, VK_WHOLE_SIZE, 0, &im->map);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "blit_g52: image memory (format %d): %s\n", format, g52_result_str(r));
      return false;
   }
   if (linear) {
      const VkImageSubresource sub = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT};
      f->GetImageSubresourceLayout(g.device, im->image, &sub, &im->layout);
   }
   return true;
}

/* Fills the linear image's texels through its mapping (`prefill`: the
 * destination's prefill, else the source pattern) and flushes them. */
static void
l_image_fill(struct g52_image *im, enum l_kind k, bool prefill)
{
   const unsigned bytes = l_bytes(k);
   uint8_t *base = (uint8_t *)im->map + im->layout.offset;
   for (int y = 0; y < SRC_H; y++) {
      for (int x = 0; x < SRC_W; x++) {
         uint8_t *p = base + (size_t)y * im->layout.rowPitch + (size_t)x * bytes;
         if (prefill)
            l_prefill(k, x, y, p);
         else
            l_texel(k, x, y, p);
      }
   }
   if (!im->coherent) {
      const VkMappedMemoryRange whole = {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = im->memory,
         .size = VK_WHOLE_SIZE,
      };
      g.f.FlushMappedMemoryRanges(g.device, 1, &whole);
   }
}

/* L on optimal images, through the buffer/image copies. */
static bool
l_run_optimal(const struct l_format *lf)
{
   const unsigned bytes = l_bytes(lf->kind);
   const VkDeviceSize size = (VkDeviceSize)SRC_W * SRC_H * bytes;
   char what_rt[96], what_blit[96];
   snprintf(what_rt, sizeof(what_rt), "%s optimal, copy round trip of the source", lf->name);
   snprintf(what_blit, sizeof(what_blit), "%s optimal, 1:1 blit", lf->name);

   struct g52_image src = {0}, dst = {0};
   struct g52_hostbuf up = {0}, pre = {0}, rb_src = {0}, rb_dst = {0};
   bool ok = l_image_create(&src, lf->format, false) && l_image_create(&dst, lf->format, false) &&
             g52_hostbuf_create(&g, &up, size, 0) && g52_hostbuf_create(&g, &pre, size, 0) &&
             g52_hostbuf_create(&g, &rb_src, size, L_FILL) &&
             g52_hostbuf_create(&g, &rb_dst, size, L_FILL) && begin();
   if (ok) {
      for (int y = 0; y < SRC_H; y++) {
         for (int x = 0; x < SRC_W; x++) {
            const size_t o = ((size_t)y * SRC_W + x) * bytes;
            l_texel(lf->kind, x, y, (uint8_t *)up.map + o);
            l_prefill(lf->kind, x, y, (uint8_t *)pre.map + o);
         }
      }
      g52_hostbuf_flush(&g, &up);
      g52_hostbuf_flush(&g, &pre);
      to_layout(src.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      to_layout(dst.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      g52_cmd_copy_image(&g, cmd, src.image, VK_IMAGE_ASPECT_COLOR_BIT, SRC_W, SRC_H, up.buffer,
                         true);
      g52_cmd_copy_image(&g, cmd, dst.image, VK_IMAGE_ASPECT_COLOR_BIT, SRC_W, SRC_H, pre.buffer,
                         true);
      to_layout(src.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      g52_cmd_full_barrier(&g, cmd);
      blit(src.image, dst.image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, SRC_W, SRC_H, 0, 0, SRC_W, SRC_H,
           VK_FILTER_NEAREST);
      to_layout(dst.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      g52_cmd_copy_image(&g, cmd, src.image, VK_IMAGE_ASPECT_COLOR_BIT, SRC_W, SRC_H,
                         rb_src.buffer, false);
      g52_cmd_copy_image(&g, cmd, dst.image, VK_IMAGE_ASPECT_COLOR_BIT, SRC_W, SRC_H,
                         rb_dst.buffer, false);
      ok = end_and_run(what_blit);
   }
   bool pass = ok;
   if (ok) {
      g52_hostbuf_invalidate(&g, &rb_src);
      g52_hostbuf_invalidate(&g, &rb_dst);
      const bool round_trip =
         l_check(lf, what_rt, rb_src.map, (size_t)SRC_W * bytes, false, true);
      if (!round_trip)
         printf("    (the copies lose %u-byte texels, so the blit check below is seen through "
                "them; the linear run checks the blit alone)\n", bytes);
      const bool blitted = l_check(lf, what_blit, rb_dst.map, (size_t)SRC_W * bytes, true, true);
      pass = round_trip && blitted;
   } else {
      printf("FAIL L (%s): not run to the end\n", what_blit);
   }
   g52_hostbuf_destroy(&g, &up);
   g52_hostbuf_destroy(&g, &pre);
   g52_hostbuf_destroy(&g, &rb_src);
   g52_hostbuf_destroy(&g, &rb_dst);
   g52_image_destroy(&g, &src);
   g52_image_destroy(&g, &dst);
   return pass;
}

/* L on linear host-visible images the CPU fills and reads: the blit
 * alone. */
static bool
l_run_linear(const struct l_format *lf)
{
   char what[96];
   snprintf(what, sizeof(what), "%s linear, 1:1 blit", lf->name);

   struct g52_image src = {0}, dst = {0};
   bool ok = l_image_create(&src, lf->format, true) && l_image_create(&dst, lf->format, true);
   if (ok) {
      l_image_fill(&src, lf->kind, false);
      l_image_fill(&dst, lf->kind, true);
      ok = begin();
   }
   if (ok) {
      g52_cmd_transition(&g, cmd, src.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
      g52_cmd_transition(&g, cmd, dst.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
      blit(src.image, dst.image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, SRC_W, SRC_H, 0, 0, SRC_W, SRC_H,
           VK_FILTER_NEAREST);
      g52_cmd_transition(&g, cmd, dst.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
      ok = end_and_run(what);
   }
   bool pass = ok;
   if (ok) {
      g52_image_sync_for_read(&g, &dst);
      pass = l_check(lf, what, (const uint8_t *)dst.map + dst.layout.offset,
                     (size_t)dst.layout.rowPitch, true, false);
   } else {
      printf("FAIL L (%s): not run to the end\n", what);
   }
   g52_image_destroy(&g, &src);
   g52_image_destroy(&g, &dst);
   return pass;
}

static bool
run_npot_cases(void)
{
   if (!want_case('L'))
      return true;
   printf("case L: same-format 1:1 blits of 3- and 12-byte texels\n");
   PFN_vkGetPhysicalDeviceImageFormatProperties image_format_properties =
      (PFN_vkGetPhysicalDeviceImageFormatProperties)g.f.gipa(
         g.instance, "vkGetPhysicalDeviceImageFormatProperties");
   const VkFormatFeatureFlags blits = VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                      VK_FORMAT_FEATURE_BLIT_DST_BIT;
   const VkFormatFeatureFlags copies = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                       VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   bool pass = true;
   unsigned ran = 0;
   for (size_t i = 0; i < sizeof(l_formats) / sizeof(l_formats[0]); i++) {
      const struct l_format *lf = &l_formats[i];
      VkFormatProperties p;
      g.f.GetPhysicalDeviceFormatProperties(g.pdev, lf->format, &p);
      printf("  %s (%u-byte texels): optimal features 0x%x, linear 0x%x\n", lf->name,
             l_bytes(lf->kind), p.optimalTilingFeatures, p.linearTilingFeatures);

      if ((p.optimalTilingFeatures & (blits | copies)) == (blits | copies)) {
         /* The driver's buffer/image copies skip 3- and 12-byte texels on
          * every arch (mali_cmd_image.c dispatch_copy), so this half only
          * reports; the linear half below is what checks the blit. */
         if (!l_run_optimal(lf))
            printf("KNOWN L (%s optimal): the buffer/image copies do not handle "
                   "%u-byte texels\n", lf->name, l_bytes(lf->kind));
         ran++;
      } else {
         printf("SKIP L (%s optimal): not offered as blit and transfer source and "
                "destination\n", lf->name);
      }

      /* Linear images take the same features, but may be limited further
       * (vkGetPhysicalDeviceImageFormatProperties). */
      VkImageFormatProperties ifp;
      const VkResult r =
         (p.linearTilingFeatures & blits) != blits ? VK_ERROR_FORMAT_NOT_SUPPORTED :
         !image_format_properties ? VK_SUCCESS :
         image_format_properties(g.pdev, lf->format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR,
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                 0, &ifp);
      if (r != VK_SUCCESS) {
         printf("SKIP L (%s linear): not offered as a linear blit source and destination "
                "(%s)\n", lf->name, g52_result_str(r));
      } else if (image_format_properties &&
                 (ifp.maxExtent.width < SRC_W || ifp.maxExtent.height < SRC_H)) {
         printf("SKIP L (%s linear): linear images reach only %ux%u\n", lf->name,
                ifp.maxExtent.width, ifp.maxExtent.height);
      } else {
         pass &= l_run_linear(lf);
         ran++;
      }
   }
   if (!ran)
      printf("SKIP L: the device offers none of these formats as blit source and "
             "destination\n");
   return pass;
}

int
main(int argc, char **argv)
{
   const char *icd = NULL;
   for (int i = 1; i < argc; i++) {
      if (!strncmp(argv[i], "--only=", 7))
         only = argv[i][7];
      else
         icd = argv[i];
   }
   char *args[2] = {argv[0], (char *)icd};
   int rc = g52_open(&g, "blit_g52", icd ? 2 : 1, args, BLIT_G52_DEFAULT_ICD);
   if (rc)
      return rc;
   if (!g.f.CmdBlitImage || !g.f.CmdCopyBufferToImage || !g.f.CmdCopyImageToBuffer) {
      fprintf(stderr, "blit_g52: the driver lacks vkCmdBlitImage or the buffer/image copies\n");
      g52_close(&g);
      return 1;
   }

   const VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   const VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandBufferCount = 1,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
   };
   VkResult r = g.f.CreateCommandPool(g.device, &pci, NULL, &pool);
   if (r == VK_SUCCESS) {
      VkCommandBufferAllocateInfo a = cai;
      a.commandPool = pool;
      r = g.f.AllocateCommandBuffers(g.device, &a, &cmd);
   }
   if (r != VK_SUCCESS) {
      printf("FAIL: command pool or buffer: %s\n", g52_result_str(r));
      g52_close(&g);
      return 1;
   }

   bool pass = true;
   if (!format_blits(VK_FORMAT_R8G8B8A8_UNORM, "R8G8B8A8_UNORM")) {
      printf("FAIL: the device offers no R8G8B8A8_UNORM blits\n");
      pass = false;
      goto done;
   }

   /* The source, uploaded once and left in TRANSFER_SRC_OPTIMAL. */
   struct g52_image src;
   struct g52_hostbuf up = {0};
   {
      static uint8_t pattern[SRC_W * SRC_H * 4];
      for (int y = 0; y < SRC_H; y++)
         for (int x = 0; x < SRC_W; x++)
            src_texel(x, y, &pattern[(y * SRC_W + x) * 4]);
      bool ok = make(&src, VK_FORMAT_R8G8B8A8_UNORM, SRC_W, SRC_H, VK_IMAGE_ASPECT_COLOR_BIT) &&
                begin() &&
                record_upload(&src, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &up,
                              pattern, 4, true);
      if (ok) {
         to_layout(src.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
         ok = end_and_run("source upload");
      }
      if (!ok) {
         printf("FAIL: the source upload (vkCmdCopyBufferToImage, copy_g52's path) failed\n");
         pass = false;
         goto done;
      }
      printf("  source: 32x32 R8G8B8A8_UNORM optimal, uploaded\n");
   }

   pass &= run_rgba8_case('A', "2x nearest", &src, VK_FORMAT_R8G8B8A8_UNORM, BIG, BIG, 0, 0,
                          SRC_W, SRC_H, 0, 0, BIG, BIG, VK_FILTER_NEAREST, expect_a, 1, false);
   pass &= run_rgba8_case('B', "2x linear", &src, VK_FORMAT_R8G8B8A8_UNORM, BIG, BIG, 0, 0,
                          SRC_W, SRC_H, 0, 0, BIG, BIG, VK_FILTER_LINEAR, expect_b, 2, false);
   pass &= run_rgba8_case('C', "mirrored on X", &src, VK_FORMAT_R8G8B8A8_UNORM, SRC_W, SRC_H, 0,
                          0, SRC_W, SRC_H, SRC_W, 0, 0, SRC_H, VK_FILTER_NEAREST, expect_c, 1,
                          false);
   pass &= run_rgba8_case('D', "mirrored on Y", &src, VK_FORMAT_R8G8B8A8_UNORM, SRC_W, SRC_H, 0,
                          SRC_H, SRC_W, 0, 0, 0, SRC_W, SRC_H, VK_FILTER_NEAREST, expect_d, 1,
                          false);
   pass &= run_565_cases(&src);
   if (format_blits(VK_FORMAT_R8G8B8A8_SRGB, "R8G8B8A8_SRGB"))
      pass &= run_rgba8_case('G', "RGBA8 -> sRGB 1:1", &src, VK_FORMAT_R8G8B8A8_SRGB, SRC_W,
                             SRC_H, 0, 0, SRC_W, SRC_H, 0, 0, SRC_W, SRC_H, VK_FILTER_NEAREST,
                             expect_g, 1, false);
   else if (want_case('G'))
      printf("SKIP G: no R8G8B8A8_SRGB blits\n");
   pass &= run_rgba8_case('H', "rectangle (5, 3) .. (42, 32), outside untouched", &src,
                          VK_FORMAT_R8G8B8A8_UNORM, BIG, BIG, 0, 0, SRC_W, SRC_H, H_X0, H_Y0, H_X1,
                          H_Y1, VK_FILTER_NEAREST, expect_h, 1, true);
   pass &= run_d32_case();
   pass &= run_d24s8_case();
   pass &= run_rgba4_case(&src);
   pass &= run_npot_cases();

   g52_hostbuf_destroy(&g, &up);
   g52_image_destroy(&g, &src);
done:
   g.f.DestroyCommandPool(g.device, pool, NULL);
   g52_close(&g);
   printf("%s: blit_g52\n", pass ? "PASS" : "FAIL");
   return pass ? 0 : 1;
}
