/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * draw_g52: the first real draw on the Bifrost (v7) path of this driver.
 * After clear_g52 proved the fragment job on the RG DS kernel, this one
 * proves the Vertex + Tiler job pair, the Renderer State Descriptors, the
 * attribute, varying and viewport descriptors, the tiler heap and the
 * Bifrost compiler: a 64x64 linear RGBA8 image is cleared to black, one
 * full-screen triangle is drawn into it with a vertex shader that reads a
 * 2D position from a vertex buffer and a fragment shader that writes the
 * constant colour (0.5, 0.25, 1.0, 1.0), and every pixel is read back and
 * checked against that colour.
 *
 * The shaders are SPIR-V embedded as arrays, generated with glslangValidator
 * 16.6.0 (`glslangValidator -V --target-env vulkan1.1`) from:
 *
 *   tri.vert:  #version 450
 *              layout(location = 0) in vec2 pos;
 *              void main() { gl_Position = vec4(pos, 0.0, 1.0); }
 *
 *   tri.frag:  #version 450
 *              layout(location = 0) out vec4 color;
 *              void main() { color = vec4(0.5, 0.25, 1.0, 1.0); }
 *
 * A second case, when the device reports shaderClipDistance, draws the
 * same triangle with a vertex shader that also writes three clip
 * distances (the left half and the bottom quarter of the image are
 * clipped; the third distance is always 1.0):
 *
 *   clip.vert: #version 450
 *              layout(location = 0) in vec2 pos;
 *              out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[3]; };
 *              void main() {
 *                 gl_Position = vec4(pos, 0.0, 1.0);
 *                 gl_ClipDistance[0] = pos.x;
 *                 gl_ClipDistance[1] = pos.y + 0.5;
 *                 gl_ClipDistance[2] = 1.0;
 *              }
 *
 * and checks that exactly the pixels whose centre has x > 0 and y > -0.5
 * in normalized device coordinates hold the triangle colour, the others
 * the clear colour.
 *
 * The pipeline uses dynamic rendering, no descriptor sets, no push
 * constants, no depth/stencil, no blending (an opaque store), no dynamic
 * state (viewport and scissor are the whole image), no culling. The
 * triangle's vertices are (-1,-1), (3,-1), (-1,3): it covers the clip
 * space square, so every pixel of the image is drawn.
 *
 * No Vulkan loader: the driver library is dlopen'ed (see clear_g52.c).
 *
 * Usage: draw_g52 [path/to/libvulkan_malisx2.so]
 *   The path defaults to $MALISX2_ICD, then to DRAW_G52_DEFAULT_ICD.
 *
 * Exit status: 0 PASS; 1 FAIL (wrong pixels, a timeout or a Vulkan error
 * once the device exists, including a pipeline that would not compile); 2
 * the program did not get as far as a VkDevice.
 *
 * Driver log: MESA_LOG=file sends it to stderr, MESA_LOG_LEVEL=info adds
 * the atom layout and build lines (see MALISX2_G52.md).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* clock_gettime under -std=c11 */
#endif

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef DRAW_G52_DEFAULT_ICD
#define DRAW_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define IMG_W 64
#define IMG_H 64
#define WAIT_NS (5ull * 1000 * 1000 * 1000)

/* The triangle colour and its R8G8B8A8_UNORM bytes (127.5 -> 128 with
 * round to nearest even, 63.75 -> 64, 255, 255). One LSB of slack per
 * channel: the render target has dithering enabled, as panvk's. */
static const float tri_rgba[4] = {0.5f, 0.25f, 1.0f, 1.0f};
static const uint8_t expect_rgba[4] = {128, 64, 255, 255};
/* The clear colour under the triangle: black, so a pixel the triangle
 * missed is told apart from one never written (the sentinel). */
static const float clear_rgba[4] = {0.0f, 0.0f, 0.0f, 1.0f};
#define SENTINEL 0xa5

/* ---------------------------------------------------------------------- */
/* Shaders (see the file comment)                                          */

static const uint32_t vert_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x0000001b, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000000,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000012, 0x00030003,
   0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
   0x00060005, 0x0000000b, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000,
   0x00060006, 0x0000000b, 0x00000000, 0x505f6c67, 0x7469736f, 0x006e6f69,
   0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953,
   0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c,
   0x61747369, 0x0065636e, 0x00070006, 0x0000000b, 0x00000003, 0x435f6c67,
   0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x0000000d, 0x00000000,
   0x00030005, 0x00000012, 0x00736f70, 0x00030047, 0x0000000b, 0x00000002,
   0x00050048, 0x0000000b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048,
   0x0000000b, 0x00000001, 0x0000000b, 0x00000001, 0x00050048, 0x0000000b,
   0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x0000000b, 0x00000003,
   0x0000000b, 0x00000004, 0x00040047, 0x00000012, 0x0000001e, 0x00000000,
   0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
   0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
   0x00040015, 0x00000008, 0x00000020, 0x00000000, 0x0004002b, 0x00000008,
   0x00000009, 0x00000001, 0x0004001c, 0x0000000a, 0x00000006, 0x00000009,
   0x0006001e, 0x0000000b, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a,
   0x00040020, 0x0000000c, 0x00000003, 0x0000000b, 0x0004003b, 0x0000000c,
   0x0000000d, 0x00000003, 0x00040015, 0x0000000e, 0x00000020, 0x00000001,
   0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040017, 0x00000010,
   0x00000006, 0x00000002, 0x00040020, 0x00000011, 0x00000001, 0x00000010,
   0x0004003b, 0x00000011, 0x00000012, 0x00000001, 0x0004002b, 0x00000006,
   0x00000014, 0x00000000, 0x0004002b, 0x00000006, 0x00000015, 0x3f800000,
   0x00040020, 0x00000019, 0x00000003, 0x00000007, 0x00050036, 0x00000002,
   0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d,
   0x00000010, 0x00000013, 0x00000012, 0x00050051, 0x00000006, 0x00000016,
   0x00000013, 0x00000000, 0x00050051, 0x00000006, 0x00000017, 0x00000013,
   0x00000001, 0x00070050, 0x00000007, 0x00000018, 0x00000016, 0x00000017,
   0x00000014, 0x00000015, 0x00050041, 0x00000019, 0x0000001a, 0x0000000d,
   0x0000000f, 0x0003003e, 0x0000001a, 0x00000018, 0x000100fd, 0x00010038,
};

static const uint32_t vert_clip_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x0000002a, 0x00000000, 0x00020011,
   0x00000001, 0x00020011, 0x00000020, 0x0006000b, 0x00000001, 0x4c534c47,
   0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
   0x0007000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000d,
   0x00000012, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
   0x6e69616d, 0x00000000, 0x00060005, 0x0000000b, 0x505f6c67, 0x65567265,
   0x78657472, 0x00000000, 0x00060006, 0x0000000b, 0x00000000, 0x505f6c67,
   0x7469736f, 0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x435f6c67,
   0x4470696c, 0x61747369, 0x0065636e, 0x00030005, 0x0000000d, 0x00000000,
   0x00030005, 0x00000012, 0x00736f70, 0x00030047, 0x0000000b, 0x00000002,
   0x00050048, 0x0000000b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048,
   0x0000000b, 0x00000001, 0x0000000b, 0x00000003, 0x00040047, 0x00000012,
   0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
   0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007,
   0x00000006, 0x00000004, 0x00040015, 0x00000008, 0x00000020, 0x00000000,
   0x0004002b, 0x00000008, 0x00000009, 0x00000003, 0x0004001c, 0x0000000a,
   0x00000006, 0x00000009, 0x0004001e, 0x0000000b, 0x00000007, 0x0000000a,
   0x00040020, 0x0000000c, 0x00000003, 0x0000000b, 0x0004003b, 0x0000000c,
   0x0000000d, 0x00000003, 0x00040015, 0x0000000e, 0x00000020, 0x00000001,
   0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040017, 0x00000010,
   0x00000006, 0x00000002, 0x00040020, 0x00000011, 0x00000001, 0x00000010,
   0x0004003b, 0x00000011, 0x00000012, 0x00000001, 0x0004002b, 0x00000006,
   0x00000014, 0x00000000, 0x0004002b, 0x00000006, 0x00000015, 0x3f800000,
   0x00040020, 0x00000019, 0x00000003, 0x00000007, 0x0004002b, 0x0000000e,
   0x0000001b, 0x00000001, 0x0004002b, 0x00000008, 0x0000001c, 0x00000000,
   0x00040020, 0x0000001d, 0x00000001, 0x00000006, 0x00040020, 0x00000020,
   0x00000003, 0x00000006, 0x0004002b, 0x00000008, 0x00000022, 0x00000001,
   0x0004002b, 0x00000006, 0x00000025, 0x3f000000, 0x0004002b, 0x0000000e,
   0x00000028, 0x00000002, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
   0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000010, 0x00000013,
   0x00000012, 0x00050051, 0x00000006, 0x00000016, 0x00000013, 0x00000000,
   0x00050051, 0x00000006, 0x00000017, 0x00000013, 0x00000001, 0x00070050,
   0x00000007, 0x00000018, 0x00000016, 0x00000017, 0x00000014, 0x00000015,
   0x00050041, 0x00000019, 0x0000001a, 0x0000000d, 0x0000000f, 0x0003003e,
   0x0000001a, 0x00000018, 0x00050041, 0x0000001d, 0x0000001e, 0x00000012,
   0x0000001c, 0x0004003d, 0x00000006, 0x0000001f, 0x0000001e, 0x00060041,
   0x00000020, 0x00000021, 0x0000000d, 0x0000001b, 0x0000000f, 0x0003003e,
   0x00000021, 0x0000001f, 0x00050041, 0x0000001d, 0x00000023, 0x00000012,
   0x00000022, 0x0004003d, 0x00000006, 0x00000024, 0x00000023, 0x00050081,
   0x00000006, 0x00000026, 0x00000024, 0x00000025, 0x00060041, 0x00000020,
   0x00000027, 0x0000000d, 0x0000001b, 0x0000001b, 0x0003003e, 0x00000027,
   0x00000026, 0x00060041, 0x00000020, 0x00000029, 0x0000000d, 0x0000001b,
   0x00000028, 0x0003003e, 0x00000029, 0x00000015, 0x000100fd, 0x00010038,
};

static const uint32_t frag_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x0000000e, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
   0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
   0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63, 0x00000072,
   0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00020013, 0x00000002,
   0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
   0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008,
   0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003,
   0x0004002b, 0x00000006, 0x0000000a, 0x3f000000, 0x0004002b, 0x00000006,
   0x0000000b, 0x3e800000, 0x0004002b, 0x00000006, 0x0000000c, 0x3f800000,
   0x0007002c, 0x00000007, 0x0000000d, 0x0000000a, 0x0000000b, 0x0000000c,
   0x0000000c, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
   0x000200f8, 0x00000005, 0x0003003e, 0x00000009, 0x0000000d, 0x000100fd,
   0x00010038,
};

/* A triangle covering clip space: (-1,-1), (3,-1), (-1,3). */
static const float vertices[6] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};

/* ---------------------------------------------------------------------- */
/* Helpers                                                                 */

static const char *
result_str(VkResult r)
{
   switch (r) {
   case VK_SUCCESS: return "VK_SUCCESS";
   case VK_NOT_READY: return "VK_NOT_READY";
   case VK_TIMEOUT: return "VK_TIMEOUT";
   case VK_INCOMPLETE: return "VK_INCOMPLETE";
   case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
   case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
   case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
   case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
   case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
   case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
   case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
   case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
   case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
   case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
   case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
   case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
   default: return "VkResult(?)";
   }
}

static int64_t
now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

struct fns {
   PFN_vkGetInstanceProcAddr gipa;
   PFN_vkCreateInstance CreateInstance;
   PFN_vkDestroyInstance DestroyInstance;
   PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
   PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
   PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
   PFN_vkGetPhysicalDeviceFeatures GetPhysicalDeviceFeatures;
   PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
   PFN_vkCreateDevice CreateDevice;
   PFN_vkGetDeviceProcAddr GetDeviceProcAddr;

   PFN_vkDestroyDevice DestroyDevice;
   PFN_vkGetDeviceQueue GetDeviceQueue;
   PFN_vkCreateImage CreateImage;
   PFN_vkDestroyImage DestroyImage;
   PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
   PFN_vkGetImageSubresourceLayout GetImageSubresourceLayout;
   PFN_vkCreateBuffer CreateBuffer;
   PFN_vkDestroyBuffer DestroyBuffer;
   PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
   PFN_vkBindBufferMemory BindBufferMemory;
   PFN_vkAllocateMemory AllocateMemory;
   PFN_vkFreeMemory FreeMemory;
   PFN_vkBindImageMemory BindImageMemory;
   PFN_vkMapMemory MapMemory;
   PFN_vkUnmapMemory UnmapMemory;
   PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges;
   PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges;
   PFN_vkCreateImageView CreateImageView;
   PFN_vkDestroyImageView DestroyImageView;
   PFN_vkCreateShaderModule CreateShaderModule;
   PFN_vkDestroyShaderModule DestroyShaderModule;
   PFN_vkCreatePipelineLayout CreatePipelineLayout;
   PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
   PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines;
   PFN_vkDestroyPipeline DestroyPipeline;
   PFN_vkCreateCommandPool CreateCommandPool;
   PFN_vkDestroyCommandPool DestroyCommandPool;
   PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
   PFN_vkBeginCommandBuffer BeginCommandBuffer;
   PFN_vkEndCommandBuffer EndCommandBuffer;
   PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
   PFN_vkCmdBeginRendering CmdBeginRendering;
   PFN_vkCmdEndRendering CmdEndRendering;
   PFN_vkCmdBindPipeline CmdBindPipeline;
   PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers;
   PFN_vkCmdDraw CmdDraw;
   PFN_vkCreateFence CreateFence;
   PFN_vkDestroyFence DestroyFence;
   PFN_vkQueueSubmit QueueSubmit;
   PFN_vkWaitForFences WaitForFences;
   PFN_vkDeviceWaitIdle DeviceWaitIdle;
   PFN_vkGetDeviceFaultInfoEXT GetDeviceFaultInfoEXT; /* optional */
};

static PFN_vkVoidFunction
load_instance_fn(const struct fns *f, VkInstance instance, const char *name, bool *ok)
{
   PFN_vkVoidFunction p = f->gipa(instance, name);
   if (!p) {
      fprintf(stderr, "draw_g52: vk_icdGetInstanceProcAddr(%s) returned NULL\n", name);
      *ok = false;
   }
   return p;
}

static PFN_vkVoidFunction
load_device_fn(const struct fns *f, VkDevice device, const char *name, const char *alt,
               bool required, bool *ok)
{
   PFN_vkVoidFunction p = f->GetDeviceProcAddr(device, name);
   if (!p && alt)
      p = f->GetDeviceProcAddr(device, alt);
   if (!p && required) {
      fprintf(stderr, "draw_g52: vkGetDeviceProcAddr(%s%s%s) returned NULL\n", name,
              alt ? " / " : "", alt ? alt : "");
      *ok = false;
   }
   return p;
}

static void
print_fault(const struct fns *f, VkDevice device)
{
   if (!f->GetDeviceFaultInfoEXT) {
      printf("  (VK_EXT_device_fault not enabled; no fault record)\n");
      return;
   }
   VkDeviceFaultCountsEXT counts = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
   VkResult r = f->GetDeviceFaultInfoEXT(device, &counts, NULL);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      printf("  vkGetDeviceFaultInfoEXT: %s\n", result_str(r));
      return;
   }
   VkDeviceFaultAddressInfoEXT addr[4];
   VkDeviceFaultVendorInfoEXT vendor[4];
   if (counts.addressInfoCount > 4)
      counts.addressInfoCount = 4;
   if (counts.vendorInfoCount > 4)
      counts.vendorInfoCount = 4;
   counts.vendorBinarySize = 0;
   VkDeviceFaultInfoEXT info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
      .pAddressInfos = addr,
      .pVendorInfos = vendor,
   };
   r = f->GetDeviceFaultInfoEXT(device, &counts, &info);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      printf("  vkGetDeviceFaultInfoEXT: %s\n", result_str(r));
      return;
   }
   printf("  device fault: %s\n", info.description);
   for (uint32_t i = 0; i < counts.vendorInfoCount; i++)
      printf("  vendor record %u: %s (kbase event code 0x%llx, atom %llu)\n", i,
             vendor[i].description, (unsigned long long)vendor[i].vendorFaultCode,
             (unsigned long long)vendor[i].vendorFaultData);
   for (uint32_t i = 0; i < counts.addressInfoCount; i++)
      printf("  address record %u: type %d address 0x%llx\n", i, addr[i].addressType,
             (unsigned long long)addr[i].reportedAddress);
}

/* A host-visible memory type for `type_bits`, host-coherent if there is
 * one; -1 when there is none. */
static int
pick_memory_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t type_bits, bool *coherent)
{
   int type = -1;
   *coherent = false;
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      VkMemoryPropertyFlags fl = mp->memoryTypes[i].propertyFlags;
      if (!(type_bits & (1u << i)) || !(fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         continue;
      const bool c = fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      if (type < 0 || (c && !*coherent)) {
         type = (int)i;
         *coherent = c;
      }
   }
   return type;
}

/* ---------------------------------------------------------------------- */

int
main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IOLBF, 0);

   const char *path = argc > 1 ? argv[1] : getenv("MALISX2_ICD");
   if (!path || !*path)
      path = DRAW_G52_DEFAULT_ICD;

   printf("draw_g52: loading %s\n", path);
   void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
   if (!lib) {
      fprintf(stderr, "draw_g52: dlopen(%s) failed: %s\n", path, dlerror());
      return 2;
   }

   struct fns f;
   memset(&f, 0, sizeof(f));
   f.gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vk_icdGetInstanceProcAddr");
   if (!f.gipa) {
      fprintf(stderr, "draw_g52: %s exports no vk_icdGetInstanceProcAddr: %s\n", path,
              dlerror());
      return 2;
   }

   bool ok = true;
   f.CreateInstance = (PFN_vkCreateInstance)load_instance_fn(&f, NULL, "vkCreateInstance", &ok);
   if (!ok)
      return 2;

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "draw_g52",
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance = VK_NULL_HANDLE;
   VkResult r = f.CreateInstance(&ici, NULL, &instance);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkCreateInstance failed: %s\n", result_str(r));
      return 2;
   }

#define INST(name) f.name = (PFN_vk##name)load_instance_fn(&f, instance, "vk" #name, &ok)
   INST(DestroyInstance);
   INST(EnumeratePhysicalDevices);
   INST(GetPhysicalDeviceProperties2);
   INST(GetPhysicalDeviceMemoryProperties);
   INST(GetPhysicalDeviceFeatures);
   INST(EnumerateDeviceExtensionProperties);
   INST(CreateDevice);
   INST(GetDeviceProcAddr);
#undef INST
   if (!ok)
      return 2;

   uint32_t pdev_count = 0;
   r = f.EnumeratePhysicalDevices(instance, &pdev_count, NULL);
   if (r != VK_SUCCESS || pdev_count == 0) {
      fprintf(stderr, "draw_g52: no physical device (%s, count %u): the driver refused /dev/mali0; "
              "see the driver log\n", result_str(r), pdev_count);
      return 2;
   }
   VkPhysicalDevice pdevs[8];
   if (pdev_count > 8)
      pdev_count = 8;
   r = f.EnumeratePhysicalDevices(instance, &pdev_count, pdevs);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      fprintf(stderr, "draw_g52: vkEnumeratePhysicalDevices failed: %s\n", result_str(r));
      return 2;
   }
   VkPhysicalDevice pdev = pdevs[0];

   VkPhysicalDeviceDriverProperties drv = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &drv,
   };
   f.GetPhysicalDeviceProperties2(pdev, &props2);
   const VkPhysicalDeviceProperties *p = &props2.properties;
   printf("  device: %s, apiVersion %u.%u.%u, driverInfo \"%s\"\n", p->deviceName,
          VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion),
          VK_API_VERSION_PATCH(p->apiVersion), drv.driverInfo);

   VkPhysicalDeviceMemoryProperties mem_props;
   f.GetPhysicalDeviceMemoryProperties(pdev, &mem_props);

   bool have_fault_ext = false;
   {
      uint32_t n = 0;
      if (f.EnumerateDeviceExtensionProperties(pdev, NULL, &n, NULL) == VK_SUCCESS && n) {
         VkExtensionProperties *ext = calloc(n, sizeof(*ext));
         if (ext && f.EnumerateDeviceExtensionProperties(pdev, NULL, &n, ext) == VK_SUCCESS) {
            for (uint32_t i = 0; i < n; i++)
               if (!strcmp(ext[i].extensionName, VK_EXT_DEVICE_FAULT_EXTENSION_NAME))
                  have_fault_ext = true;
         }
         free(ext);
      }
   }

   const float prio = 1.0f;
   const VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceFaultFeaturesEXT fault_feat = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT,
      .deviceFault = VK_TRUE,
   };
   const char *dev_exts[] = {VK_EXT_DEVICE_FAULT_EXTENSION_NAME};
   VkPhysicalDeviceFeatures supported;
   memset(&supported, 0, sizeof(supported));
   f.GetPhysicalDeviceFeatures(pdev, &supported);
   const bool have_clip = supported.shaderClipDistance;
   const VkPhysicalDeviceFeatures enabled = {.shaderClipDistance = have_clip};
   const VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = have_fault_ext ? &fault_feat : NULL,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &enabled,
      .enabledExtensionCount = have_fault_ext ? 1 : 0,
      .ppEnabledExtensionNames = dev_exts,
   };
   VkDevice device = VK_NULL_HANDLE;
   r = f.CreateDevice(pdev, &dci, NULL, &device);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkCreateDevice failed: %s (see the log)\n", result_str(r));
      return 2;
   }
   printf("  VkDevice created%s\n", have_fault_ext ? " (VK_EXT_device_fault enabled)" : "");

#define DEV(name) \
   f.name = (PFN_vk##name)load_device_fn(&f, device, "vk" #name, NULL, true, &ok)
   DEV(DestroyDevice);
   DEV(GetDeviceQueue);
   DEV(CreateImage);
   DEV(DestroyImage);
   DEV(GetImageMemoryRequirements);
   DEV(CreateBuffer);
   DEV(DestroyBuffer);
   DEV(GetBufferMemoryRequirements);
   DEV(BindBufferMemory);
   DEV(AllocateMemory);
   DEV(FreeMemory);
   DEV(BindImageMemory);
   DEV(MapMemory);
   DEV(UnmapMemory);
   DEV(FlushMappedMemoryRanges);
   DEV(InvalidateMappedMemoryRanges);
   DEV(CreateImageView);
   DEV(DestroyImageView);
   DEV(CreateShaderModule);
   DEV(DestroyShaderModule);
   DEV(CreatePipelineLayout);
   DEV(DestroyPipelineLayout);
   DEV(CreateGraphicsPipelines);
   DEV(DestroyPipeline);
   DEV(CreateCommandPool);
   DEV(DestroyCommandPool);
   DEV(AllocateCommandBuffers);
   DEV(BeginCommandBuffer);
   DEV(EndCommandBuffer);
   DEV(CmdPipelineBarrier);
   DEV(CmdBindPipeline);
   DEV(CmdBindVertexBuffers);
   DEV(CmdDraw);
   DEV(CreateFence);
   DEV(DestroyFence);
   DEV(QueueSubmit);
   DEV(WaitForFences);
   DEV(DeviceWaitIdle);
#undef DEV
   f.CmdBeginRendering = (PFN_vkCmdBeginRendering)load_device_fn(
      &f, device, "vkCmdBeginRendering", "vkCmdBeginRenderingKHR", true, &ok);
   f.CmdEndRendering = (PFN_vkCmdEndRendering)load_device_fn(
      &f, device, "vkCmdEndRendering", "vkCmdEndRenderingKHR", true, &ok);
   f.GetImageSubresourceLayout = (PFN_vkGetImageSubresourceLayout)load_device_fn(
      &f, device, "vkGetImageSubresourceLayout", NULL, false, &ok);
   if (have_fault_ext)
      f.GetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)load_device_fn(
         &f, device, "vkGetDeviceFaultInfoEXT", NULL, false, &ok);
   if (!ok)
      return 1;

   VkQueue queue = VK_NULL_HANDLE;
   f.GetDeviceQueue(device, 0, 0, &queue);

   /* The image: linear, colour attachment only, host-visible memory. */
   const VkImageCreateInfo ici_img = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {IMG_W, IMG_H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image = VK_NULL_HANDLE;
   r = f.CreateImage(device, &ici_img, NULL, &image);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkCreateImage failed: %s\n", result_str(r));
      return 1;
   }
   VkMemoryRequirements reqs;
   f.GetImageMemoryRequirements(device, image, &reqs);
   bool coherent = false;
   int type = pick_memory_type(&mem_props, reqs.memoryTypeBits, &coherent);
   if (type < 0) {
      fprintf(stderr, "draw_g52: no host-visible memory type for the image\n");
      return 1;
   }
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)type,
   };
   VkDeviceMemory memory = VK_NULL_HANDLE;
   r = f.AllocateMemory(device, &mai, NULL, &memory);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkAllocateMemory (image) failed: %s\n", result_str(r));
      return 1;
   }
   r = f.BindImageMemory(device, image, memory, 0);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkBindImageMemory failed: %s\n", result_str(r));
      return 1;
   }
   void *map = NULL;
   r = f.MapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &map);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkMapMemory (image) failed: %s\n", result_str(r));
      return 1;
   }
   memset(map, SENTINEL, (size_t)reqs.size);
   const VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = memory,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
   };
   if (!coherent)
      f.FlushMappedMemoryRanges(device, 1, &whole);

   VkSubresourceLayout layout = {.rowPitch = IMG_W * 4, .offset = 0};
   if (f.GetImageSubresourceLayout) {
      const VkImageSubresource sub = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT};
      f.GetImageSubresourceLayout(device, image, &sub, &layout);
   }
   printf("  image: %ux%u R8G8B8A8_UNORM linear, %llu bytes, row pitch %llu, memory type %d\n",
          IMG_W, IMG_H, (unsigned long long)reqs.size, (unsigned long long)layout.rowPitch, type);

   const VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkImageView view = VK_NULL_HANDLE;
   r = f.CreateImageView(device, &vci, NULL, &view);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkCreateImageView failed: %s\n", result_str(r));
      return 1;
   }

   /* The vertex buffer: three vec2 in host-visible memory. */
   const VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(vertices),
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer vbuf = VK_NULL_HANDLE;
   r = f.CreateBuffer(device, &bci, NULL, &vbuf);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkCreateBuffer failed: %s\n", result_str(r));
      return 1;
   }
   VkMemoryRequirements breqs;
   f.GetBufferMemoryRequirements(device, vbuf, &breqs);
   bool bcoherent = false;
   int btype = pick_memory_type(&mem_props, breqs.memoryTypeBits, &bcoherent);
   if (btype < 0) {
      fprintf(stderr, "draw_g52: no host-visible memory type for the vertex buffer\n");
      return 1;
   }
   const VkMemoryAllocateInfo bmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = breqs.size,
      .memoryTypeIndex = (uint32_t)btype,
   };
   VkDeviceMemory bmemory = VK_NULL_HANDLE;
   r = f.AllocateMemory(device, &bmai, NULL, &bmemory);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkAllocateMemory (vertex buffer) failed: %s\n", result_str(r));
      return 1;
   }
   r = f.BindBufferMemory(device, vbuf, bmemory, 0);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkBindBufferMemory failed: %s\n", result_str(r));
      return 1;
   }
   void *bmap = NULL;
   r = f.MapMemory(device, bmemory, 0, VK_WHOLE_SIZE, 0, &bmap);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkMapMemory (vertex buffer) failed: %s\n", result_str(r));
      return 1;
   }
   memcpy(bmap, vertices, sizeof(vertices));
   if (!bcoherent) {
      const VkMappedMemoryRange bwhole = {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = bmemory,
         .offset = 0,
         .size = VK_WHOLE_SIZE,
      };
      f.FlushMappedMemoryRanges(device, 1, &bwhole);
   }
   f.UnmapMemory(device, bmemory);
   printf("  vertex buffer: %zu bytes, memory type %d\n", sizeof(vertices), btype);

   int status = 0;
   for (unsigned clip = 0; clip < (have_clip ? 2u : 1u); clip++) {
   printf("  case %u: %s\n", clip, clip ? "three clip distances" : "the plain triangle");

   /* Shaders and the pipeline. */
   const VkShaderModuleCreateInfo vsmci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = clip ? sizeof(vert_clip_spv) : sizeof(vert_spv),
      .pCode = clip ? vert_clip_spv : vert_spv,
   };
   const VkShaderModuleCreateInfo fsmci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(frag_spv),
      .pCode = frag_spv,
   };
   VkShaderModule vsm = VK_NULL_HANDLE, fsm = VK_NULL_HANDLE;
   r = f.CreateShaderModule(device, &vsmci, NULL, &vsm);
   if (r == VK_SUCCESS)
      r = f.CreateShaderModule(device, &fsmci, NULL, &fsm);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkCreateShaderModule failed: %s\n", result_str(r));
      return 1;
   }
   const VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pl = VK_NULL_HANDLE;
   r = f.CreatePipelineLayout(device, &plci, NULL, &pl);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkCreatePipelineLayout failed: %s\n", result_str(r));
      return 1;
   }

   const VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vsm,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fsm,
         .pName = "main",
      },
   };
   const VkVertexInputBindingDescription vb_desc = {
      .binding = 0,
      .stride = 2 * sizeof(float),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription va_desc = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
      .offset = 0,
   };
   const VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vb_desc,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &va_desc,
   };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {0.0f, 0.0f, (float)IMG_W, (float)IMG_H, 0.0f, 1.0f};
   const VkRect2D scissor = {{0, 0}, {IMG_W, IMG_H}};
   const VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = VK_FALSE,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &cba,
   };
   const VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
   const VkPipelineRenderingCreateInfo prci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &color_format,
   };
   const VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &prci,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pColorBlendState = &cb,
      .layout = pl,
   };
   VkPipeline pipeline = VK_NULL_HANDLE;
   const int64_t tc0 = now_ns();
   r = f.CreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, NULL, &pipeline);
   const int64_t tc1 = now_ns();
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkCreateGraphicsPipelines failed: %s (the Bifrost compile or "
              "a TODO v7 refusal; see the log)\n", result_str(r));
      return 1;
   }
   printf("  pipeline created in %.2f ms (both shaders through the Bifrost compiler)\n",
          (double)(tc1 - tc0) / 1e6);

   const VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkCommandPool pool = VK_NULL_HANDLE;
   r = f.CreateCommandPool(device, &cpci, NULL, &pool);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkCreateCommandPool failed: %s\n", result_str(r));
      return 1;
   }
   const VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd = VK_NULL_HANDLE;
   r = f.AllocateCommandBuffers(device, &cbai, &cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkAllocateCommandBuffers failed: %s\n", result_str(r));
      return 1;
   }

   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   r = f.BeginCommandBuffer(cmd, &bi);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkBeginCommandBuffer failed: %s\n", result_str(r));
      return 1;
   }

   VkImageMemoryBarrier to_attachment = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1,
                        &to_attachment);

   const VkRenderingAttachmentInfo color = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .resolveMode = VK_RESOLVE_MODE_NONE,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = {clear_rgba[0], clear_rgba[1], clear_rgba[2], clear_rgba[3]},
   };
   const VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {IMG_W, IMG_H}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color,
   };
   f.CmdBeginRendering(cmd, &ri);
   f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   const VkDeviceSize vb_offset = 0;
   f.CmdBindVertexBuffers(cmd, 0, 1, &vbuf, &vb_offset);
   f.CmdDraw(cmd, 3, 1, 0, 0);
   f.CmdEndRendering(cmd);

   VkImageMemoryBarrier to_host = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 0, NULL, 1, &to_host);

   r = f.EndCommandBuffer(cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkEndCommandBuffer failed: %s (recording failed: the draw's "
              "descriptors, a TODO v7 stub or command memory; see the log)\n", result_str(r));
      return 1;
   }
   printf("  command buffer recorded (clear, one draw of 3 vertices)\n");

   const VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence = VK_NULL_HANDLE;
   r = f.CreateFence(device, &fci, NULL, &fence);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkCreateFence failed: %s\n", result_str(r));
      return 1;
   }
   const VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   };
   const int64_t t0 = now_ns();
   r = f.QueueSubmit(queue, 1, &si, fence);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "draw_g52: vkQueueSubmit failed: %s (kbase JOB_SUBMIT; see the log)\n",
              result_str(r));
      if (r == VK_ERROR_DEVICE_LOST)
         print_fault(&f, device);
      return 1;
   }
   printf("  submitted (one vtc atom: vertex + tiler jobs; one fragment atom); waiting up to "
          "%llu ms\n", (unsigned long long)(WAIT_NS / 1000000));

   r = f.WaitForFences(device, 1, &fence, VK_TRUE, WAIT_NS);
   const int64_t t1 = now_ns();
   printf("  vkWaitForFences: %s after %.3f ms\n", result_str(r), (double)(t1 - t0) / 1e6);
   if (r == VK_TIMEOUT) {
      printf("FAIL: the fence never signalled: an atom did not complete (a hung vertex or "
             "tiler job, a tiler heap that could not grow, or its event was never read). "
             "Check dmesg for mali/kbase lines.\n");
      return 1;
   }
   if (r != VK_SUCCESS) {
      printf("FAIL: vkWaitForFences returned %s\n", result_str(r));
      if (r == VK_ERROR_DEVICE_LOST)
         print_fault(&f, device);
      return 1;
   }

   if (!coherent)
      f.InvalidateMappedMemoryRanges(device, 1, &whole);

   const uint8_t *base = (const uint8_t *)map + layout.offset;
   uint32_t bad = 0, untouched = 0, black = 0;
   int first_x = -1, first_y = -1;
   uint8_t first_got[4] = {0};
   static const uint8_t clear_bytes[4] = {0, 0, 0, 255};
   uint32_t expect_drawn = 0;
   for (uint32_t y = 0; y < IMG_H; y++) {
      const uint8_t *row = base + (size_t)y * layout.rowPitch;
      for (uint32_t x = 0; x < IMG_W; x++) {
         const uint8_t *px = row + (size_t)x * 4;
         /* Clip case: the pixel centre's NDC x > 0 and y > -0.5. */
         const float ndc_x = (x + 0.5f) * 2.0f / IMG_W - 1.0f;
         const float ndc_y = (y + 0.5f) * 2.0f / IMG_H - 1.0f;
         const bool drawn = !clip || (ndc_x > 0.0f && ndc_y > -0.5f);
         const uint8_t *want = drawn ? expect_rgba : clear_bytes;
         expect_drawn += drawn;
         bool okpx = true;
         for (unsigned c = 0; c < 4; c++) {
            const int d = (int)px[c] - (int)want[c];
            if (d < -1 || d > 1)
               okpx = false;
         }
         if (okpx)
            continue;
         if (px[0] == SENTINEL && px[1] == SENTINEL && px[2] == SENTINEL && px[3] == SENTINEL)
            untouched++;
         else if (px[0] == 0 && px[1] == 0 && px[2] == 0 && px[3] == 255)
            black++;
         if (!bad) {
            first_x = (int)x;
            first_y = (int)y;
            memcpy(first_got, px, 4);
         }
         bad++;
      }
   }
   printf("  pixel (0,0) = %u %u %u %u, pixel (%u,%u) = %u %u %u %u, expected %u %u %u %u\n",
          base[0], base[1], base[2], base[3], IMG_W - 1, IMG_H - 1,
          base[(IMG_H - 1) * layout.rowPitch + (IMG_W - 1) * 4],
          base[(IMG_H - 1) * layout.rowPitch + (IMG_W - 1) * 4 + 1],
          base[(IMG_H - 1) * layout.rowPitch + (IMG_W - 1) * 4 + 2],
          base[(IMG_H - 1) * layout.rowPitch + (IMG_W - 1) * 4 + 3], expect_rgba[0],
          expect_rgba[1], expect_rgba[2], expect_rgba[3]);

   if (bad == 0) {
      printf("  case %u ok: %u pixels the triangle colour (%.2f, %.2f, %.2f, %.2f), %u the "
             "clear colour\n",
             clip, expect_drawn, tri_rgba[0], tri_rgba[1], tri_rgba[2], tri_rgba[3],
             IMG_W * IMG_H - expect_drawn);
   } else {
      printf("FAIL: case %u: %u of %u pixels wrong (%u black from the clear, %u still hold the "
             "0x%02x sentinel); first at (%d,%d) = %u %u %u %u\n",
             clip, bad, IMG_W * IMG_H, black, untouched, SENTINEL, first_x, first_y,
             first_got[0], first_got[1], first_got[2], first_got[3]);
      status = 1;
   }

   f.DeviceWaitIdle(device);
   f.DestroyFence(device, fence, NULL);
   f.DestroyCommandPool(device, pool, NULL);
   f.DestroyPipeline(device, pipeline, NULL);
   f.DestroyPipelineLayout(device, pl, NULL);
   f.DestroyShaderModule(device, vsm, NULL);
   f.DestroyShaderModule(device, fsm, NULL);
   }
   if (status == 0)
      printf("PASS: %s\n", have_clip ? "the plain triangle and the clip distance case"
                                      : "the plain triangle (no shaderClipDistance)");

   f.UnmapMemory(device, memory);
   f.DestroyBuffer(device, vbuf, NULL);
   f.FreeMemory(device, bmemory, NULL);
   f.DestroyImageView(device, view, NULL);
   f.DestroyImage(device, image, NULL);
   f.FreeMemory(device, memory, NULL);
   f.DestroyDevice(device, NULL);
   f.DestroyInstance(instance, NULL);
   dlclose(lib);
   return status;
}
