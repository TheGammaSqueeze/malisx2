/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * compute_g52: the first compute dispatch on the Bifrost (v7) path of this
 * driver. After draw_g52 proved the Vertex + Tiler job pair, this one
 * proves the Compute job, the Bifrost descriptor tables (storage buffers
 * read from the set through the desc.sets system values, a storage image
 * through the attribute unit, filled by the descriptor copy job), push
 * constants in FAU and the compute system values. Two dispatches of the
 * same shader in one command buffer:
 *
 *  - vkCmdDispatch, 4 workgroups of 64: reads words 0..255 of a storage
 *    buffer, writes each value times pc.mul (3) plus the global invocation
 *    id plus pc.add (0x00c0ffee) into a second storage buffer, and stores a
 *    gradient into a 16x16 RGBA8 storage image;
 *  - vkCmdDispatchBase with base workgroup 4, 4 workgroups, after new push
 *    constants (pc.add 0x0badf00d): words 256..511, no image stores.
 *
 * pc.add sits in the upper half of a 64-bit FAU word and the base
 * workgroup in the system values, so a dropped high half, stale push
 * constants or an ignored base all show up as wrong words. Both buffers and
 * the image are read back and checked.
 *
 * The shader is SPIR-V embedded as an array, generated with glslangValidator
 * 16.6.0 (`glslangValidator -V --target-env vulkan1.1`) from:
 *
 *   #version 450
 *   layout(local_size_x = 64) in;
 *   layout(set = 0, binding = 0) readonly buffer In { uint v[]; } src;
 *   layout(set = 0, binding = 1) writeonly buffer Out { uint v[]; } dst;
 *   layout(set = 0, binding = 2, rgba8) uniform writeonly image2D img;
 *   layout(push_constant) uniform PC { uint mul; uint add; } pc;
 *   void main() {
 *      uint i = gl_GlobalInvocationID.x;
 *      dst.v[i] = src.v[i] * pc.mul + i + pc.add;
 *      if (i < 256u) {
 *         ivec2 p = ivec2(int(i & 15u), int(i >> 4u));
 *         imageStore(img, p, vec4(float(p.x) / 15.0, float(p.y) / 15.0, 0.5, 1.0));
 *      }
 *   }
 *
 * The image is created with linear tiling when the driver allows storage
 * use on it, else optimal (u-interleaved) and read back by detiling the
 * single 16x16 tile on the CPU (the layout mali_image_layout.c gives a
 * 16x16 RGBA8 image: one tile, 1024 bytes, texel order of Mesa's
 * pan_tiling.c).
 *
 * No Vulkan loader: the driver library is dlopen'ed (see clear_g52.c).
 *
 * Usage: compute_g52 [path/to/libvulkan_malisx2.so]
 *   The path defaults to $MALISX2_ICD, then to COMPUTE_G52_DEFAULT_ICD.
 *
 * Exit status: 0 PASS; 1 FAIL (wrong values, a timeout or a Vulkan error
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

#ifndef COMPUTE_G52_DEFAULT_ICD
#define COMPUTE_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define N 256             /* invocations of one dispatch */
#define WORDS (2 * N)     /* the words both dispatches write */
#define WG 64
#define IMG_W 16
#define IMG_H 16
#define WAIT_NS (5ull * 1000 * 1000 * 1000)
#define PC_MUL 3u
#define PC_ADD 0x00c0ffeeu  /* vkCmdDispatch, words 0..N-1 */
#define PC_ADD2 0x0badf00du /* vkCmdDispatchBase, words N..WORDS-1 */

static const uint32_t comp_spv[] = {
   0x07230203, 0x00010300, 0x0008000b, 0x00000056, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000005,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00060010, 0x00000004,
   0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002,
   0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005,
   0x00000008, 0x00000069, 0x00080005, 0x0000000b, 0x475f6c67, 0x61626f6c,
   0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00030005, 0x00000011,
   0x0074754f, 0x00040006, 0x00000011, 0x00000000, 0x00000076, 0x00030005,
   0x00000013, 0x00747364, 0x00030005, 0x00000018, 0x00006e49, 0x00040006,
   0x00000018, 0x00000000, 0x00000076, 0x00030005, 0x0000001a, 0x00637273,
   0x00030005, 0x0000001f, 0x00004350, 0x00040006, 0x0000001f, 0x00000000,
   0x006c756d, 0x00040006, 0x0000001f, 0x00000001, 0x00646461, 0x00030005,
   0x00000021, 0x00006370, 0x00030005, 0x00000035, 0x00000070, 0x00030005,
   0x00000042, 0x00676d69, 0x00040047, 0x0000000b, 0x0000000b, 0x0000001c,
   0x00040047, 0x00000010, 0x00000006, 0x00000004, 0x00030047, 0x00000011,
   0x00000002, 0x00040048, 0x00000011, 0x00000000, 0x00000019, 0x00050048,
   0x00000011, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000013,
   0x00000019, 0x00040047, 0x00000013, 0x00000021, 0x00000001, 0x00040047,
   0x00000013, 0x00000022, 0x00000000, 0x00040047, 0x00000017, 0x00000006,
   0x00000004, 0x00030047, 0x00000018, 0x00000002, 0x00040048, 0x00000018,
   0x00000000, 0x00000018, 0x00050048, 0x00000018, 0x00000000, 0x00000023,
   0x00000000, 0x00030047, 0x0000001a, 0x00000018, 0x00040047, 0x0000001a,
   0x00000021, 0x00000000, 0x00040047, 0x0000001a, 0x00000022, 0x00000000,
   0x00030047, 0x0000001f, 0x00000002, 0x00050048, 0x0000001f, 0x00000000,
   0x00000023, 0x00000000, 0x00050048, 0x0000001f, 0x00000001, 0x00000023,
   0x00000004, 0x00030047, 0x00000042, 0x00000019, 0x00040047, 0x00000042,
   0x00000021, 0x00000002, 0x00040047, 0x00000042, 0x00000022, 0x00000000,
   0x00040047, 0x00000055, 0x0000000b, 0x00000019, 0x00020013, 0x00000002,
   0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
   0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040017,
   0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001,
   0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b,
   0x00000006, 0x0000000c, 0x00000000, 0x00040020, 0x0000000d, 0x00000001,
   0x00000006, 0x0003001d, 0x00000010, 0x00000006, 0x0003001e, 0x00000011,
   0x00000010, 0x00040020, 0x00000012, 0x0000000c, 0x00000011, 0x0004003b,
   0x00000012, 0x00000013, 0x0000000c, 0x00040015, 0x00000014, 0x00000020,
   0x00000001, 0x0004002b, 0x00000014, 0x00000015, 0x00000000, 0x0003001d,
   0x00000017, 0x00000006, 0x0003001e, 0x00000018, 0x00000017, 0x00040020,
   0x00000019, 0x0000000c, 0x00000018, 0x0004003b, 0x00000019, 0x0000001a,
   0x0000000c, 0x00040020, 0x0000001c, 0x0000000c, 0x00000006, 0x0004001e,
   0x0000001f, 0x00000006, 0x00000006, 0x00040020, 0x00000020, 0x00000009,
   0x0000001f, 0x0004003b, 0x00000020, 0x00000021, 0x00000009, 0x00040020,
   0x00000022, 0x00000009, 0x00000006, 0x0004002b, 0x00000014, 0x00000028,
   0x00000001, 0x0004002b, 0x00000006, 0x0000002e, 0x00000100, 0x00020014,
   0x0000002f, 0x00040017, 0x00000033, 0x00000014, 0x00000002, 0x00040020,
   0x00000034, 0x00000007, 0x00000033, 0x0004002b, 0x00000006, 0x00000037,
   0x0000000f, 0x0004002b, 0x00000006, 0x0000003b, 0x00000004, 0x00030016,
   0x0000003f, 0x00000020, 0x00090019, 0x00000040, 0x0000003f, 0x00000001,
   0x00000000, 0x00000000, 0x00000000, 0x00000002, 0x00000004, 0x00040020,
   0x00000041, 0x00000000, 0x00000040, 0x0004003b, 0x00000041, 0x00000042,
   0x00000000, 0x00040020, 0x00000045, 0x00000007, 0x00000014, 0x0004002b,
   0x0000003f, 0x00000049, 0x41700000, 0x0004002b, 0x00000006, 0x0000004b,
   0x00000001, 0x0004002b, 0x0000003f, 0x00000050, 0x3f000000, 0x0004002b,
   0x0000003f, 0x00000051, 0x3f800000, 0x00040017, 0x00000052, 0x0000003f,
   0x00000004, 0x0004002b, 0x00000006, 0x00000054, 0x00000040, 0x0006002c,
   0x00000009, 0x00000055, 0x00000054, 0x0000004b, 0x0000004b, 0x00050036,
   0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
   0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x0004003b, 0x00000034,
   0x00000035, 0x00000007, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b,
   0x0000000c, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000e, 0x0003003e,
   0x00000008, 0x0000000f, 0x0004003d, 0x00000006, 0x00000016, 0x00000008,
   0x0004003d, 0x00000006, 0x0000001b, 0x00000008, 0x00060041, 0x0000001c,
   0x0000001d, 0x0000001a, 0x00000015, 0x0000001b, 0x0004003d, 0x00000006,
   0x0000001e, 0x0000001d, 0x00050041, 0x00000022, 0x00000023, 0x00000021,
   0x00000015, 0x0004003d, 0x00000006, 0x00000024, 0x00000023, 0x00050084,
   0x00000006, 0x00000025, 0x0000001e, 0x00000024, 0x0004003d, 0x00000006,
   0x00000026, 0x00000008, 0x00050080, 0x00000006, 0x00000027, 0x00000025,
   0x00000026, 0x00050041, 0x00000022, 0x00000029, 0x00000021, 0x00000028,
   0x0004003d, 0x00000006, 0x0000002a, 0x00000029, 0x00050080, 0x00000006,
   0x0000002b, 0x00000027, 0x0000002a, 0x00060041, 0x0000001c, 0x0000002c,
   0x00000013, 0x00000015, 0x00000016, 0x0003003e, 0x0000002c, 0x0000002b,
   0x0004003d, 0x00000006, 0x0000002d, 0x00000008, 0x000500b0, 0x0000002f,
   0x00000030, 0x0000002d, 0x0000002e, 0x000300f7, 0x00000032, 0x00000000,
   0x000400fa, 0x00000030, 0x00000031, 0x00000032, 0x000200f8, 0x00000031,
   0x0004003d, 0x00000006, 0x00000036, 0x00000008, 0x000500c7, 0x00000006,
   0x00000038, 0x00000036, 0x00000037, 0x0004007c, 0x00000014, 0x00000039,
   0x00000038, 0x0004003d, 0x00000006, 0x0000003a, 0x00000008, 0x000500c2,
   0x00000006, 0x0000003c, 0x0000003a, 0x0000003b, 0x0004007c, 0x00000014,
   0x0000003d, 0x0000003c, 0x00050050, 0x00000033, 0x0000003e, 0x00000039,
   0x0000003d, 0x0003003e, 0x00000035, 0x0000003e, 0x0004003d, 0x00000040,
   0x00000043, 0x00000042, 0x0004003d, 0x00000033, 0x00000044, 0x00000035,
   0x00050041, 0x00000045, 0x00000046, 0x00000035, 0x0000000c, 0x0004003d,
   0x00000014, 0x00000047, 0x00000046, 0x0004006f, 0x0000003f, 0x00000048,
   0x00000047, 0x00050088, 0x0000003f, 0x0000004a, 0x00000048, 0x00000049,
   0x00050041, 0x00000045, 0x0000004c, 0x00000035, 0x0000004b, 0x0004003d,
   0x00000014, 0x0000004d, 0x0000004c, 0x0004006f, 0x0000003f, 0x0000004e,
   0x0000004d, 0x00050088, 0x0000003f, 0x0000004f, 0x0000004e, 0x00000049,
   0x00070050, 0x00000052, 0x00000053, 0x0000004a, 0x0000004f, 0x00000050,
   0x00000051, 0x00040063, 0x00000043, 0x00000044, 0x00000053, 0x000200f9,
   0x00000032, 0x000200f8, 0x00000032, 0x000100fd, 0x00010038,
};

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
   case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
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

/* What the shader writes to output word i: the first dispatch's words use
 * pc.add PC_ADD, the vkCmdDispatchBase ones PC_ADD2 (u32 arithmetic, as on
 * the GPU). */
static uint32_t
word_want(const uint32_t *in_words, uint32_t i)
{
   return in_words[i] * PC_MUL + i + (i < N ? PC_ADD : PC_ADD2);
}

struct fns {
   PFN_vkGetInstanceProcAddr gipa;
   PFN_vkCreateInstance CreateInstance;
   PFN_vkDestroyInstance DestroyInstance;
   PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
   PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
   PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
   PFN_vkGetPhysicalDeviceImageFormatProperties GetPhysicalDeviceImageFormatProperties;
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
   PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
   PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
   PFN_vkCreateDescriptorPool CreateDescriptorPool;
   PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
   PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
   PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
   PFN_vkCreatePipelineLayout CreatePipelineLayout;
   PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
   PFN_vkCreateComputePipelines CreateComputePipelines;
   PFN_vkDestroyPipeline DestroyPipeline;
   PFN_vkCreateCommandPool CreateCommandPool;
   PFN_vkDestroyCommandPool DestroyCommandPool;
   PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
   PFN_vkBeginCommandBuffer BeginCommandBuffer;
   PFN_vkEndCommandBuffer EndCommandBuffer;
   PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
   PFN_vkCmdBindPipeline CmdBindPipeline;
   PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
   PFN_vkCmdPushConstants CmdPushConstants;
   PFN_vkCmdDispatch CmdDispatch;
   PFN_vkCmdDispatchBase CmdDispatchBase;
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
      fprintf(stderr, "compute_g52: vk_icdGetInstanceProcAddr(%s) returned NULL\n", name);
      *ok = false;
   }
   return p;
}

static PFN_vkVoidFunction
load_device_fn(const struct fns *f, VkDevice device, const char *name, bool required, bool *ok)
{
   PFN_vkVoidFunction p = f->GetDeviceProcAddr(device, name);
   if (!p && required) {
      fprintf(stderr, "compute_g52: vkGetDeviceProcAddr(%s) returned NULL\n", name);
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

/* A host-visible buffer with its memory mapped. */
struct hbuf {
   VkBuffer buf;
   VkDeviceMemory mem;
   void *map;
   bool coherent;
   VkDeviceSize size;
};

static bool
hbuf_create(const struct fns *f, VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            VkDeviceSize size, VkBufferUsageFlags usage, const char *what, struct hbuf *b)
{
   memset(b, 0, sizeof(*b));
   b->size = size;
   const VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkResult r = f->CreateBuffer(dev, &bci, NULL, &b->buf);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkCreateBuffer (%s) failed: %s\n", what, result_str(r));
      return false;
   }
   VkMemoryRequirements reqs;
   f->GetBufferMemoryRequirements(dev, b->buf, &reqs);
   int type = pick_memory_type(mp, reqs.memoryTypeBits, &b->coherent);
   if (type < 0) {
      fprintf(stderr, "compute_g52: no host-visible memory type for the %s buffer\n", what);
      return false;
   }
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)type,
   };
   r = f->AllocateMemory(dev, &mai, NULL, &b->mem);
   if (r == VK_SUCCESS)
      r = f->BindBufferMemory(dev, b->buf, b->mem, 0);
   if (r == VK_SUCCESS)
      r = f->MapMemory(dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: memory for the %s buffer failed: %s\n", what, result_str(r));
      return false;
   }
   return true;
}

static void
hbuf_flush(const struct fns *f, VkDevice dev, const struct hbuf *b, bool invalidate)
{
   if (b->coherent)
      return;
   const VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = b->mem,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
   };
   if (invalidate)
      f->InvalidateMappedMemoryRanges(dev, 1, &whole);
   else
      f->FlushMappedMemoryRanges(dev, 1, &whole);
}

static void
hbuf_destroy(const struct fns *f, VkDevice dev, struct hbuf *b)
{
   if (b->map)
      f->UnmapMemory(dev, b->mem);
   if (b->buf)
      f->DestroyBuffer(dev, b->buf, NULL);
   if (b->mem)
      f->FreeMemory(dev, b->mem, NULL);
   memset(b, 0, sizeof(*b));
}

/* Bits k of v moved to bit 2k (4 bits): the u-interleaved tile order. */
static unsigned
spread4(unsigned v)
{
   return (v & 1) | ((v & 2) << 1) | ((v & 4) << 2) | ((v & 8) << 3);
}

/* ---------------------------------------------------------------------- */

int
main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IOLBF, 0);

   const char *path = argc > 1 ? argv[1] : getenv("MALISX2_ICD");
   if (!path || !*path)
      path = COMPUTE_G52_DEFAULT_ICD;

   printf("compute_g52: loading %s\n", path);
   void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
   if (!lib) {
      fprintf(stderr, "compute_g52: dlopen(%s) failed: %s\n", path, dlerror());
      return 2;
   }

   struct fns f;
   memset(&f, 0, sizeof(f));
   f.gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vk_icdGetInstanceProcAddr");
   if (!f.gipa) {
      fprintf(stderr, "compute_g52: %s exports no vk_icdGetInstanceProcAddr: %s\n", path,
              dlerror());
      return 2;
   }

   bool ok = true;
   f.CreateInstance = (PFN_vkCreateInstance)load_instance_fn(&f, NULL, "vkCreateInstance", &ok);
   if (!ok)
      return 2;

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "compute_g52",
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
      fprintf(stderr, "compute_g52: vkCreateInstance failed: %s\n", result_str(r));
      return 2;
   }

#define INST(name) f.name = (PFN_vk##name)load_instance_fn(&f, instance, "vk" #name, &ok)
   INST(DestroyInstance);
   INST(EnumeratePhysicalDevices);
   INST(GetPhysicalDeviceProperties2);
   INST(GetPhysicalDeviceMemoryProperties);
   INST(GetPhysicalDeviceImageFormatProperties);
   INST(EnumerateDeviceExtensionProperties);
   INST(CreateDevice);
   INST(GetDeviceProcAddr);
#undef INST
   if (!ok)
      return 2;

   uint32_t pdev_count = 0;
   r = f.EnumeratePhysicalDevices(instance, &pdev_count, NULL);
   if (r != VK_SUCCESS || pdev_count == 0) {
      fprintf(stderr, "compute_g52: no physical device (%s, count %u): the driver refused "
              "/dev/mali0; see the driver log\n", result_str(r), pdev_count);
      return 2;
   }
   VkPhysicalDevice pdevs[8];
   if (pdev_count > 8)
      pdev_count = 8;
   r = f.EnumeratePhysicalDevices(instance, &pdev_count, pdevs);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      fprintf(stderr, "compute_g52: vkEnumeratePhysicalDevices failed: %s\n", result_str(r));
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

   /* Linear storage images are allowed when the format properties say so. */
   VkImageFormatProperties ifp;
   const bool linear_storage =
      f.GetPhysicalDeviceImageFormatProperties(pdev, VK_FORMAT_R8G8B8A8_UNORM,
                                               VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR,
                                               VK_IMAGE_USAGE_STORAGE_BIT, 0, &ifp) == VK_SUCCESS;

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
   const VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = have_fault_ext ? &fault_feat : NULL,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .enabledExtensionCount = have_fault_ext ? 1 : 0,
      .ppEnabledExtensionNames = dev_exts,
   };
   VkDevice device = VK_NULL_HANDLE;
   r = f.CreateDevice(pdev, &dci, NULL, &device);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkCreateDevice failed: %s (see the log)\n", result_str(r));
      return 2;
   }
   printf("  VkDevice created%s\n", have_fault_ext ? " (VK_EXT_device_fault enabled)" : "");

#define DEV(name) f.name = (PFN_vk##name)load_device_fn(&f, device, "vk" #name, true, &ok)
   DEV(DestroyDevice);
   DEV(GetDeviceQueue);
   DEV(CreateImage);
   DEV(DestroyImage);
   DEV(GetImageMemoryRequirements);
   DEV(GetImageSubresourceLayout);
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
   DEV(CreateDescriptorSetLayout);
   DEV(DestroyDescriptorSetLayout);
   DEV(CreateDescriptorPool);
   DEV(DestroyDescriptorPool);
   DEV(AllocateDescriptorSets);
   DEV(UpdateDescriptorSets);
   DEV(CreatePipelineLayout);
   DEV(DestroyPipelineLayout);
   DEV(CreateComputePipelines);
   DEV(DestroyPipeline);
   DEV(CreateCommandPool);
   DEV(DestroyCommandPool);
   DEV(AllocateCommandBuffers);
   DEV(BeginCommandBuffer);
   DEV(EndCommandBuffer);
   DEV(CmdPipelineBarrier);
   DEV(CmdBindPipeline);
   DEV(CmdBindDescriptorSets);
   DEV(CmdPushConstants);
   DEV(CmdDispatch);
   DEV(CmdDispatchBase);
   DEV(CreateFence);
   DEV(DestroyFence);
   DEV(QueueSubmit);
   DEV(WaitForFences);
   DEV(DeviceWaitIdle);
#undef DEV
   if (have_fault_ext)
      f.GetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)load_device_fn(
         &f, device, "vkGetDeviceFaultInfoEXT", false, &ok);
   if (!ok)
      return 1;

   VkQueue queue = VK_NULL_HANDLE;
   f.GetDeviceQueue(device, 0, 0, &queue);

   /* The two storage buffers. */
   struct hbuf in, out;
   if (!hbuf_create(&f, device, &mem_props, WORDS * sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "input", &in) ||
       !hbuf_create(&f, device, &mem_props, WORDS * sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "output", &out))
      return 1;
   uint32_t *in_words = in.map;
   for (uint32_t i = 0; i < WORDS; i++)
      in_words[i] = i * 7u + 11u;
   memset(out.map, 0xa5, WORDS * sizeof(uint32_t));
   hbuf_flush(&f, device, &in, false);
   hbuf_flush(&f, device, &out, false);
   printf("  storage buffers: 2 x %u bytes\n", (unsigned)(WORDS * sizeof(uint32_t)));

   /* The storage image. */
   const VkImageTiling tiling = linear_storage ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
   const VkImageCreateInfo ici_img = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {IMG_W, IMG_H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = tiling,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image = VK_NULL_HANDLE;
   r = f.CreateImage(device, &ici_img, NULL, &image);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkCreateImage failed: %s\n", result_str(r));
      return 1;
   }
   VkMemoryRequirements reqs;
   f.GetImageMemoryRequirements(device, image, &reqs);
   bool icoherent = false;
   int itype = pick_memory_type(&mem_props, reqs.memoryTypeBits, &icoherent);
   if (itype < 0) {
      fprintf(stderr, "compute_g52: no host-visible memory type for the image\n");
      return 1;
   }
   const VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)itype,
   };
   VkDeviceMemory imem = VK_NULL_HANDLE;
   void *imap = NULL;
   r = f.AllocateMemory(device, &imai, NULL, &imem);
   if (r == VK_SUCCESS)
      r = f.BindImageMemory(device, image, imem, 0);
   if (r == VK_SUCCESS)
      r = f.MapMemory(device, imem, 0, VK_WHOLE_SIZE, 0, &imap);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: memory for the image failed: %s\n", result_str(r));
      return 1;
   }
   memset(imap, 0xa5, (size_t)reqs.size);
   const VkMappedMemoryRange iwhole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = imem,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
   };
   if (!icoherent)
      f.FlushMappedMemoryRanges(device, 1, &iwhole);
   VkSubresourceLayout layout = {.rowPitch = IMG_W * 4, .offset = 0};
   if (tiling == VK_IMAGE_TILING_LINEAR) {
      const VkImageSubresource sub = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT};
      f.GetImageSubresourceLayout(device, image, &sub, &layout);
   }
   printf("  storage image: %ux%u R8G8B8A8_UNORM %s, %llu bytes%s\n", IMG_W, IMG_H,
          tiling == VK_IMAGE_TILING_LINEAR ? "linear" : "optimal (u-interleaved, detiled on the CPU)",
          (unsigned long long)reqs.size,
          tiling == VK_IMAGE_TILING_LINEAR ? "" : "");

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
      fprintf(stderr, "compute_g52: vkCreateImageView (storage) failed: %s\n", result_str(r));
      return 1;
   }

   /* Descriptors. */
   const VkDescriptorSetLayoutBinding bindings[3] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
   };
   const VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
   r = f.CreateDescriptorSetLayout(device, &dslci, NULL, &dsl);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkCreateDescriptorSetLayout failed: %s\n", result_str(r));
      return 1;
   }
   const VkDescriptorPoolSize pool_sizes[2] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
   };
   const VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool dpool = VK_NULL_HANDLE;
   r = f.CreateDescriptorPool(device, &dpci, NULL, &dpool);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkCreateDescriptorPool failed: %s\n", result_str(r));
      return 1;
   }
   const VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool,
      .descriptorSetCount = 1,
      .pSetLayouts = &dsl,
   };
   VkDescriptorSet set = VK_NULL_HANDLE;
   r = f.AllocateDescriptorSets(device, &dsai, &set);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkAllocateDescriptorSets failed: %s\n", result_str(r));
      return 1;
   }
   const VkDescriptorBufferInfo bi_in = {in.buf, 0, VK_WHOLE_SIZE};
   const VkDescriptorBufferInfo bi_out = {out.buf, 0, VK_WHOLE_SIZE};
   const VkDescriptorImageInfo ii = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
   const VkWriteDescriptorSet writes[3] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pBufferInfo = &bi_in},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pBufferInfo = &bi_out},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 2,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
       .pImageInfo = &ii},
   };
   f.UpdateDescriptorSets(device, 3, writes, 0, NULL);
   printf("  descriptor set written (2 storage buffers, 1 storage image)\n");

   /* Shader and pipeline. */
   const VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(comp_spv),
      .pCode = comp_spv,
   };
   VkShaderModule sm = VK_NULL_HANDLE;
   r = f.CreateShaderModule(device, &smci, NULL, &sm);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkCreateShaderModule failed: %s\n", result_str(r));
      return 1;
   }
   const VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
   const VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl = VK_NULL_HANDLE;
   r = f.CreatePipelineLayout(device, &plci, NULL, &pl);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkCreatePipelineLayout failed: %s\n", result_str(r));
      return 1;
   }
   const VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      /* The second dispatch has a nonzero base workgroup. */
      .flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = sm,
         .pName = "main",
      },
      .layout = pl,
   };
   VkPipeline pipeline = VK_NULL_HANDLE;
   const int64_t tc0 = now_ns();
   r = f.CreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipeline);
   const int64_t tc1 = now_ns();
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkCreateComputePipelines failed: %s (the Bifrost compile, "
              "the descriptor lowering or a TODO v7 refusal; see the log)\n", result_str(r));
      return 1;
   }
   printf("  compute pipeline created in %.2f ms (Bifrost compiler)\n", (double)(tc1 - tc0) / 1e6);

   const VkCommandPoolCreateInfo cpoolci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkCommandPool pool = VK_NULL_HANDLE;
   r = f.CreateCommandPool(device, &cpoolci, NULL, &pool);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkCreateCommandPool failed: %s\n", result_str(r));
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
      fprintf(stderr, "compute_g52: vkAllocateCommandBuffers failed: %s\n", result_str(r));
      return 1;
   }
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   r = f.BeginCommandBuffer(cmd, &bi);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkBeginCommandBuffer failed: %s\n", result_str(r));
      return 1;
   }

   const VkImageMemoryBarrier to_general = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   const VkMemoryBarrier host_to_shader = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
   };
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                        1, &host_to_shader, 0, NULL, 1, &to_general);

   f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   f.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
   const uint32_t pc[2] = {PC_MUL, PC_ADD};
   f.CmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
   f.CmdDispatch(cmd, N / WG, 1, 1);
   /* Words N.. : the base workgroup moves gl_GlobalInvocationID past the
    * first dispatch's (and past the image stores), with a new pc.add. The
    * two dispatches write disjoint words: no barrier between them. */
   const uint32_t pc2[2] = {PC_MUL, PC_ADD2};
   f.CmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), pc2);
   f.CmdDispatchBase(cmd, N / WG, 0, 0, N / WG, 1, 1);

   const VkMemoryBarrier shader_to_host = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                        1, &shader_to_host, 0, NULL, 0, NULL);

   r = f.EndCommandBuffer(cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkEndCommandBuffer failed: %s (recording failed: the "
              "descriptor tables, the copy job, a TODO v7 stub or command memory; see the "
              "log)\n", result_str(r));
      return 1;
   }
   printf("  command buffer recorded (vkCmdDispatch %u x 1 x 1, then vkCmdDispatchBase from "
          "workgroup %u, %u x 1 x 1; workgroups of %u)\n", N / WG, N / WG, N / WG, WG);

   const VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence = VK_NULL_HANDLE;
   r = f.CreateFence(device, &fci, NULL, &fence);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "compute_g52: vkCreateFence failed: %s\n", result_str(r));
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
      fprintf(stderr, "compute_g52: vkQueueSubmit failed: %s (kbase JOB_SUBMIT; see the log)\n",
              result_str(r));
      if (r == VK_ERROR_DEVICE_LOST)
         print_fault(&f, device);
      return 1;
   }
   printf("  submitted (one vtc atom: a descriptor copy job and a compute job per dispatch); "
          "waiting up to %llu ms\n", (unsigned long long)(WAIT_NS / 1000000));

   r = f.WaitForFences(device, 1, &fence, VK_TRUE, WAIT_NS);
   const int64_t t1 = now_ns();
   printf("  vkWaitForFences: %s after %.3f ms\n", result_str(r), (double)(t1 - t0) / 1e6);
   if (r == VK_TIMEOUT) {
      printf("FAIL: the fence never signalled: a compute job did not complete, or its event "
             "was never read. Check dmesg for mali/kbase lines.\n");
      return 1;
   }
   if (r != VK_SUCCESS) {
      printf("FAIL: vkWaitForFences returned %s\n", result_str(r));
      if (r == VK_ERROR_DEVICE_LOST)
         print_fault(&f, device);
      return 1;
   }

   hbuf_flush(&f, device, &out, true);
   if (!icoherent)
      f.InvalidateMappedMemoryRanges(device, 1, &iwhole);

   /* The buffer: words 0..N-1 from vkCmdDispatch with PC_ADD, words
    * N..WORDS-1 from vkCmdDispatchBase with PC_ADD2. */
   const uint32_t *out_words = out.map;
   uint32_t bad = 0, untouched = 0, bad_half[2] = {0, 0}, untouched_half[2] = {0, 0};
   /* The known wrong shapes: pc.add read as 0, the second dispatch with
    * the first one's push constants, the first dispatch's words written by
    * the second (its base lost). */
   uint32_t no_add = 0, stale_pc = 0, first_half_add2 = 0;
   int first_bad = -1;
   for (uint32_t i = 0; i < WORDS; i++) {
      const unsigned half = i >= N;
      const uint32_t want = word_want(in_words, i);
      if (out_words[i] == want)
         continue;
      if (out_words[i] == 0xa5a5a5a5u) {
         untouched++;
         untouched_half[half]++;
      }
      const uint32_t base_word = in_words[i] * PC_MUL + i;
      if (out_words[i] == base_word)
         no_add++;
      else if (half && out_words[i] == base_word + PC_ADD)
         stale_pc++;
      else if (!half && out_words[i] == base_word + PC_ADD2)
         first_half_add2++;
      if (first_bad < 0)
         first_bad = (int)i;
      bad++;
      bad_half[half]++;
   }
   printf("  buffer (vkCmdDispatch): out[0] = 0x%08x (want 0x%08x), out[%u] = 0x%08x "
          "(want 0x%08x)\n", out_words[0], word_want(in_words, 0), N - 1, out_words[N - 1],
          word_want(in_words, N - 1));
   printf("  buffer (vkCmdDispatchBase): out[%u] = 0x%08x (want 0x%08x), out[%u] = 0x%08x "
          "(want 0x%08x)\n", N, out_words[N], word_want(in_words, N), WORDS - 1,
          out_words[WORDS - 1], word_want(in_words, WORDS - 1));
   if (bad) {
      printf("  buffer: %u of %u words wrong (%u still hold the 0xa5 sentinel); first at %d = "
             "0x%08x, want 0x%08x\n", bad, WORDS, untouched, first_bad, out_words[first_bad],
             word_want(in_words, (uint32_t)first_bad));
      printf("  buffer: vkCmdDispatch half %u wrong (%u untouched), vkCmdDispatchBase half %u "
             "wrong (%u untouched)\n", bad_half[0], untouched_half[0], bad_half[1],
             untouched_half[1]);
      if (no_add)
         printf("  buffer: %u words lack pc.add: it read as 0 (the upper half of the push "
                "constant FAU word)\n", no_add);
      if (stale_pc)
         printf("  buffer: %u of the second dispatch's words carry the first dispatch's pc.add: "
                "stale push constants\n", stale_pc);
      if (first_half_add2)
         printf("  buffer: %u of the first dispatch's words were written by the second "
                "dispatch%s: its base workgroup (system values) was ignored\n", first_half_add2,
                untouched_half[1] == N ? ", which left its own words untouched" : "");
   }

   /* The image: R = x * 17, G = y * 17, B = 128, A = 255 (one LSB of slack:
    * float to UNORM rounding). */
   const uint8_t *ibase = (const uint8_t *)imap + layout.offset;
   uint32_t ibad = 0, iuntouched = 0;
   int fx = -1, fy = -1;
   uint8_t fgot[4] = {0};
   for (uint32_t y = 0; y < IMG_H; y++) {
      for (uint32_t x = 0; x < IMG_W; x++) {
         const uint8_t *px;
         if (tiling == VK_IMAGE_TILING_LINEAR) {
            px = ibase + (size_t)y * layout.rowPitch + (size_t)x * 4;
         } else {
            /* One 16x16 u-interleaved tile: block index spread(y) * 3 ^ spread(x). */
            const unsigned idx = (spread4(y) * 3) ^ spread4(x);
            px = ibase + (size_t)idx * 4;
         }
         const int want[4] = {(int)x * 17, (int)y * 17, 128, 255};
         bool okpx = true;
         for (unsigned c = 0; c < 4; c++) {
            const int d = (int)px[c] - want[c];
            if (d < -1 || d > 1)
               okpx = false;
         }
         if (okpx)
            continue;
         if (px[0] == 0xa5 && px[1] == 0xa5 && px[2] == 0xa5 && px[3] == 0xa5)
            iuntouched++;
         if (!ibad) {
            fx = (int)x;
            fy = (int)y;
            memcpy(fgot, px, 4);
         }
         ibad++;
      }
   }
   const uint8_t *p00 = ibase;
   const uint8_t *pll = tiling == VK_IMAGE_TILING_LINEAR
                           ? ibase + (size_t)(IMG_H - 1) * layout.rowPitch + (IMG_W - 1) * 4
                           : ibase + (size_t)(((spread4(IMG_H - 1) * 3) ^ spread4(IMG_W - 1)) * 4);
   printf("  image: pixel (0,0) = %u %u %u %u (want 0 0 128 255), pixel (%u,%u) = %u %u %u %u "
          "(want 255 255 128 255)\n", p00[0], p00[1], p00[2], p00[3], IMG_W - 1, IMG_H - 1,
          pll[0], pll[1], pll[2], pll[3]);
   if (ibad)
      printf("  image: %u of %u pixels wrong (%u still hold the 0xa5 sentinel); first at (%d,%d) "
             "= %u %u %u %u, want %d %d 128 255\n", ibad, IMG_W * IMG_H, iuntouched, fx, fy,
             fgot[0], fgot[1], fgot[2], fgot[3], fx * 17, fy * 17);

   int status;
   if (!bad && !ibad) {
      printf("PASS: %u buffer words (two dispatches) and %u image pixels as expected\n", WORDS,
             IMG_W * IMG_H);
      status = 0;
   } else {
      printf("FAIL: %u buffer words and %u image pixels wrong\n", bad, ibad);
      status = 1;
   }

   f.DeviceWaitIdle(device);
   f.DestroyFence(device, fence, NULL);
   f.DestroyCommandPool(device, pool, NULL);
   f.DestroyPipeline(device, pipeline, NULL);
   f.DestroyPipelineLayout(device, pl, NULL);
   f.DestroyShaderModule(device, sm, NULL);
   f.DestroyDescriptorPool(device, dpool, NULL);
   f.DestroyDescriptorSetLayout(device, dsl, NULL);
   f.DestroyImageView(device, view, NULL);
   f.UnmapMemory(device, imem);
   f.DestroyImage(device, image, NULL);
   f.FreeMemory(device, imem, NULL);
   hbuf_destroy(&f, device, &in);
   hbuf_destroy(&f, device, &out);
   f.DestroyDevice(device, NULL);
   f.DestroyInstance(instance, NULL);
   dlclose(lib);
   return status;
}
