/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * The parts of a G52 device test that every test repeats (clear_g52.c and
 * draw_g52.c carry their own copies; the later tests include this): dlopen
 * the driver and reach it through vk_icdGetInstanceProcAddr (no loader, no
 * ICD manifest), create the instance, pick the first physical device,
 * create the device with VK_EXT_device_fault when offered, load the entry
 * points the tests use, and the small helpers around images, buffers,
 * pipelines and a submit that waits on a fence and prints the device fault
 * record on a loss.
 *
 * Plain C11, static inline functions only (each test is one translation
 * unit and uses a part of them).
 * Every failure prints why to stderr with the test's name in front; the
 * caller decides the exit status (2 before a VkDevice exists, 1 after).
 */

#ifndef G52_HARNESS_H
#define G52_HARNESS_H

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

#define G52_WAIT_NS (5ull * 1000 * 1000 * 1000)

static inline const char *
g52_result_str(VkResult r)
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

static inline int64_t
g52_now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

struct g52_fns {
   PFN_vkGetInstanceProcAddr gipa;
   PFN_vkCreateInstance CreateInstance;
   PFN_vkDestroyInstance DestroyInstance;
   PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
   PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
   PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
   PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties;
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
   PFN_vkResetCommandPool ResetCommandPool;
   PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
   PFN_vkBeginCommandBuffer BeginCommandBuffer;
   PFN_vkEndCommandBuffer EndCommandBuffer;
   PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
   PFN_vkCmdBeginRendering CmdBeginRendering;
   PFN_vkCmdEndRendering CmdEndRendering;
   PFN_vkCmdBindPipeline CmdBindPipeline;
   PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers;
   PFN_vkCmdDraw CmdDraw;
   /* Transfers (optional: NULL when the driver lacks them). */
   PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;
   PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
   PFN_vkCmdBlitImage CmdBlitImage;
   PFN_vkCmdClearAttachments CmdClearAttachments;
   PFN_vkCreateFence CreateFence;
   PFN_vkDestroyFence DestroyFence;
   PFN_vkResetFences ResetFences;
   PFN_vkQueueSubmit QueueSubmit;
   PFN_vkWaitForFences WaitForFences;
   PFN_vkDeviceWaitIdle DeviceWaitIdle;
   PFN_vkGetDeviceFaultInfoEXT GetDeviceFaultInfoEXT; /* optional */
};

struct g52 {
   const char *prog;
   void *lib;
   struct g52_fns f;
   VkInstance instance;
   VkPhysicalDevice pdev;
   VkPhysicalDeviceMemoryProperties mem_props;
   VkDevice device;
   VkQueue queue;
   bool have_fault_ext;
};

static inline PFN_vkVoidFunction
g52_load_instance_fn(struct g52 *g, const char *name, bool *ok)
{
   PFN_vkVoidFunction p = g->f.gipa(g->instance, name);
   if (!p) {
      fprintf(stderr, "%s: vk_icdGetInstanceProcAddr(%s) returned NULL\n", g->prog, name);
      *ok = false;
   }
   return p;
}

static inline PFN_vkVoidFunction
g52_load_device_fn(struct g52 *g, const char *name, const char *alt, bool required, bool *ok)
{
   PFN_vkVoidFunction p = g->f.GetDeviceProcAddr(g->device, name);
   if (!p && alt)
      p = g->f.GetDeviceProcAddr(g->device, alt);
   if (!p && required) {
      fprintf(stderr, "%s: vkGetDeviceProcAddr(%s%s%s) returned NULL\n", g->prog, name,
              alt ? " / " : "", alt ? alt : "");
      *ok = false;
   }
   return p;
}

static inline void
g52_print_fault(struct g52 *g)
{
   const struct g52_fns *f = &g->f;
   if (!f->GetDeviceFaultInfoEXT) {
      printf("  (VK_EXT_device_fault not enabled; no fault record)\n");
      return;
   }
   VkDeviceFaultCountsEXT counts = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
   VkResult r = f->GetDeviceFaultInfoEXT(g->device, &counts, NULL);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      printf("  vkGetDeviceFaultInfoEXT: %s\n", g52_result_str(r));
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
   r = f->GetDeviceFaultInfoEXT(g->device, &counts, &info);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      printf("  vkGetDeviceFaultInfoEXT: %s\n", g52_result_str(r));
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

/* A memory type for `type_bits`: host-visible (host-coherent preferred)
 * when `host` is set, else the first one offered; -1 when there is none. */
static inline int
g52_pick_memory_type(const struct g52 *g, uint32_t type_bits, bool host, bool *coherent)
{
   const VkPhysicalDeviceMemoryProperties *mp = &g->mem_props;
   int type = -1;
   *coherent = false;
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      VkMemoryPropertyFlags fl = mp->memoryTypes[i].propertyFlags;
      if (!(type_bits & (1u << i)))
         continue;
      if (!host)
         return (int)i;
      if (!(fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         continue;
      const bool c = fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      if (type < 0 || (c && !*coherent)) {
         type = (int)i;
         *coherent = c;
      }
   }
   return type;
}

/*
 * Opens the driver named by argv[1], $MALISX2_ICD or `default_icd`, and
 * creates the instance, device and queue. Returns 0, or the exit status
 * (2) after printing why.
 */
static inline int
g52_open(struct g52 *g, const char *prog, int argc, char **argv, const char *default_icd)
{
   memset(g, 0, sizeof(*g));
   g->prog = prog;
   setvbuf(stdout, NULL, _IOLBF, 0);

   const char *path = argc > 1 ? argv[1] : getenv("MALISX2_ICD");
   if (!path || !*path)
      path = default_icd;

   printf("%s: loading %s\n", prog, path);
   g->lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
   if (!g->lib) {
      fprintf(stderr, "%s: dlopen(%s) failed: %s\n", prog, path, dlerror());
      return 2;
   }
   struct g52_fns *f = &g->f;
   f->gipa = (PFN_vkGetInstanceProcAddr)dlsym(g->lib, "vk_icdGetInstanceProcAddr");
   if (!f->gipa) {
      fprintf(stderr, "%s: %s exports no vk_icdGetInstanceProcAddr: %s\n", prog, path, dlerror());
      return 2;
   }

   bool ok = true;
   f->CreateInstance = (PFN_vkCreateInstance)g52_load_instance_fn(g, "vkCreateInstance", &ok);
   if (!ok)
      return 2;

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = prog,
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkResult r = f->CreateInstance(&ici, NULL, &g->instance);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateInstance failed: %s\n", prog, g52_result_str(r));
      return 2;
   }

#define INST(name) f->name = (PFN_vk##name)g52_load_instance_fn(g, "vk" #name, &ok)
   INST(DestroyInstance);
   INST(EnumeratePhysicalDevices);
   INST(GetPhysicalDeviceProperties2);
   INST(GetPhysicalDeviceMemoryProperties);
   INST(GetPhysicalDeviceFormatProperties);
   INST(EnumerateDeviceExtensionProperties);
   INST(CreateDevice);
   INST(GetDeviceProcAddr);
#undef INST
   if (!ok)
      return 2;

   uint32_t pdev_count = 0;
   r = f->EnumeratePhysicalDevices(g->instance, &pdev_count, NULL);
   if (r != VK_SUCCESS || pdev_count == 0) {
      fprintf(stderr, "%s: no physical device (%s, count %u): the driver refused /dev/mali0; "
              "see the driver log\n", prog, g52_result_str(r), pdev_count);
      return 2;
   }
   VkPhysicalDevice pdevs[8];
   if (pdev_count > 8)
      pdev_count = 8;
   r = f->EnumeratePhysicalDevices(g->instance, &pdev_count, pdevs);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      fprintf(stderr, "%s: vkEnumeratePhysicalDevices failed: %s\n", prog, g52_result_str(r));
      return 2;
   }
   g->pdev = pdevs[0];

   VkPhysicalDeviceDriverProperties drv = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &drv,
   };
   f->GetPhysicalDeviceProperties2(g->pdev, &props2);
   const VkPhysicalDeviceProperties *p = &props2.properties;
   printf("  device: %s, apiVersion %u.%u.%u, driverInfo \"%s\"\n", p->deviceName,
          VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion),
          VK_API_VERSION_PATCH(p->apiVersion), drv.driverInfo);
   f->GetPhysicalDeviceMemoryProperties(g->pdev, &g->mem_props);

   {
      uint32_t n = 0;
      if (f->EnumerateDeviceExtensionProperties(g->pdev, NULL, &n, NULL) == VK_SUCCESS && n) {
         VkExtensionProperties *ext = calloc(n, sizeof(*ext));
         if (ext && f->EnumerateDeviceExtensionProperties(g->pdev, NULL, &n, ext) == VK_SUCCESS) {
            for (uint32_t i = 0; i < n; i++)
               if (!strcmp(ext[i].extensionName, VK_EXT_DEVICE_FAULT_EXTENSION_NAME))
                  g->have_fault_ext = true;
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
      .pNext = g->have_fault_ext ? &fault_feat : NULL,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .enabledExtensionCount = g->have_fault_ext ? 1 : 0,
      .ppEnabledExtensionNames = dev_exts,
   };
   r = f->CreateDevice(g->pdev, &dci, NULL, &g->device);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateDevice failed: %s (see the log)\n", prog, g52_result_str(r));
      return 2;
   }
   printf("  VkDevice created%s\n", g->have_fault_ext ? " (VK_EXT_device_fault enabled)" : "");

#define DEV(name) f->name = (PFN_vk##name)g52_load_device_fn(g, "vk" #name, NULL, true, &ok)
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
   DEV(ResetCommandPool);
   DEV(AllocateCommandBuffers);
   DEV(BeginCommandBuffer);
   DEV(EndCommandBuffer);
   DEV(CmdPipelineBarrier);
   DEV(CmdBindPipeline);
   DEV(CmdBindVertexBuffers);
   DEV(CmdDraw);
   DEV(CreateFence);
   DEV(DestroyFence);
   DEV(ResetFences);
   DEV(QueueSubmit);
   DEV(WaitForFences);
   DEV(DeviceWaitIdle);
#undef DEV
   f->CmdBeginRendering = (PFN_vkCmdBeginRendering)g52_load_device_fn(
      g, "vkCmdBeginRendering", "vkCmdBeginRenderingKHR", true, &ok);
   f->CmdEndRendering = (PFN_vkCmdEndRendering)g52_load_device_fn(
      g, "vkCmdEndRendering", "vkCmdEndRenderingKHR", true, &ok);
   f->GetImageSubresourceLayout = (PFN_vkGetImageSubresourceLayout)g52_load_device_fn(
      g, "vkGetImageSubresourceLayout", NULL, false, &ok);
   f->CmdCopyBufferToImage = (PFN_vkCmdCopyBufferToImage)g52_load_device_fn(
      g, "vkCmdCopyBufferToImage", NULL, false, &ok);
   f->CmdCopyImageToBuffer = (PFN_vkCmdCopyImageToBuffer)g52_load_device_fn(
      g, "vkCmdCopyImageToBuffer", NULL, false, &ok);
   f->CmdBlitImage = (PFN_vkCmdBlitImage)g52_load_device_fn(g, "vkCmdBlitImage", NULL, false,
                                                             &ok);
   f->CmdClearAttachments = (PFN_vkCmdClearAttachments)g52_load_device_fn(
      g, "vkCmdClearAttachments", NULL, false, &ok);
   if (g->have_fault_ext)
      f->GetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)g52_load_device_fn(
         g, "vkGetDeviceFaultInfoEXT", NULL, false, &ok);
   if (!ok)
      return 2;

   f->GetDeviceQueue(g->device, 0, 0, &g->queue);
   return 0;
}

static inline void
g52_close(struct g52 *g)
{
   if (g->device) {
      g->f.DeviceWaitIdle(g->device);
      g->f.DestroyDevice(g->device, NULL);
   }
   if (g->instance)
      g->f.DestroyInstance(g->instance, NULL);
   if (g->lib)
      dlclose(g->lib);
   memset(g, 0, sizeof(*g));
}

/* ---------------------------------------------------------------------- */
/* Images and buffers                                                      */

struct g52_image {
   VkImage image;
   VkDeviceMemory memory;
   VkImageView view;
   VkDeviceSize size;
   void *map;                   /* host-visible images only */
   bool coherent;
   VkSubresourceLayout layout;  /* linear images: the readback layout */
   VkFormat format;
   uint32_t width, height;
};

/*
 * A 2D single-level image with a view of `aspects`. Host-visible images
 * (readback targets) are mapped and filled with `sentinel`; the others
 * take the first memory type the image accepts.
 */
static inline bool
g52_image_create(struct g52 *g, struct g52_image *im, VkFormat format, uint32_t w, uint32_t h,
                 VkImageTiling tiling, VkImageUsageFlags usage, VkImageAspectFlags aspects,
                 bool host, uint8_t sentinel)
{
   const struct g52_fns *f = &g->f;
   memset(im, 0, sizeof(*im));
   im->format = format;
   im->width = w;
   im->height = h;

   const VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {w, h, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = tiling,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkResult r = f->CreateImage(g->device, &ici, NULL, &im->image);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateImage (format %d) failed: %s\n", g->prog, format,
              g52_result_str(r));
      return false;
   }
   VkMemoryRequirements reqs;
   f->GetImageMemoryRequirements(g->device, im->image, &reqs);
   im->size = reqs.size;
   int type = g52_pick_memory_type(g, reqs.memoryTypeBits, host, &im->coherent);
   if (type < 0) {
      fprintf(stderr, "%s: no %smemory type for the image (bits 0x%x)\n", g->prog,
              host ? "host-visible " : "", reqs.memoryTypeBits);
      return false;
   }
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)type,
   };
   r = f->AllocateMemory(g->device, &mai, NULL, &im->memory);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkAllocateMemory (image) failed: %s\n", g->prog, g52_result_str(r));
      return false;
   }
   r = f->BindImageMemory(g->device, im->image, im->memory, 0);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkBindImageMemory failed: %s\n", g->prog, g52_result_str(r));
      return false;
   }
   if (host) {
      r = f->MapMemory(g->device, im->memory, 0, VK_WHOLE_SIZE, 0, &im->map);
      if (r != VK_SUCCESS) {
         fprintf(stderr, "%s: vkMapMemory (image) failed: %s\n", g->prog, g52_result_str(r));
         return false;
      }
      memset(im->map, sentinel, (size_t)reqs.size);
      if (!im->coherent) {
         const VkMappedMemoryRange whole = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = im->memory,
            .size = VK_WHOLE_SIZE,
         };
         f->FlushMappedMemoryRanges(g->device, 1, &whole);
      }
   }
   im->layout.rowPitch = (VkDeviceSize)w * 4;
   if (tiling == VK_IMAGE_TILING_LINEAR && f->GetImageSubresourceLayout) {
      const VkImageSubresource sub = {.aspectMask = aspects};
      f->GetImageSubresourceLayout(g->device, im->image, &sub, &im->layout);
   }

   const VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = im->image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = {aspects, 0, 1, 0, 1},
   };
   r = f->CreateImageView(g->device, &vci, NULL, &im->view);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateImageView (format %d, aspects 0x%x) failed: %s\n", g->prog,
              format, aspects, g52_result_str(r));
      return false;
   }
   printf("  image: %ux%u format %d %s, %llu bytes%s, memory type %d\n", w, h, format,
          tiling == VK_IMAGE_TILING_LINEAR ? "linear" : "optimal", (unsigned long long)reqs.size,
          host ? ", host-visible" : "", type);
   return true;
}

/* Re-fills a host-visible image with the sentinel (before a second run). */
static inline void
g52_image_refill(struct g52 *g, struct g52_image *im, uint8_t sentinel)
{
   if (!im->map)
      return;
   memset(im->map, sentinel, (size_t)im->size);
   if (!im->coherent) {
      const VkMappedMemoryRange whole = {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = im->memory,
         .size = VK_WHOLE_SIZE,
      };
      g->f.FlushMappedMemoryRanges(g->device, 1, &whole);
   }
}

/* Makes the GPU's writes visible to the mapping and returns pixel (x, y)
 * of a linear 4-byte-per-pixel image. */
static inline void
g52_image_sync_for_read(struct g52 *g, struct g52_image *im)
{
   if (im->coherent)
      return;
   const VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = im->memory,
      .size = VK_WHOLE_SIZE,
   };
   g->f.InvalidateMappedMemoryRanges(g->device, 1, &whole);
}

static inline const uint8_t *
g52_image_pixel(const struct g52_image *im, uint32_t x, uint32_t y)
{
   return (const uint8_t *)im->map + im->layout.offset + (size_t)y * im->layout.rowPitch +
          (size_t)x * 4;
}

static inline void
g52_image_destroy(struct g52 *g, struct g52_image *im)
{
   const struct g52_fns *f = &g->f;
   if (im->view)
      f->DestroyImageView(g->device, im->view, NULL);
   if (im->map)
      f->UnmapMemory(g->device, im->memory);
   if (im->image)
      f->DestroyImage(g->device, im->image, NULL);
   if (im->memory)
      f->FreeMemory(g->device, im->memory, NULL);
   memset(im, 0, sizeof(*im));
}

struct g52_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
};

/* A host-visible buffer holding `data`. */
static inline bool
g52_buffer_create(struct g52 *g, struct g52_buffer *b, const void *data, size_t size,
                  VkBufferUsageFlags usage)
{
   const struct g52_fns *f = &g->f;
   memset(b, 0, sizeof(*b));
   const VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkResult r = f->CreateBuffer(g->device, &bci, NULL, &b->buffer);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateBuffer failed: %s\n", g->prog, g52_result_str(r));
      return false;
   }
   VkMemoryRequirements reqs;
   f->GetBufferMemoryRequirements(g->device, b->buffer, &reqs);
   bool coherent = false;
   int type = g52_pick_memory_type(g, reqs.memoryTypeBits, true, &coherent);
   if (type < 0) {
      fprintf(stderr, "%s: no host-visible memory type for the buffer\n", g->prog);
      return false;
   }
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)type,
   };
   r = f->AllocateMemory(g->device, &mai, NULL, &b->memory);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkAllocateMemory (buffer) failed: %s\n", g->prog, g52_result_str(r));
      return false;
   }
   r = f->BindBufferMemory(g->device, b->buffer, b->memory, 0);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkBindBufferMemory failed: %s\n", g->prog, g52_result_str(r));
      return false;
   }
   void *map = NULL;
   r = f->MapMemory(g->device, b->memory, 0, VK_WHOLE_SIZE, 0, &map);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkMapMemory (buffer) failed: %s\n", g->prog, g52_result_str(r));
      return false;
   }
   memcpy(map, data, size);
   if (!coherent) {
      const VkMappedMemoryRange whole = {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = b->memory,
         .size = VK_WHOLE_SIZE,
      };
      f->FlushMappedMemoryRanges(g->device, 1, &whole);
   }
   f->UnmapMemory(g->device, b->memory);
   return true;
}

static inline void
g52_buffer_destroy(struct g52 *g, struct g52_buffer *b)
{
   if (b->buffer)
      g->f.DestroyBuffer(g->device, b->buffer, NULL);
   if (b->memory)
      g->f.FreeMemory(g->device, b->memory, NULL);
   memset(b, 0, sizeof(*b));
}

/*
 * A host-visible buffer that stays mapped: the source of an upload into an
 * image (vkCmdCopyBufferToImage) or the target of a readback
 * (vkCmdCopyImageToBuffer). Filled with `fill` bytes.
 */
struct g52_hostbuf {
   VkBuffer buffer;
   VkDeviceMemory memory;
   void *map;
   VkDeviceSize size;
   bool coherent;
};

static inline bool
g52_hostbuf_create(struct g52 *g, struct g52_hostbuf *b, VkDeviceSize size, uint8_t fill)
{
   const struct g52_fns *f = &g->f;
   memset(b, 0, sizeof(*b));
   b->size = size;
   const VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkResult r = f->CreateBuffer(g->device, &bci, NULL, &b->buffer);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateBuffer (host buffer) failed: %s\n", g->prog, g52_result_str(r));
      return false;
   }
   VkMemoryRequirements reqs;
   f->GetBufferMemoryRequirements(g->device, b->buffer, &reqs);
   int type = g52_pick_memory_type(g, reqs.memoryTypeBits, true, &b->coherent);
   if (type < 0) {
      fprintf(stderr, "%s: no host-visible memory type for the host buffer\n", g->prog);
      return false;
   }
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)type,
   };
   r = f->AllocateMemory(g->device, &mai, NULL, &b->memory);
   if (r == VK_SUCCESS)
      r = f->BindBufferMemory(g->device, b->buffer, b->memory, 0);
   if (r == VK_SUCCESS)
      r = f->MapMemory(g->device, b->memory, 0, VK_WHOLE_SIZE, 0, &b->map);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: host buffer memory: %s\n", g->prog, g52_result_str(r));
      return false;
   }
   memset(b->map, fill, (size_t)size);
   return true;
}

/* After writing b->map from the CPU, before the GPU reads it. */
static inline void
g52_hostbuf_flush(struct g52 *g, struct g52_hostbuf *b)
{
   if (b->coherent)
      return;
   const VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = b->memory,
      .size = VK_WHOLE_SIZE,
   };
   g->f.FlushMappedMemoryRanges(g->device, 1, &whole);
}

/* After the GPU wrote the buffer, before the CPU reads b->map. */
static inline void
g52_hostbuf_invalidate(struct g52 *g, struct g52_hostbuf *b)
{
   if (b->coherent)
      return;
   const VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = b->memory,
      .size = VK_WHOLE_SIZE,
   };
   g->f.InvalidateMappedMemoryRanges(g->device, 1, &whole);
}

static inline void
g52_hostbuf_destroy(struct g52 *g, struct g52_hostbuf *b)
{
   if (b->map)
      g->f.UnmapMemory(g->device, b->memory);
   if (b->buffer)
      g->f.DestroyBuffer(g->device, b->buffer, NULL);
   if (b->memory)
      g->f.FreeMemory(g->device, b->memory, NULL);
   memset(b, 0, sizeof(*b));
}

/* One tightly packed buffer <-> image copy of a whole 2D single-layer
 * image's aspect (level 0, layer 0), recorded into cmd. */
static inline void
g52_cmd_copy_image(struct g52 *g, VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                   uint32_t w, uint32_t h, VkBuffer buffer, bool to_image)
{
   const VkBufferImageCopy r = {
      .bufferOffset = 0,
      .imageSubresource = {aspect, 0, 0, 1},
      .imageExtent = {w, h, 1},
   };
   if (to_image)
      g->f.CmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
   else
      g->f.CmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &r);
}

/* A full barrier between transfers, fragment and compute work (the tests
 * favour certainty over overlap). */
static inline void
g52_cmd_full_barrier(struct g52 *g, VkCommandBuffer cmd)
{
   const VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
   };
   g->f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

/* ---------------------------------------------------------------------- */
/* Pipelines                                                               */

struct g52_pipeline_desc {
   const uint32_t *vs_spv;
   size_t vs_size;
   const uint32_t *fs_spv;
   size_t fs_size;
   VkFormat attrib_format;       /* location 0, binding 0 */
   uint32_t vertex_stride;
   uint32_t width, height;       /* the static viewport and scissor */
   VkRect2D scissor;             /* the static scissor if its extent is not 0 */
   VkFormat color_format;
   VkFormat depth_format;        /* VK_FORMAT_UNDEFINED without */
   VkFormat stencil_format;
   const VkPipelineDepthStencilStateCreateInfo *ds;   /* NULL: all off */
   VkPipelineColorBlendAttachmentState blend;
   float blend_constants[4];
};

/* One vertex + fragment pipeline for dynamic rendering, triangle list, no
 * culling, no dynamic state, no descriptor sets. Prints the creation time. */
static inline VkPipeline
g52_pipeline_create(struct g52 *g, const struct g52_pipeline_desc *d, VkPipelineLayout layout,
                    const char *what)
{
   const struct g52_fns *f = &g->f;
   const VkShaderModuleCreateInfo vsmci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = d->vs_size,
      .pCode = d->vs_spv,
   };
   const VkShaderModuleCreateInfo fsmci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = d->fs_size,
      .pCode = d->fs_spv,
   };
   VkShaderModule vsm = VK_NULL_HANDLE, fsm = VK_NULL_HANDLE;
   VkResult r = f->CreateShaderModule(g->device, &vsmci, NULL, &vsm);
   if (r == VK_SUCCESS)
      r = f->CreateShaderModule(g->device, &fsmci, NULL, &fsm);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateShaderModule (%s) failed: %s\n", g->prog, what,
              g52_result_str(r));
      return VK_NULL_HANDLE;
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
      .stride = d->vertex_stride,
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription va_desc = {
      .location = 0,
      .binding = 0,
      .format = d->attrib_format,
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
   const VkViewport viewport = {0.0f, 0.0f, (float)d->width, (float)d->height, 0.0f, 1.0f};
   const VkRect2D scissor = d->scissor.extent.width && d->scissor.extent.height ?
                               d->scissor : (VkRect2D){{0, 0}, {d->width, d->height}};
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
   const VkPipelineDepthStencilStateCreateInfo ds_off = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
   };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &d->blend,
   };
   memcpy(cb.blendConstants, d->blend_constants, sizeof(cb.blendConstants));
   const VkPipelineRenderingCreateInfo prci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &d->color_format,
      .depthAttachmentFormat = d->depth_format,
      .stencilAttachmentFormat = d->stencil_format,
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
      .pDepthStencilState = d->ds ? d->ds : &ds_off,
      .pColorBlendState = &cb,
      .layout = layout,
   };
   VkPipeline pipeline = VK_NULL_HANDLE;
   const int64_t t0 = g52_now_ns();
   r = f->CreateGraphicsPipelines(g->device, VK_NULL_HANDLE, 1, &gpci, NULL, &pipeline);
   const int64_t t1 = g52_now_ns();
   f->DestroyShaderModule(g->device, vsm, NULL);
   f->DestroyShaderModule(g->device, fsm, NULL);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateGraphicsPipelines (%s) failed: %s (the Bifrost compile or a "
              "TODO v7 refusal; see the log)\n", g->prog, what, g52_result_str(r));
      return VK_NULL_HANDLE;
   }
   printf("  pipeline %s created in %.2f ms\n", what, (double)(t1 - t0) / 1e6);
   return pipeline;
}

/* ---------------------------------------------------------------------- */
/* Recording and submission                                                */

/* A layout transition of the whole image as a pipeline barrier. */
static inline void
g52_cmd_transition(struct g52 *g, VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspects,
                   VkImageLayout from, VkImageLayout to, VkPipelineStageFlags src_stage,
                   VkAccessFlags src_access, VkPipelineStageFlags dst_stage,
                   VkAccessFlags dst_access)
{
   const VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src_access,
      .dstAccessMask = dst_access,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {aspects, 0, 1, 0, 1},
   };
   g->f.CmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

/* Submits `cmd` and waits for it. Returns the wait's result (VK_SUCCESS,
 * VK_TIMEOUT, or an error, with the fault record printed on a loss). */
static inline VkResult
g52_submit_wait(struct g52 *g, VkCommandBuffer cmd, const char *what)
{
   const struct g52_fns *f = &g->f;
   const VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence = VK_NULL_HANDLE;
   VkResult r = f->CreateFence(g->device, &fci, NULL, &fence);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateFence failed: %s\n", g->prog, g52_result_str(r));
      return r;
   }
   const VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   };
   const int64_t t0 = g52_now_ns();
   r = f->QueueSubmit(g->queue, 1, &si, fence);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s: vkQueueSubmit (%s) failed: %s (kbase JOB_SUBMIT; see the log)\n",
              g->prog, what, g52_result_str(r));
      if (r == VK_ERROR_DEVICE_LOST)
         g52_print_fault(g);
      f->DestroyFence(g->device, fence, NULL);
      return r;
   }
   printf("  submitted %s; waiting up to %llu ms\n", what,
          (unsigned long long)(G52_WAIT_NS / 1000000));
   r = f->WaitForFences(g->device, 1, &fence, VK_TRUE, G52_WAIT_NS);
   const int64_t t1 = g52_now_ns();
   printf("  vkWaitForFences: %s after %.3f ms\n", g52_result_str(r), (double)(t1 - t0) / 1e6);
   if (r == VK_TIMEOUT) {
      printf("FAIL: the fence never signalled: an atom did not complete (a hung job, a tiler "
             "heap that could not grow, or its event was never read). Check dmesg for "
             "mali/kbase lines.\n");
   } else if (r != VK_SUCCESS) {
      printf("FAIL: vkWaitForFences returned %s\n", g52_result_str(r));
      if (r == VK_ERROR_DEVICE_LOST)
         g52_print_fault(g);
   }
   if (r != VK_TIMEOUT)
      f->DestroyFence(g->device, fence, NULL);
   return r;
}

#endif /* G52_HARNESS_H */
