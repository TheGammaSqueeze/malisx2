/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * clear_g52: the smallest Vulkan program that proves the job-manager
 * submission path of this driver on a kbase kernel. It creates an instance
 * and a device on the first physical device, allocates a 64x64 linear RGBA8
 * image in host-visible memory, records one render pass with no draws
 * (loadOp CLEAR, storeOp STORE: one fragment job and one kbase atom),
 * submits it with a fence, waits, maps the memory and checks every pixel
 * against the clear colour (0.25, 0.5, 0.75, 1.0).
 *
 * No Vulkan loader is involved: the driver library is dlopen'ed and every
 * entry point comes from its vk_icdGetInstanceProcAddr, so the device needs
 * no ICD manifest and no system Vulkan. Plain C, links libdl only.
 *
 * Usage: clear_g52 [path/to/libvulkan_malisx2.so]
 *   The path defaults to $MALISX2_ICD, then to CLEAR_G52_DEFAULT_ICD, which
 *   the build sets (the Android build: /data/local/tmp/malisx2/).
 *
 * Exit status: 0 PASS; 1 FAIL (wrong pixels, a timeout or a Vulkan error
 * once the device exists); 2 the program did not get as far as a VkDevice
 * (no driver, no physical device, vkCreateDevice failed).
 *
 * The driver logs through Mesa's logger: on Android to logcat (tag
 * "mesa"), or to stderr with MESA_LOG=file; MESA_LOG_LEVEL=info shows the
 * atom layout and build lines too.
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

#ifndef CLEAR_G52_DEFAULT_ICD
#define CLEAR_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define IMG_W 64
#define IMG_H 64
#define WAIT_NS (5ull * 1000 * 1000 * 1000)

/* The clear colour and its R8G8B8A8_UNORM bytes (round to nearest even,
 * as the driver packs the tile-buffer clear value: 63.75, 127.5, 191.25,
 * 255 -> 64, 128, 191, 255). One LSB of slack is allowed per channel so a
 * round-half-up implementation still passes; the exact bytes are printed. */
static const float clear_rgba[4] = {0.25f, 0.5f, 0.75f, 1.0f};
static const uint8_t expect_rgba[4] = {64, 128, 191, 255};
#define SENTINEL 0xa5

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

/* Every entry point the program uses, by name, so a missing one is
 * reported by name instead of crashing on a NULL call. */
struct fns {
   PFN_vkGetInstanceProcAddr gipa;
   PFN_vkCreateInstance CreateInstance;
   PFN_vkDestroyInstance DestroyInstance;
   PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
   PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
   PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
   PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
   PFN_vkCreateDevice CreateDevice;
   PFN_vkGetDeviceProcAddr GetDeviceProcAddr;

   PFN_vkDestroyDevice DestroyDevice;
   PFN_vkGetDeviceQueue GetDeviceQueue;
   PFN_vkCreateImage CreateImage;
   PFN_vkDestroyImage DestroyImage;
   PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
   PFN_vkGetImageSubresourceLayout GetImageSubresourceLayout;
   PFN_vkAllocateMemory AllocateMemory;
   PFN_vkFreeMemory FreeMemory;
   PFN_vkBindImageMemory BindImageMemory;
   PFN_vkMapMemory MapMemory;
   PFN_vkUnmapMemory UnmapMemory;
   PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges;
   PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges;
   PFN_vkCreateImageView CreateImageView;
   PFN_vkDestroyImageView DestroyImageView;
   PFN_vkCreateCommandPool CreateCommandPool;
   PFN_vkDestroyCommandPool DestroyCommandPool;
   PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
   PFN_vkBeginCommandBuffer BeginCommandBuffer;
   PFN_vkEndCommandBuffer EndCommandBuffer;
   PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
   PFN_vkCmdBeginRendering CmdBeginRendering;
   PFN_vkCmdEndRendering CmdEndRendering;
   PFN_vkCreateFence CreateFence;
   PFN_vkDestroyFence DestroyFence;
   PFN_vkQueueSubmit QueueSubmit;
   PFN_vkWaitForFences WaitForFences;
   PFN_vkGetFenceStatus GetFenceStatus;
   PFN_vkDeviceWaitIdle DeviceWaitIdle;
   PFN_vkGetDeviceFaultInfoEXT GetDeviceFaultInfoEXT; /* optional */
};

static PFN_vkVoidFunction
load_instance_fn(const struct fns *f, VkInstance instance, const char *name, bool *ok)
{
   PFN_vkVoidFunction p = f->gipa(instance, name);
   if (!p) {
      fprintf(stderr, "clear_g52: vk_icdGetInstanceProcAddr(%s) returned NULL\n", name);
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
      fprintf(stderr, "clear_g52: vkGetDeviceProcAddr(%s%s%s) returned NULL\n", name,
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

static const char *
driver_id_str(VkDriverId id)
{
   switch (id) {
   case VK_DRIVER_ID_ARM_PROPRIETARY: return "ARM_PROPRIETARY";
   case VK_DRIVER_ID_MESA_PANVK: return "MESA_PANVK";
   default: return "other";
   }
}

/* ---------------------------------------------------------------------- */

int
main(int argc, char **argv)
{
   /* Line-buffered stdout, so these lines and the driver's log on stderr
    * come out in order through adb and pipes. */
   setvbuf(stdout, NULL, _IOLBF, 0);

   const char *path = argc > 1 ? argv[1] : getenv("MALISX2_ICD");
   if (!path || !*path)
      path = CLEAR_G52_DEFAULT_ICD;

   printf("clear_g52: loading %s\n", path);
   void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
   if (!lib) {
      fprintf(stderr, "clear_g52: dlopen(%s) failed: %s\n", path, dlerror());
      return 2;
   }

   struct fns f;
   memset(&f, 0, sizeof(f));
   f.gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vk_icdGetInstanceProcAddr");
   if (!f.gipa) {
      fprintf(stderr, "clear_g52: %s exports no vk_icdGetInstanceProcAddr: %s\n", path,
              dlerror());
      return 2;
   }
   typedef VkResult(VKAPI_PTR * PFN_negotiate)(uint32_t *);
   PFN_negotiate negotiate =
      (PFN_negotiate)dlsym(lib, "vk_icdNegotiateLoaderICDInterfaceVersion");
   if (negotiate) {
      uint32_t version = 5;
      VkResult r = negotiate(&version);
      printf("  loader/ICD interface version %u (%s)\n", version, result_str(r));
   }

   PFN_vkEnumerateInstanceVersion enum_version =
      (PFN_vkEnumerateInstanceVersion)f.gipa(NULL, "vkEnumerateInstanceVersion");
   if (enum_version) {
      uint32_t v = 0;
      enum_version(&v);
      printf("  instance version %u.%u.%u\n", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v),
             VK_API_VERSION_PATCH(v));
   }

   bool ok = true;
   f.CreateInstance = (PFN_vkCreateInstance)load_instance_fn(&f, NULL, "vkCreateInstance", &ok);
   if (!ok)
      return 2;

   /* apiVersion 1.3: vkCmdBeginRendering is a core 1.3 entry point, and
    * Mesa's runtime hands core entry points out through vkGetDeviceProcAddr
    * only up to the version the application asked for. A 1.1 device must
    * accept a higher apiVersion (it may not return INCOMPATIBLE_DRIVER). */
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "clear_g52",
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
      fprintf(stderr, "clear_g52: vkCreateInstance failed: %s\n", result_str(r));
      return 2;
   }

#define INST(name) f.name = (PFN_vk##name)load_instance_fn(&f, instance, "vk" #name, &ok)
   INST(DestroyInstance);
   INST(EnumeratePhysicalDevices);
   INST(GetPhysicalDeviceProperties2);
   INST(GetPhysicalDeviceMemoryProperties);
   INST(EnumerateDeviceExtensionProperties);
   INST(CreateDevice);
   INST(GetDeviceProcAddr);
#undef INST
   if (!ok)
      return 2;

   uint32_t pdev_count = 0;
   r = f.EnumeratePhysicalDevices(instance, &pdev_count, NULL);
   if (r != VK_SUCCESS || pdev_count == 0) {
      fprintf(stderr,
              "clear_g52: no physical device (%s, count %u): the driver refused /dev/mali0; "
              "see the driver log (kbase VERSION_CHECK, GPU product, atom-layout probe)\n",
              result_str(r), pdev_count);
      return 2;
   }
   VkPhysicalDevice pdevs[8];
   if (pdev_count > 8)
      pdev_count = 8;
   r = f.EnumeratePhysicalDevices(instance, &pdev_count, pdevs);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      fprintf(stderr, "clear_g52: vkEnumeratePhysicalDevices failed: %s\n", result_str(r));
      return 2;
   }
   VkPhysicalDevice pdev = pdevs[0];
   printf("  %u physical device(s); using the first\n", pdev_count);

   VkPhysicalDeviceDriverProperties drv = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &drv,
   };
   f.GetPhysicalDeviceProperties2(pdev, &props2);
   const VkPhysicalDeviceProperties *p = &props2.properties;
   printf("  device: %s\n", p->deviceName);
   printf("  apiVersion %u.%u.%u, driverVersion 0x%08x (%u.%u.%u), vendor 0x%04x, device 0x%08x\n",
          VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion),
          VK_API_VERSION_PATCH(p->apiVersion), p->driverVersion,
          VK_VERSION_MAJOR(p->driverVersion), VK_VERSION_MINOR(p->driverVersion),
          VK_VERSION_PATCH(p->driverVersion), p->vendorID, p->deviceID);
   printf("  driverID %d (%s), driverName \"%s\"\n", drv.driverID, driver_id_str(drv.driverID),
          drv.driverName);
   printf("  driverInfo \"%s\"\n", drv.driverInfo);

   VkPhysicalDeviceMemoryProperties mem_props;
   f.GetPhysicalDeviceMemoryProperties(pdev, &mem_props);
   printf("  %u memory heap(s), %u memory type(s):\n", mem_props.memoryHeapCount,
          mem_props.memoryTypeCount);
   for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
      VkMemoryPropertyFlags fl = mem_props.memoryTypes[i].propertyFlags;
      printf("    type %u: heap %u flags 0x%x%s%s%s%s%s\n", i, mem_props.memoryTypes[i].heapIndex,
             fl, (fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " device-local" : "",
             (fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " host-visible" : "",
             (fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? " host-coherent" : "",
             (fl & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? " host-cached" : "",
             (fl & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) ? " lazily-allocated" : "");
   }

   /* VK_EXT_device_fault, when the driver has it: the fault description
    * after VK_ERROR_DEVICE_LOST names the kbase event code and atom. */
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
      fprintf(stderr,
              "clear_g52: vkCreateDevice failed: %s (the kbase context: VERSION_CHECK, "
              "SET_FLAGS, MEM_EXEC_INIT, MEM_JIT_INIT, the atom-layout probe; see the log)\n",
              result_str(r));
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
   DEV(AllocateMemory);
   DEV(FreeMemory);
   DEV(BindImageMemory);
   DEV(MapMemory);
   DEV(UnmapMemory);
   DEV(FlushMappedMemoryRanges);
   DEV(InvalidateMappedMemoryRanges);
   DEV(CreateImageView);
   DEV(DestroyImageView);
   DEV(CreateCommandPool);
   DEV(DestroyCommandPool);
   DEV(AllocateCommandBuffers);
   DEV(BeginCommandBuffer);
   DEV(EndCommandBuffer);
   DEV(CmdPipelineBarrier);
   DEV(CreateFence);
   DEV(DestroyFence);
   DEV(QueueSubmit);
   DEV(WaitForFences);
   DEV(GetFenceStatus);
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

   /* The image: linear, colour attachment only (a sampled or storage usage
    * would need the texture descriptors, which the v7 build does not pack
    * yet), host-visible memory. */
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
      fprintf(stderr, "clear_g52: vkCreateImage failed: %s\n", result_str(r));
      return 1;
   }

   VkMemoryRequirements reqs;
   f.GetImageMemoryRequirements(device, image, &reqs);
   printf("  image: %ux%u R8G8B8A8_UNORM linear, %llu bytes, alignment %llu, type bits 0x%x\n",
          IMG_W, IMG_H, (unsigned long long)reqs.size, (unsigned long long)reqs.alignment,
          reqs.memoryTypeBits);

   /* Host-visible and host-coherent if there is one, host-visible otherwise
    * (then the readback is invalidated first). */
   int type = -1;
   bool coherent = false;
   for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
      VkMemoryPropertyFlags fl = mem_props.memoryTypes[i].propertyFlags;
      if (!(reqs.memoryTypeBits & (1u << i)) || !(fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         continue;
      const bool c = fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      if (type < 0 || (c && !coherent)) {
         type = (int)i;
         coherent = c;
      }
   }
   if (type < 0) {
      fprintf(stderr, "clear_g52: no host-visible memory type for the image\n");
      return 1;
   }
   printf("  memory type %d (%s)\n", type, coherent ? "host-coherent" : "host-cached, not coherent");

   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)type,
   };
   VkDeviceMemory memory = VK_NULL_HANDLE;
   r = f.AllocateMemory(device, &mai, NULL, &memory);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "clear_g52: vkAllocateMemory failed: %s (kbase MEM_ALLOC; see the log)\n",
              result_str(r));
      return 1;
   }
   r = f.BindImageMemory(device, image, memory, 0);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "clear_g52: vkBindImageMemory failed: %s\n", result_str(r));
      return 1;
   }

   void *map = NULL;
   r = f.MapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &map);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "clear_g52: vkMapMemory failed: %s\n", result_str(r));
      return 1;
   }
   /* A sentinel under the clear, so "nothing was written" and "the wrong
    * value was written" look different. */
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
   printf("  subresource layout: offset %llu, row pitch %llu\n",
          (unsigned long long)layout.offset, (unsigned long long)layout.rowPitch);

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
      fprintf(stderr, "clear_g52: vkCreateImageView failed: %s\n", result_str(r));
      return 1;
   }

   const VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkCommandPool pool = VK_NULL_HANDLE;
   r = f.CreateCommandPool(device, &cpci, NULL, &pool);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "clear_g52: vkCreateCommandPool failed: %s\n", result_str(r));
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
      fprintf(stderr, "clear_g52: vkAllocateCommandBuffers failed: %s\n", result_str(r));
      return 1;
   }

   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   r = f.BeginCommandBuffer(cmd, &bi);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "clear_g52: vkBeginCommandBuffer failed: %s\n", result_str(r));
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
      fprintf(stderr,
              "clear_g52: vkEndCommandBuffer failed: %s (recording failed: command memory, "
              "the framebuffer descriptor or the fragment job; see the log)\n",
              result_str(r));
      return 1;
   }
   printf("  command buffer recorded (one render pass, no draws)\n");

   const VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence = VK_NULL_HANDLE;
   r = f.CreateFence(device, &fci, NULL, &fence);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "clear_g52: vkCreateFence failed: %s\n", result_str(r));
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
      fprintf(stderr, "clear_g52: vkQueueSubmit failed: %s (kbase JOB_SUBMIT; see the log)\n",
              result_str(r));
      if (r == VK_ERROR_DEVICE_LOST)
         print_fault(&f, device);
      return 1;
   }
   printf("  submitted; waiting up to %llu ms for the fence\n",
          (unsigned long long)(WAIT_NS / 1000000));

   r = f.WaitForFences(device, 1, &fence, VK_TRUE, WAIT_NS);
   const int64_t t1 = now_ns();
   printf("  vkWaitForFences: %s after %.3f ms\n", result_str(r), (double)(t1 - t0) / 1e6);
   if (r == VK_TIMEOUT) {
      printf("FAIL: the fence never signalled: the fragment atom did not complete (a hung "
             "job, or its event was never read). Check dmesg for mali/kbase lines.\n");
      /* Do not touch the memory: the GPU may still write it. Let the
       * driver's teardown deal with the atom. */
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
   uint32_t bad = 0, untouched = 0;
   int first_x = -1, first_y = -1;
   uint8_t first_got[4] = {0};
   for (uint32_t y = 0; y < IMG_H; y++) {
      const uint8_t *row = base + (size_t)y * layout.rowPitch;
      for (uint32_t x = 0; x < IMG_W; x++) {
         const uint8_t *px = row + (size_t)x * 4;
         bool okpx = true;
         for (unsigned c = 0; c < 4; c++) {
            const int d = (int)px[c] - (int)expect_rgba[c];
            if (d < -1 || d > 1)
               okpx = false;
         }
         if (okpx)
            continue;
         if (px[0] == SENTINEL && px[1] == SENTINEL && px[2] == SENTINEL && px[3] == SENTINEL)
            untouched++;
         if (!bad) {
            first_x = (int)x;
            first_y = (int)y;
            memcpy(first_got, px, 4);
         }
         bad++;
      }
   }
   printf("  pixel (0,0) = %u %u %u %u, expected %u %u %u %u\n", base[0], base[1], base[2],
          base[3], expect_rgba[0], expect_rgba[1], expect_rgba[2], expect_rgba[3]);

   int status;
   if (bad == 0) {
      printf("PASS: all %u pixels cleared to (%.2f, %.2f, %.2f, %.2f)\n", IMG_W * IMG_H,
             clear_rgba[0], clear_rgba[1], clear_rgba[2], clear_rgba[3]);
      status = 0;
   } else {
      printf("FAIL: %u of %u pixels wrong (%u still hold the 0x%02x sentinel); first at "
             "(%d,%d) = %u %u %u %u, expected %u %u %u %u\n",
             bad, IMG_W * IMG_H, untouched, SENTINEL, first_x, first_y, first_got[0],
             first_got[1], first_got[2], first_got[3], expect_rgba[0], expect_rgba[1],
             expect_rgba[2], expect_rgba[3]);
      status = 1;
   }

   /* Teardown, so the driver's own exit path (POST_TERM, frees) runs too. */
   f.DeviceWaitIdle(device);
   f.UnmapMemory(device, memory);
   f.DestroyFence(device, fence, NULL);
   f.DestroyCommandPool(device, pool, NULL);
   f.DestroyImageView(device, view, NULL);
   f.DestroyImage(device, image, NULL);
   f.FreeMemory(device, memory, NULL);
   f.DestroyDevice(device, NULL);
   f.DestroyInstance(instance, NULL);
   dlclose(lib);
   return status;
}
