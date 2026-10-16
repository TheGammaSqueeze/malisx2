/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * copy_g52: the transfer commands on the Bifrost (v7) path of this driver,
 * all of which run as compute dispatches of the driver's internal shaders
 * (mali_cmd_copy.c, mali_cmd_image.c, mali_cmd_meta_gfx.c): push constants
 * only, global loads and stores. One command buffer records, with barriers
 * between the dependent steps:
 *
 *   1. vkCmdCopyBuffer      A -> B, the whole 64 KiB (16-byte copies);
 *   2. vkCmdFillBuffer      C with 0xdeadbeef;
 *   3. vkCmdUpdateBuffer    E with 100 bytes of pattern (a 4-byte copy from
 *                           command memory);
 *   4. vkCmdCopyBufferToImage  A[0 .. 16 KiB) -> I1, a 64x64 RGBA8 optimal
 *                           (u-interleaved) image: the 4x4 quad copy;
 *   5. vkCmdCopyBufferToImage  P (a 19x11 patch) -> I1 at (9, 5): the ragged
 *                           edge paths;
 *   6. vkCmdCopyImage       I1 -> I2 (tiled to tiled);
 *   7. vkCmdBlitImage       I2 -> I3, 1:1, same format (a copy on v7);
 *   8. vkCmdCopyImageToBuffer  I3 -> D, the whole image, linear rows.
 *
 * Then every buffer is checked on the CPU: B == A, C == fill, E == pattern,
 * D == A with the patch at (9, 5).
 *
 * No Vulkan loader: the driver library is dlopen'ed (see clear_g52.c).
 *
 * Usage: copy_g52 [path/to/libvulkan_malisx2.so]
 *   The path defaults to $MALISX2_ICD, then to COPY_G52_DEFAULT_ICD.
 *
 * Exit status: 0 PASS; 1 FAIL (wrong bytes, a timeout or a Vulkan error
 * once the device exists); 2 the program did not get as far as a VkDevice.
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

#ifndef COPY_G52_DEFAULT_ICD
#define COPY_G52_DEFAULT_ICD "libvulkan_malisx2.so"
#endif

#define BUF_BYTES (64u * 1024u)
#define IMG_W 64
#define IMG_H 64
#define IMG_BYTES (IMG_W * IMG_H * 4u)
#define PATCH_X 9
#define PATCH_Y 5
#define PATCH_W 19
#define PATCH_H 11
#define PATCH_BYTES (PATCH_W * PATCH_H * 4u)
#define UPDATE_BYTES 100u
#define FILL_VALUE 0xdeadbeefu
#define WAIT_NS (5ull * 1000 * 1000 * 1000)

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
   PFN_vkCreateCommandPool CreateCommandPool;
   PFN_vkDestroyCommandPool DestroyCommandPool;
   PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
   PFN_vkBeginCommandBuffer BeginCommandBuffer;
   PFN_vkEndCommandBuffer EndCommandBuffer;
   PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
   PFN_vkCmdCopyBuffer CmdCopyBuffer;
   PFN_vkCmdFillBuffer CmdFillBuffer;
   PFN_vkCmdUpdateBuffer CmdUpdateBuffer;
   PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;
   PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
   PFN_vkCmdCopyImage CmdCopyImage;
   PFN_vkCmdBlitImage CmdBlitImage;
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
      fprintf(stderr, "copy_g52: vk_icdGetInstanceProcAddr(%s) returned NULL\n", name);
      *ok = false;
   }
   return p;
}

static PFN_vkVoidFunction
load_device_fn(const struct fns *f, VkDevice device, const char *name, bool required, bool *ok)
{
   PFN_vkVoidFunction p = f->GetDeviceProcAddr(device, name);
   if (!p && required) {
      fprintf(stderr, "copy_g52: vkGetDeviceProcAddr(%s) returned NULL\n", name);
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
      fprintf(stderr, "copy_g52: vkCreateBuffer (%s) failed: %s\n", what, result_str(r));
      return false;
   }
   VkMemoryRequirements reqs;
   f->GetBufferMemoryRequirements(dev, b->buf, &reqs);
   int type = pick_memory_type(mp, reqs.memoryTypeBits, &b->coherent);
   if (type < 0) {
      fprintf(stderr, "copy_g52: no host-visible memory type for the %s buffer\n", what);
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
      fprintf(stderr, "copy_g52: memory for the %s buffer failed: %s\n", what, result_str(r));
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


/* The patterns: byte i of A is a function of i that never repeats within
 * 64 KiB with period 4 (so swapped words show), the patch is distinct. */
static uint8_t
pattern_a(uint32_t i)
{
   return (uint8_t)(i * 7u + (i >> 8) * 13u + 3u);
}

static uint8_t
pattern_patch(uint32_t i)
{
   return (uint8_t)(0xc0u ^ (i * 5u + (i >> 6)));
}

static uint8_t
pattern_update(uint32_t i)
{
   return (uint8_t)(0x30u + i * 3u);
}

struct himg {
   VkImage img;
   VkDeviceMemory mem;
};

static bool
himg_create(const struct fns *f, VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            const char *what, struct himg *im)
{
   memset(im, 0, sizeof(*im));
   const VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {IMG_W, IMG_H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkResult r = f->CreateImage(dev, &ici, NULL, &im->img);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "copy_g52: vkCreateImage (%s) failed: %s\n", what, result_str(r));
      return false;
   }
   VkMemoryRequirements reqs;
   f->GetImageMemoryRequirements(dev, im->img, &reqs);
   /* Any memory type: nothing reads the images on the CPU. */
   int type = -1;
   for (uint32_t i = 0; i < mp->memoryTypeCount && type < 0; i++)
      if (reqs.memoryTypeBits & (1u << i))
         type = (int)i;
   if (type < 0) {
      fprintf(stderr, "copy_g52: no memory type for the %s image\n", what);
      return false;
   }
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = (uint32_t)type,
   };
   r = f->AllocateMemory(dev, &mai, NULL, &im->mem);
   if (r == VK_SUCCESS)
      r = f->BindImageMemory(dev, im->img, im->mem, 0);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "copy_g52: memory for the %s image failed: %s\n", what, result_str(r));
      return false;
   }
   return true;
}

static void
himg_destroy(const struct fns *f, VkDevice dev, struct himg *im)
{
   if (im->img)
      f->DestroyImage(dev, im->img, NULL);
   if (im->mem)
      f->FreeMemory(dev, im->mem, NULL);
   memset(im, 0, sizeof(*im));
}

static VkImageMemoryBarrier
image_barrier(VkImage img, VkAccessFlags src, VkAccessFlags dst, VkImageLayout from,
              VkImageLayout to)
{
   const VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src,
      .dstAccessMask = dst,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   return b;
}

/* Compare `n` bytes against a pattern; reports the first mismatch. */
static uint32_t
check_bytes(const char *what, const uint8_t *got, uint32_t n, uint8_t (*pattern)(uint32_t),
            const uint8_t *sentinel_fill)
{
   uint32_t bad = 0, untouched = 0;
   int first = -1;
   for (uint32_t i = 0; i < n; i++) {
      if (got[i] == pattern(i))
         continue;
      if (sentinel_fill && got[i] == *sentinel_fill)
         untouched++;
      if (first < 0)
         first = (int)i;
      bad++;
   }
   if (bad)
      printf("  %s: %u of %u bytes wrong (%u still hold the sentinel); first at %d = 0x%02x, "
             "want 0x%02x\n", what, bad, n, untouched, first, got[first],
             pattern((uint32_t)first));
   else
      printf("  %s: %u bytes as expected\n", what, n);
   return bad;
}

/* ---------------------------------------------------------------------- */

int
main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IOLBF, 0);

   const char *path = argc > 1 ? argv[1] : getenv("MALISX2_ICD");
   if (!path || !*path)
      path = COPY_G52_DEFAULT_ICD;

   printf("copy_g52: loading %s\n", path);
   void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
   if (!lib) {
      fprintf(stderr, "copy_g52: dlopen(%s) failed: %s\n", path, dlerror());
      return 2;
   }

   struct fns f;
   memset(&f, 0, sizeof(f));
   f.gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vk_icdGetInstanceProcAddr");
   if (!f.gipa) {
      fprintf(stderr, "copy_g52: %s exports no vk_icdGetInstanceProcAddr: %s\n", path, dlerror());
      return 2;
   }

   bool ok = true;
   f.CreateInstance = (PFN_vkCreateInstance)load_instance_fn(&f, NULL, "vkCreateInstance", &ok);
   if (!ok)
      return 2;

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "copy_g52",
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
      fprintf(stderr, "copy_g52: vkCreateInstance failed: %s\n", result_str(r));
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
      fprintf(stderr, "copy_g52: no physical device (%s, count %u): the driver refused "
              "/dev/mali0; see the driver log\n", result_str(r), pdev_count);
      return 2;
   }
   VkPhysicalDevice pdevs[8];
   if (pdev_count > 8)
      pdev_count = 8;
   r = f.EnumeratePhysicalDevices(instance, &pdev_count, pdevs);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      fprintf(stderr, "copy_g52: vkEnumeratePhysicalDevices failed: %s\n", result_str(r));
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
      fprintf(stderr, "copy_g52: vkCreateDevice failed: %s (see the log)\n", result_str(r));
      return 2;
   }
   printf("  VkDevice created%s\n", have_fault_ext ? " (VK_EXT_device_fault enabled)" : "");

#define DEV(name) f.name = (PFN_vk##name)load_device_fn(&f, device, "vk" #name, true, &ok)
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
   DEV(CreateCommandPool);
   DEV(DestroyCommandPool);
   DEV(AllocateCommandBuffers);
   DEV(BeginCommandBuffer);
   DEV(EndCommandBuffer);
   DEV(CmdPipelineBarrier);
   DEV(CmdCopyBuffer);
   DEV(CmdFillBuffer);
   DEV(CmdUpdateBuffer);
   DEV(CmdCopyBufferToImage);
   DEV(CmdCopyImageToBuffer);
   DEV(CmdCopyImage);
   DEV(CmdBlitImage);
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

   /* Buffers: A source, B copy target, C fill target, D image readback, E
    * update target, P the patch. */
   struct hbuf a, b, c, d, e, pbuf;
   const VkBufferUsageFlags src_use = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
   const VkBufferUsageFlags dst_use = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
   if (!hbuf_create(&f, device, &mem_props, BUF_BYTES, src_use, "A", &a) ||
       !hbuf_create(&f, device, &mem_props, BUF_BYTES, dst_use, "B", &b) ||
       !hbuf_create(&f, device, &mem_props, BUF_BYTES, dst_use, "C", &c) ||
       !hbuf_create(&f, device, &mem_props, IMG_BYTES, dst_use, "D", &d) ||
       !hbuf_create(&f, device, &mem_props, 256, dst_use, "E", &e) ||
       !hbuf_create(&f, device, &mem_props, PATCH_BYTES, src_use, "P", &pbuf))
      return 1;
   uint8_t *am = a.map, *pm = pbuf.map;
   for (uint32_t i = 0; i < BUF_BYTES; i++)
      am[i] = pattern_a(i);
   for (uint32_t i = 0; i < PATCH_BYTES; i++)
      pm[i] = pattern_patch(i);
   const uint8_t sentinel = 0xa5;
   memset(b.map, sentinel, BUF_BYTES);
   memset(c.map, sentinel, BUF_BYTES);
   memset(d.map, sentinel, IMG_BYTES);
   memset(e.map, sentinel, 256);
   hbuf_flush(&f, device, &a, false);
   hbuf_flush(&f, device, &b, false);
   hbuf_flush(&f, device, &c, false);
   hbuf_flush(&f, device, &d, false);
   hbuf_flush(&f, device, &e, false);
   hbuf_flush(&f, device, &pbuf, false);
   printf("  buffers: A, B, C %u bytes, D %u bytes, E 256 bytes, P %u bytes\n", BUF_BYTES,
          IMG_BYTES, PATCH_BYTES);

   struct himg i1, i2, i3;
   if (!himg_create(&f, device, &mem_props, "I1", &i1) ||
       !himg_create(&f, device, &mem_props, "I2", &i2) ||
       !himg_create(&f, device, &mem_props, "I3", &i3))
      return 1;
   printf("  images: 3 x %ux%u R8G8B8A8_UNORM optimal tiling\n", IMG_W, IMG_H);

   const VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkCommandPool pool = VK_NULL_HANDLE;
   r = f.CreateCommandPool(device, &cpci, NULL, &pool);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "copy_g52: vkCreateCommandPool failed: %s\n", result_str(r));
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
      fprintf(stderr, "copy_g52: vkAllocateCommandBuffers failed: %s\n", result_str(r));
      return 1;
   }
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   r = f.BeginCommandBuffer(cmd, &bi);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "copy_g52: vkBeginCommandBuffer failed: %s\n", result_str(r));
      return 1;
   }

   const VkMemoryBarrier host_to_xfer = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
   };
   const VkImageMemoryBarrier to_dst[3] = {
      image_barrier(i1.img, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
      image_barrier(i2.img, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
      image_barrier(i3.img, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
   };
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &host_to_xfer, 0, NULL, 3, to_dst);

   /* 1. A -> B. */
   const VkBufferCopy whole = {0, 0, BUF_BYTES};
   f.CmdCopyBuffer(cmd, a.buf, b.buf, 1, &whole);
   /* 2. Fill C. */
   f.CmdFillBuffer(cmd, c.buf, 0, VK_WHOLE_SIZE, FILL_VALUE);
   /* 3. Update E. */
   uint8_t upd[UPDATE_BYTES];
   for (uint32_t i = 0; i < UPDATE_BYTES; i++)
      upd[i] = pattern_update(i);
   f.CmdUpdateBuffer(cmd, e.buf, 0, UPDATE_BYTES, upd);
   /* 4. A -> I1, the whole image. */
   const VkBufferImageCopy full = {
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageOffset = {0, 0, 0},
      .imageExtent = {IMG_W, IMG_H, 1},
   };
   f.CmdCopyBufferToImage(cmd, a.buf, i1.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &full);
   /* 5. P -> I1 at (PATCH_X, PATCH_Y), after the full copy. */
   const VkMemoryBarrier xfer_to_xfer = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
   };
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                        &xfer_to_xfer, 0, NULL, 0, NULL);
   const VkBufferImageCopy patch = {
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageOffset = {PATCH_X, PATCH_Y, 0},
      .imageExtent = {PATCH_W, PATCH_H, 1},
   };
   f.CmdCopyBufferToImage(cmd, pbuf.buf, i1.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &patch);
   /* 6. I1 -> I2. */
   const VkImageMemoryBarrier i1_to_src = image_barrier(
      i1.img, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                        NULL, 0, NULL, 1, &i1_to_src);
   const VkImageCopy icopy = {
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffset = {0, 0, 0},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffset = {0, 0, 0},
      .extent = {IMG_W, IMG_H, 1},
   };
   f.CmdCopyImage(cmd, i1.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, i2.img,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &icopy);
   /* 7. I2 -> I3, 1:1 blit. */
   const VkImageMemoryBarrier i2_to_src = image_barrier(
      i2.img, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                        NULL, 0, NULL, 1, &i2_to_src);
   const VkImageBlit blit = {
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets = {{0, 0, 0}, {IMG_W, IMG_H, 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets = {{0, 0, 0}, {IMG_W, IMG_H, 1}},
   };
   f.CmdBlitImage(cmd, i2.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, i3.img,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
   /* 8. I3 -> D. */
   const VkImageMemoryBarrier i3_to_src = image_barrier(
      i3.img, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                        NULL, 0, NULL, 1, &i3_to_src);
   f.CmdCopyImageToBuffer(cmd, i3.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d.buf, 1, &full);

   const VkMemoryBarrier xfer_to_host = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                        &xfer_to_host, 0, NULL, 0, NULL);

   r = f.EndCommandBuffer(cmd);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "copy_g52: vkEndCommandBuffer failed: %s (recording failed: a copy "
              "shader, a TODO v7 stub or command memory; see the log)\n", result_str(r));
      return 1;
   }
   printf("  command buffer recorded (8 transfer commands)\n");

   const VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence = VK_NULL_HANDLE;
   r = f.CreateFence(device, &fci, NULL, &fence);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "copy_g52: vkCreateFence failed: %s\n", result_str(r));
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
      fprintf(stderr, "copy_g52: vkQueueSubmit failed: %s (kbase JOB_SUBMIT; see the log)\n",
              result_str(r));
      if (r == VK_ERROR_DEVICE_LOST)
         print_fault(&f, device);
      return 1;
   }
   printf("  submitted; waiting up to %llu ms\n", (unsigned long long)(WAIT_NS / 1000000));

   r = f.WaitForFences(device, 1, &fence, VK_TRUE, WAIT_NS);
   const int64_t t1 = now_ns();
   printf("  vkWaitForFences: %s after %.3f ms\n", result_str(r), (double)(t1 - t0) / 1e6);
   if (r == VK_TIMEOUT) {
      printf("FAIL: the fence never signalled: a compute copy did not complete, or its event "
             "was never read. Check dmesg for mali/kbase lines.\n");
      return 1;
   }
   if (r != VK_SUCCESS) {
      printf("FAIL: vkWaitForFences returned %s\n", result_str(r));
      if (r == VK_ERROR_DEVICE_LOST)
         print_fault(&f, device);
      return 1;
   }

   hbuf_flush(&f, device, &b, true);
   hbuf_flush(&f, device, &c, true);
   hbuf_flush(&f, device, &d, true);
   hbuf_flush(&f, device, &e, true);

   uint32_t bad = 0;
   bad += check_bytes("B (vkCmdCopyBuffer)", b.map, BUF_BYTES, pattern_a, &sentinel);

   {
      const uint32_t *cw = c.map;
      uint32_t cbad = 0;
      int first = -1;
      for (uint32_t i = 0; i < BUF_BYTES / 4; i++) {
         if (cw[i] != FILL_VALUE) {
            if (first < 0)
               first = (int)i;
            cbad++;
         }
      }
      if (cbad)
         printf("  C (vkCmdFillBuffer): %u of %u words wrong; first at %d = 0x%08x, want 0x%08x\n",
                cbad, BUF_BYTES / 4, first, cw[first], FILL_VALUE);
      else
         printf("  C (vkCmdFillBuffer): %u words as expected\n", BUF_BYTES / 4);
      bad += cbad;
   }

   bad += check_bytes("E (vkCmdUpdateBuffer)", e.map, UPDATE_BYTES, pattern_update, &sentinel);
   {
      const uint8_t *em = e.map;
      uint32_t over = 0;
      for (uint32_t i = UPDATE_BYTES; i < 256; i++)
         over += em[i] != sentinel;
      if (over) {
         printf("  E: %u bytes past the update were written\n", over);
         bad += over;
      }
   }

   /* D: A's first 16 KiB with the patch at (PATCH_X, PATCH_Y). */
   {
      const uint8_t *dm = d.map;
      uint32_t dbad = 0, untouched = 0, in_patch_bad = 0;
      int first = -1;
      for (uint32_t y = 0; y < IMG_H; y++) {
         for (uint32_t x = 0; x < IMG_W; x++) {
            for (uint32_t k = 0; k < 4; k++) {
               const uint32_t i = (y * IMG_W + x) * 4 + k;
               uint8_t want;
               const bool patched = x >= PATCH_X && x < PATCH_X + PATCH_W && y >= PATCH_Y &&
                                    y < PATCH_Y + PATCH_H;
               if (patched)
                  want = pattern_patch(((y - PATCH_Y) * PATCH_W + (x - PATCH_X)) * 4 + k);
               else
                  want = pattern_a(i);
               if (dm[i] == want)
                  continue;
               if (dm[i] == sentinel)
                  untouched++;
               if (patched)
                  in_patch_bad++;
               if (first < 0)
                  first = (int)i;
               dbad++;
            }
         }
      }
      if (dbad)
         printf("  D (buffer -> I1 -> I2 -> I3 -> buffer): %u of %u bytes wrong (%u in the "
                "patch, %u still hold the sentinel); first at byte %d (pixel %d,%d) = 0x%02x\n",
                dbad, IMG_BYTES, in_patch_bad, untouched, first, (first / 4) % IMG_W,
                (first / 4) / IMG_W, dm[first]);
      else
         printf("  D (buffer -> I1 -> I2 -> I3 -> buffer): %u bytes as expected, patch "
                "included\n", IMG_BYTES);
      bad += dbad;
   }

   int status;
   if (!bad) {
      printf("PASS: every transfer landed\n");
      status = 0;
   } else {
      printf("FAIL: %u bytes or words wrong in all\n", bad);
      status = 1;
   }

   f.DeviceWaitIdle(device);
   f.DestroyFence(device, fence, NULL);
   f.DestroyCommandPool(device, pool, NULL);
   himg_destroy(&f, device, &i1);
   himg_destroy(&f, device, &i2);
   himg_destroy(&f, device, &i3);
   hbuf_destroy(&f, device, &a);
   hbuf_destroy(&f, device, &b);
   hbuf_destroy(&f, device, &c);
   hbuf_destroy(&f, device, &d);
   hbuf_destroy(&f, device, &e);
   hbuf_destroy(&f, device, &pbuf);
   f.DestroyDevice(device, NULL);
   f.DestroyInstance(instance, NULL);
   dlclose(lib);
   return status;
}
