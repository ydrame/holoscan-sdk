/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/input.h>
#include <linux/kd.h>
#include <linux/major.h>
#include <linux/vt.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

#include "common.h"

enum display_mode {
  DISPLAY_MODE_AUTO = 0,
  DISPLAY_MODE_HEADLESS,
  DISPLAY_MODE_KMS,
#if defined(ENABLE_XCB)
  DISPLAY_MODE_XCB,
#endif
  DISPLAY_MODE_KHR,
};

static enum display_mode display_mode = DISPLAY_MODE_AUTO;
static const char* arg_out_file = "./cube.png";

void failv(const char* format, va_list args) {
  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");
  exit(1);
}

void printflike(1, 2) fail(const char* format, ...) {
  va_list args;

  va_start(args, format);
  failv(format, args);
  va_end(args);
}

void printflike(2, 3) fail_if(int cond, const char* format, ...) {
  va_list args;

  if (!cond) return;

  va_start(args, format);
  failv(format, args);
  va_end(args);
}

static char* __attribute__((returns_nonnull)) xstrdup(const char* s) {
  char* dup = strdup(s);
  if (!dup) {
    fprintf(stderr, "out of memory\n");
    abort();
  }

  return dup;
}

static int find_image_memory(struct vkcube* vc, unsigned allowed) {
  VkMemoryPropertyFlags flags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | (vc->protected_ ? VK_MEMORY_PROPERTY_PROTECTED_BIT : 0);

  for (unsigned i = 0; (1u << i) <= allowed && i <= vc->memory_properties.memoryTypeCount; ++i) {
    if ((allowed & (1u << i)) && (vc->memory_properties.memoryTypes[i].propertyFlags & flags))
      return i;
  }
  return -1;
}

static void init_vk(struct vkcube* vc, const char* extension) {
  VkResult res = vkCreateInstance(
      &(VkInstanceCreateInfo){
          .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
          .pApplicationInfo =
              &(VkApplicationInfo){
                  .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                  .pApplicationName = "vkcube",
                  .apiVersion = VK_MAKE_VERSION(1, 1, 0),
              },
          .enabledExtensionCount = extension ? 2 : 0,
          .ppEnabledExtensionNames =
              (const char* [2]){
                  VK_KHR_SURFACE_EXTENSION_NAME,
                  extension,
              },
      },
      NULL,
      &vc->instance);
  fail_if(res != VK_SUCCESS, "Failed to create Vulkan instance.\n");

  uint32_t count;
  res = vkEnumeratePhysicalDevices(vc->instance, &count, NULL);
  fail_if(res != VK_SUCCESS || count == 0, "No Vulkan devices found.\n");
  VkPhysicalDevice pd[count];
  vkEnumeratePhysicalDevices(vc->instance, &count, pd);
  vc->physical_device = pd[0];
  printf("%d physical devices\n", count);

  VkPhysicalDeviceProtectedMemoryFeatures protected_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES,
  };
  VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &protected_features,
  };
  vkGetPhysicalDeviceFeatures2(vc->physical_device, &features);

  if (protected_chain && !protected_features.protectedMemory)
    printf("Requested protected memory but not supported by device, dropping...\n");
  vc->protected_ = protected_chain && protected_features.protectedMemory;

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(vc->physical_device, &properties);
  printf("vendor id %04x, device name %s\n", properties.vendorID, properties.deviceName);

  vkGetPhysicalDeviceMemoryProperties(vc->physical_device, &vc->memory_properties);

  vkGetPhysicalDeviceQueueFamilyProperties(vc->physical_device, &count, NULL);
  assert(count > 0);
  VkQueueFamilyProperties props[count];
  vkGetPhysicalDeviceQueueFamilyProperties(vc->physical_device, &count, props);
  assert(props[0].queueFlags & VK_QUEUE_GRAPHICS_BIT);

  vkCreateDevice(vc->physical_device,
                 &(VkDeviceCreateInfo){
                     .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                     .queueCreateInfoCount = 1,
                     .pQueueCreateInfos =
                         &(VkDeviceQueueCreateInfo){
                             .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                             .queueFamilyIndex = 0,
                             .queueCount = 1,
                             .flags = vc->protected_ ? VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT : 0,
                             .pQueuePriorities = (float[]){1.0f},
                         },
                     .enabledExtensionCount = 1,
                     .ppEnabledExtensionNames =
                         (const char* const[]){
                             VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                         },
                 },
                 NULL,
                 &vc->device);

  vkGetDeviceQueue2(vc->device,
                    &(VkDeviceQueueInfo2){
                        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
                        .flags = vc->protected_ ? VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT : 0,
                        .queueFamilyIndex = 0,
                        .queueIndex = 0,
                    },
                    &vc->queue);
}

static void init_vk_objects(struct vkcube* vc) {
  vkCreateRenderPass(
      vc->device,
      &(VkRenderPassCreateInfo){
          .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
          .attachmentCount = 1,
          .pAttachments = (VkAttachmentDescription[]){{
              .format = vc->image_format,
              .samples = 1,
              .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
              .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
              .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
              .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
          }},
          .subpassCount = 1,
          .pSubpasses = (VkSubpassDescription[]){{
              .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
              .inputAttachmentCount = 0,
              .colorAttachmentCount = 1,
              .pColorAttachments =
                  (VkAttachmentReference[]){
                      {.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
              .pResolveAttachments =
                  (VkAttachmentReference[]){{.attachment = VK_ATTACHMENT_UNUSED,
                                             .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
              .pDepthStencilAttachment = NULL,
              .preserveAttachmentCount = 0,
              .pPreserveAttachments = NULL,
          }},
          .dependencyCount = 0},
      NULL,
      &vc->render_pass);

  vc->model.init(vc);

  vkCreateCommandPool(vc->device,
                      &(const VkCommandPoolCreateInfo){
                          .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                          .queueFamilyIndex = 0,
                          .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                                   (vc->protected_ ? VK_COMMAND_POOL_CREATE_PROTECTED_BIT : 0)},
                      NULL,
                      &vc->cmd_pool);

  vkCreateSemaphore(vc->device,
                    &(VkSemaphoreCreateInfo){
                        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                    },
                    NULL,
                    &vc->semaphore);
}

static void init_buffer(struct vkcube* vc, struct vkcube_buffer* b) {
  vkCreateImageView(vc->device,
                    &(VkImageViewCreateInfo){
                        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        .image = b->image,
                        .viewType = VK_IMAGE_VIEW_TYPE_2D,
                        .format = vc->image_format,
                        .components =
                            {
                                .r = VK_COMPONENT_SWIZZLE_R,
                                .g = VK_COMPONENT_SWIZZLE_G,
                                .b = VK_COMPONENT_SWIZZLE_B,
                                .a = VK_COMPONENT_SWIZZLE_A,
                            },
                        .subresourceRange =
                            {
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                            },
                    },
                    NULL,
                    &b->view);

  vkCreateFramebuffer(vc->device,
                      &(VkFramebufferCreateInfo){.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                                 .renderPass = vc->render_pass,
                                                 .attachmentCount = 1,
                                                 .pAttachments = &b->view,
                                                 .width = vc->width,
                                                 .height = vc->height,
                                                 .layers = 1},
                      NULL,
                      &b->framebuffer);

  vkCreateFence(vc->device,
                &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                     .flags = VK_FENCE_CREATE_SIGNALED_BIT},
                NULL,
                &b->fence);

  vkAllocateCommandBuffers(vc->device,
                           &(VkCommandBufferAllocateInfo){
                               .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                               .commandPool = vc->cmd_pool,
                               .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                               .commandBufferCount = 1,
                           },
                           &b->cmd_buffer);
}

/* Headless code - write one frame to png */

// Return -1 on failure.
static int init_headless(struct vkcube* vc) {
  init_vk(vc, NULL);
  vc->image_format = VK_FORMAT_B8G8R8A8_SRGB;
  init_vk_objects(vc);

  struct vkcube_buffer* b = &vc->buffers[0];

  vkCreateImage(vc->device,
                &(VkImageCreateInfo){
                    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    .imageType = VK_IMAGE_TYPE_2D,
                    .format = vc->image_format,
                    .extent = {.width = vc->width, .height = vc->height, .depth = 1},
                    .mipLevels = 1,
                    .arrayLayers = 1,
                    .samples = 1,
                    .tiling = VK_IMAGE_TILING_LINEAR,
                    .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    .flags = vc->protected_ ? VK_IMAGE_CREATE_PROTECTED_BIT : 0,
                },
                NULL,
                &b->image);

  VkMemoryRequirements requirements;
  vkGetImageMemoryRequirements(vc->device, b->image, &requirements);

  vkAllocateMemory(vc->device,
                   &(VkMemoryAllocateInfo){
                       .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                       .allocationSize = requirements.size,
                       .memoryTypeIndex = find_image_memory(vc, requirements.memoryTypeBits),
                   },
                   NULL,
                   &b->mem);

  vkBindImageMemory(vc->device, b->image, b->mem, 0);

  b->stride = vc->width * 4;

  init_buffer(vc, &vc->buffers[0]);

  return 0;
}

#ifdef HAVE_VULKAN_INTEL_H

/* KMS display code - render to kernel modesetting fb */

#include <vulkan/vulkan_intel.h>

static struct termios save_tio;

static void restore_vt(void) {
  struct vt_mode mode = {.mode = VT_AUTO};
  ioctl(STDIN_FILENO, VT_SETMODE, &mode);

  tcsetattr(STDIN_FILENO, TCSANOW, &save_tio);
  ioctl(STDIN_FILENO, KDSETMODE, KD_TEXT);
}

static void handle_signal(int sig) {
  restore_vt();
}

static int init_vt(struct vkcube* vc) {
  struct termios tio;
  struct stat buf;
  int ret;

  /* First, save term io setting so we can restore properly. */
  tcgetattr(STDIN_FILENO, &save_tio);

  /* Make sure we're on a vt. */
  ret = fstat(STDIN_FILENO, &buf);
  fail_if(ret == -1, "failed to stat stdin\n");

  if (major(buf.st_rdev) != TTY_MAJOR) {
    fprintf(stderr, "stdin not a vt, running in no-display mode\n");
    return -1;
  }

  atexit(restore_vt);

  /* Set console input to raw mode. */
  tio = save_tio;
  tio.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &tio);

  /* Restore console on SIGINT and friends. */
  struct sigaction act = {.sa_handler = handle_signal, .sa_flags = SA_RESETHAND};
  sigaction(SIGINT, &act, NULL);
  sigaction(SIGSEGV, &act, NULL);
  sigaction(SIGABRT, &act, NULL);

  /* We don't drop drm master, so block VT switching while we're
   * running. Otherwise, switching to X on another VT will crash X when it
   * fails to get drm master. */
  struct vt_mode mode = {.mode = VT_PROCESS, .relsig = 0, .acqsig = 0};
  ret = ioctl(STDIN_FILENO, VT_SETMODE, &mode);
  fail_if(ret == -1, "failed to take control of vt handling\n");

  /* Set KD_GRAPHICS to disable fbcon while we render. */
  ret = ioctl(STDIN_FILENO, KDSETMODE, KD_GRAPHICS);
  fail_if(ret == -1, "failed to switch console to graphics mode\n");

  return 0;
}

// Return -1 on failure.
static int init_kms(struct vkcube* vc) {
  drmModeRes* resources;
  drmModeConnector* connector;
  drmModeEncoder* encoder;
  int i;

  if (init_vt(vc) == -1) return -1;

  vc->fd = open("/dev/dri/card0", O_RDWR);
  fail_if(vc->fd == -1, "failed to open /dev/dri/card0\n");

  /* Get KMS resources and find the first active connecter. We'll use that
     connector and the crtc driving it in the mode it's currently running. */
  resources = drmModeGetResources(vc->fd);
  fail_if(!resources, "drmModeGetResources failed: %s\n", strerror(errno));

  for (i = 0; i < resources->count_connectors; i++) {
    connector = drmModeGetConnector(vc->fd, resources->connectors[i]);
    if (connector->connection == DRM_MODE_CONNECTED) break;
    drmModeFreeConnector(connector);
    connector = NULL;
  }

  fail_if(!connector, "no connected connector!\n");
  encoder = drmModeGetEncoder(vc->fd, connector->encoder_id);
  fail_if(!encoder, "failed to get encoder\n");
  vc->crtc = drmModeGetCrtc(vc->fd, encoder->crtc_id);
  fail_if(!vc->crtc, "failed to get crtc\n");
  printf("mode info: hdisplay %d, vdisplay %d\n", vc->crtc->mode.hdisplay, vc->crtc->mode.vdisplay);

  vc->connector = connector;
  vc->width = vc->crtc->mode.hdisplay;
  vc->height = vc->crtc->mode.vdisplay;

  vc->gbm_device = gbm_create_device(vc->fd);

  init_vk(vc, NULL);
  vc->image_format = VK_FORMAT_R8G8B8A8_SRGB;
  init_vk_objects(vc);

  PFN_vkCreateDmaBufImageINTEL create_dma_buf_image =
      (PFN_vkCreateDmaBufImageINTEL)vkGetDeviceProcAddr(vc->device, "vkCreateDmaBufImageINTEL");

  for (uint32_t i = 0; i < 2; i++) {
    struct vkcube_buffer* b = &vc->buffers[i];
    int fd, stride, ret;

    b->gbm_bo = gbm_bo_create(
        vc->gbm_device, vc->width, vc->height, GBM_FORMAT_XRGB8888, GBM_BO_USE_SCANOUT);

    fd = gbm_bo_get_fd(b->gbm_bo);
    stride = gbm_bo_get_stride(b->gbm_bo);
    create_dma_buf_image(
        vc->device,
        &(VkDmaBufImageCreateInfo){.sType = VK_STRUCTURE_TYPE_DMA_BUF_IMAGE_CREATE_INFO_INTEL,
                                   .fd = fd,
                                   .format = vc->image_format,
                                   .extent = {vc->width, vc->height, 1},
                                   .strideInBytes = stride},
        NULL,
        &b->mem,
        &b->image);
    close(fd);

    b->stride = gbm_bo_get_stride(b->gbm_bo);
    uint32_t bo_handles[4] = {
        gbm_bo_get_handle(b->gbm_bo).s32,
    };
    uint32_t pitches[4] = {
        stride,
    };
    uint32_t offsets[4] = {
        0,
    };
    ret = drmModeAddFB2(vc->fd,
                        vc->width,
                        vc->height,
                        DRM_FORMAT_XRGB8888,
                        bo_handles,
                        pitches,
                        offsets,
                        &b->fb,
                        0);
    fail_if(ret == -1, "addfb2 failed\n");

    init_buffer(vc, b);
  }

  return 0;
}

static void page_flip_handler(int fd, unsigned int frame, unsigned int sec, unsigned int usec,
                              void* data) {}

static void renderloop_vt(struct vkcube* vc) {
  int len, ret;
  char buf[16];
  struct pollfd pfd[2];
  struct vkcube_buffer* b;

  pfd[0].fd = STDIN_FILENO;
  pfd[0].events = POLLIN;
  pfd[1].fd = vc->fd;
  pfd[1].events = POLLIN;

  drmEventContext evctx = {
      .version = 2,
      .page_flip_handler = page_flip_handler,
  };

  ret = drmModeSetCrtc(vc->fd,
                       vc->crtc->crtc_id,
                       vc->buffers[0].fb,
                       0,
                       0,
                       &vc->connector->connector_id,
                       1,
                       &vc->crtc->mode);
  fail_if(ret < 0, "modeset failed: %m\n");

  ret =
      drmModePageFlip(vc->fd, vc->crtc->crtc_id, vc->buffers[0].fb, DRM_MODE_PAGE_FLIP_EVENT, NULL);
  fail_if(ret < 0, "pageflip failed: %m\n");

  while (1) {
    ret = poll(pfd, 2, -1);
    fail_if(ret == -1, "poll failed\n");
    if (pfd[0].revents & POLLIN) {
      len = read(STDIN_FILENO, buf, sizeof(buf));
      switch (buf[0]) {
        case 'q':
          return;
        case '\e':
          if (len == 1) return;
      }
    }
    if (pfd[1].revents & POLLIN) {
      drmHandleEvent(vc->fd, &evctx);
      b = &vc->buffers[vc->current & 1];
      vc->model.render(vc, b, false);

      ret = drmModePageFlip(vc->fd, vc->crtc->crtc_id, b->fb, DRM_MODE_PAGE_FLIP_EVENT, NULL);
      fail_if(ret < 0, "pageflip failed: %m\n");
      vc->current++;
    }
  }
}

#else

static int init_kms(struct vkcube* vc) {
  return -1;
}

static void renderloop_vt(struct vkcube* vc) {}

#endif

#if defined(ENABLE_XCB) || defined(ENABLE_WAYLAND)

static VkFormat choose_surface_format(struct vkcube* vc) {
  uint32_t num_formats = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(vc->physical_device, vc->surface, &num_formats, NULL);
  assert(num_formats > 0);

  VkSurfaceFormatKHR formats[num_formats];

  vkGetPhysicalDeviceSurfaceFormatsKHR(vc->physical_device, vc->surface, &num_formats, formats);

  VkFormat format = VK_FORMAT_UNDEFINED;
  for (int i = 0; i < num_formats; i++) {
    switch (formats[i].format) {
      case VK_FORMAT_R8G8B8A8_SRGB:
      case VK_FORMAT_B8G8R8A8_SRGB:
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_B8G8R8A8_UNORM:
        /* These formats are all fine */
        format = formats[i].format;
        break;
      case VK_FORMAT_R8G8B8_SRGB:
      case VK_FORMAT_B8G8R8_SRGB:
      case VK_FORMAT_R8G8B8_UNORM:
      case VK_FORMAT_R5G6B5_UNORM_PACK16:
      case VK_FORMAT_B5G6R5_UNORM_PACK16:
        /* We would like to support these but they don't seem to work. */
      default:
        continue;
    }
  }

  assert(format != VK_FORMAT_UNDEFINED);

  return format;
}

#endif

static void create_swapchain(struct vkcube* vc) {
  VkSurfaceCapabilitiesKHR surface_caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vc->physical_device, vc->surface, &surface_caps);
  assert(surface_caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);

  VkBool32 supported;
  vkGetPhysicalDeviceSurfaceSupportKHR(vc->physical_device, 0, vc->surface, &supported);
  assert(supported);

  uint32_t count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(vc->physical_device, vc->surface, &count, NULL);
  VkPresentModeKHR present_modes[count];
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      vc->physical_device, vc->surface, &count, present_modes);
  int i;
  VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
  for (i = 0; i < count; i++) {
    if (present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
      present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
      break;
    }
  }

  uint32_t minImageCount = 2;
  if (minImageCount < surface_caps.minImageCount) {
    if (surface_caps.minImageCount > MAX_NUM_IMAGES)
      fail("surface_caps.minImageCount is too large (is: %d, max: %d)",
           surface_caps.minImageCount,
           MAX_NUM_IMAGES);
    minImageCount = surface_caps.minImageCount;
  }

  if (surface_caps.maxImageCount > 0 && minImageCount > surface_caps.maxImageCount) {
    minImageCount = surface_caps.maxImageCount;
  }

  vkCreateSwapchainKHR(vc->device,
                       &(VkSwapchainCreateInfoKHR){
                           .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                           .flags = vc->protected_ ? VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR : 0,
                           .surface = vc->surface,
                           .minImageCount = minImageCount,
                           .imageFormat = vc->image_format,
                           .imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR,
                           .imageExtent = {vc->width, vc->height},
                           .imageArrayLayers = 1,
                           .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                           .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
                           .queueFamilyIndexCount = 1,
                           .pQueueFamilyIndices = (uint32_t[]){0},
                           .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                           .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                           .presentMode = present_mode,
                       },
                       NULL,
                       &vc->swap_chain);

  vkGetSwapchainImagesKHR(vc->device, vc->swap_chain, &vc->image_count, NULL);
  assert(vc->image_count > 0);
  VkImage swap_chain_images[vc->image_count];
  vkGetSwapchainImagesKHR(vc->device, vc->swap_chain, &vc->image_count, swap_chain_images);

  assert(vc->image_count <= MAX_NUM_IMAGES);
  for (uint32_t i = 0; i < vc->image_count; i++) {
    vc->buffers[i].image = swap_chain_images[i];
    init_buffer(vc, &vc->buffers[i]);
  }
}

/* XCB display code - render to X window */
#if defined(ENABLE_XCB)

static xcb_atom_t get_atom(struct xcb_connection_t* conn, const char* name) {
  xcb_intern_atom_cookie_t cookie;
  xcb_intern_atom_reply_t* reply;
  xcb_atom_t atom;

  cookie = xcb_intern_atom(conn, 0, strlen(name), name);
  reply = xcb_intern_atom_reply(conn, cookie, NULL);
  if (reply) atom = reply->atom;
  else
    atom = XCB_NONE;

  free(reply);
  return atom;
}

// Return -1 on failure.
static int init_xcb(struct vkcube* vc) {
  xcb_screen_iterator_t iter;
  static const char title[] = "Vulkan Cube";

  vc->xcb.conn = xcb_connect(0, 0);
  if (xcb_connection_has_error(vc->xcb.conn)) return -1;

  vc->xcb.window = xcb_generate_id(vc->xcb.conn);

  uint32_t window_values[] = {XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_STRUCTURE_NOTIFY |
                              XCB_EVENT_MASK_KEY_PRESS};

  iter = xcb_setup_roots_iterator(xcb_get_setup(vc->xcb.conn));

  xcb_create_window(vc->xcb.conn,
                    XCB_COPY_FROM_PARENT,
                    vc->xcb.window,
                    iter.data->root,
                    0,
                    0,
                    vc->width,
                    vc->height,
                    0,
                    XCB_WINDOW_CLASS_INPUT_OUTPUT,
                    iter.data->root_visual,
                    XCB_CW_EVENT_MASK,
                    window_values);

  vc->xcb.atom_wm_protocols = get_atom(vc->xcb.conn, "WM_PROTOCOLS");
  vc->xcb.atom_wm_delete_window = get_atom(vc->xcb.conn, "WM_DELETE_WINDOW");
  xcb_change_property(vc->xcb.conn,
                      XCB_PROP_MODE_REPLACE,
                      vc->xcb.window,
                      vc->xcb.atom_wm_protocols,
                      XCB_ATOM_ATOM,
                      32,
                      1,
                      &vc->xcb.atom_wm_delete_window);

  xcb_change_property(vc->xcb.conn,
                      XCB_PROP_MODE_REPLACE,
                      vc->xcb.window,
                      get_atom(vc->xcb.conn, "_NET_WM_NAME"),
                      get_atom(vc->xcb.conn, "UTF8_STRING"),
                      8,  // sizeof(char),
                      strlen(title),
                      title);

  xcb_map_window(vc->xcb.conn, vc->xcb.window);

  xcb_flush(vc->xcb.conn);

  init_vk(vc, VK_KHR_XCB_SURFACE_EXTENSION_NAME);

  PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR get_xcb_presentation_support =
      (PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR)vkGetInstanceProcAddr(
          vc->instance, "vkGetPhysicalDeviceXcbPresentationSupportKHR");
  PFN_vkCreateXcbSurfaceKHR create_xcb_surface =
      (PFN_vkCreateXcbSurfaceKHR)vkGetInstanceProcAddr(vc->instance, "vkCreateXcbSurfaceKHR");

  if (!get_xcb_presentation_support(vc->physical_device, 0, vc->xcb.conn, iter.data->root_visual)) {
    fail("Vulkan not supported on given X window");
  }

  create_xcb_surface(vc->instance,
                     &(VkXcbSurfaceCreateInfoKHR){
                         .sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                         .connection = vc->xcb.conn,
                         .window = vc->xcb.window,
                     },
                     NULL,
                     &vc->surface);

  vc->image_format = choose_surface_format(vc);

  init_vk_objects(vc);

  vc->image_count = 0;

  return 0;
}

static void schedule_xcb_repaint(struct vkcube* vc) {
  xcb_client_message_event_t client_message;

  client_message.response_type = XCB_CLIENT_MESSAGE;
  client_message.format = 32;
  client_message.window = vc->xcb.window;
  client_message.type = XCB_ATOM_NOTICE;

  xcb_send_event(vc->xcb.conn, 0, vc->xcb.window, 0, (char*)&client_message);
  xcb_flush(vc->xcb.conn);
}

static bool renderloop_xcb(struct vkcube* vc) {
  xcb_generic_event_t* event;
  xcb_key_press_event_t* key_press;
  xcb_client_message_event_t* client_message;
  xcb_configure_notify_event_t* configure;

  // while (1) {
  bool repaint = false;
  event = xcb_wait_for_event(vc->xcb.conn);
  while (event) {
    switch (event->response_type & 0x7f) {
      case XCB_CLIENT_MESSAGE:
        client_message = (xcb_client_message_event_t*)event;
        if (client_message->window != vc->xcb.window) break;

        if (client_message->type == vc->xcb.atom_wm_protocols &&
            client_message->data.data32[0] == vc->xcb.atom_wm_delete_window) {
          // exit(0);
          return true;
        }

        if (client_message->type == XCB_ATOM_NOTICE) repaint = true;
        break;

      case XCB_CONFIGURE_NOTIFY:
        configure = (xcb_configure_notify_event_t*)event;
        if (vc->width != configure->width || vc->height != configure->height) {
          if (vc->image_count > 0) {
            vkDestroySwapchainKHR(vc->device, vc->swap_chain, NULL);
            vc->image_count = 0;
          }

          vc->width = configure->width;
          vc->height = configure->height;
        }
        break;

      case XCB_EXPOSE:
        schedule_xcb_repaint(vc);
        break;
    }
    free(event);

    event = xcb_poll_for_event(vc->xcb.conn);
  }

    if (repaint) {
      if (vc->image_count == 0) create_swapchain(vc);

      uint32_t index;
      VkResult result;
      result = vkAcquireNextImageKHR(
          vc->device, vc->swap_chain, 60, vc->semaphore, VK_NULL_HANDLE, &index);
      switch (result) {
        case VK_SUCCESS:
          break;
        case VK_NOT_READY:             /* try later */
        case VK_TIMEOUT:               /* try later */
        case VK_ERROR_OUT_OF_DATE_KHR: /* handled by native events */
          schedule_xcb_repaint(vc);
          // continue;
          return false;
        default:
          return false;
      }

      assert(index <= MAX_NUM_IMAGES);
      vc->model.render(vc, &vc->buffers[index], true);

      vkQueuePresentKHR(vc->queue,
                        &(VkPresentInfoKHR){
                            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                            .swapchainCount = 1,
                            .pSwapchains =
                                (VkSwapchainKHR[]){
                                    vc->swap_chain,
                                },
                            .pImageIndices =
                                (uint32_t[]){
                                    index,
                                },
                            .pResults = &result,
                        });

      vkQueueWaitIdle(vc->queue);

      schedule_xcb_repaint(vc);
    }

    xcb_flush(vc->xcb.conn);
    // }
    return false;
}
#endif

static int display_idx = -1;
static int display_mode_idx = -1;
static int display_plane_idx = -1;

// Return -1 on failure.
static int init_khr(struct vkcube* vc) {
  init_vk(vc, VK_KHR_DISPLAY_EXTENSION_NAME);
  vc->image_format = VK_FORMAT_B8G8R8A8_SRGB;
  init_vk_objects(vc);

  /* */
  uint32_t display_count = 0;
  vkGetPhysicalDeviceDisplayPropertiesKHR(vc->physical_device, &display_count, NULL);
  if (!display_count) {
    fprintf(stderr, "No available display\n");
    return -1;
  }

  VkDisplayPropertiesKHR* displays =
      (VkDisplayPropertiesKHR*)malloc(display_count * sizeof(*displays));
  vkGetPhysicalDeviceDisplayPropertiesKHR(vc->physical_device, &display_count, displays);

  if (display_idx < 0) {
    for (uint32_t i = 0; i < display_count; i++) {
      fprintf(stdout, "display [%i]:\n", i);
      fprintf(stdout, "   name: %s\n", displays[i].displayName);
      fprintf(stdout,
              "   physical dimensions: %ux%u\n",
              displays[i].physicalDimensions.width,
              displays[i].physicalDimensions.height);
      fprintf(stdout,
              "   physical resolution: %ux%u\n",
              displays[i].physicalResolution.width,
              displays[i].physicalResolution.height);
      fprintf(stdout, "   plane reorder: %s\n", displays[i].planeReorderPossible ? "yes" : "no");
      fprintf(stdout, "   persistent content: %s\n", displays[i].persistentContent ? "yes" : "no");
    }
    free(displays);
    return -1;
  } else if (display_idx >= display_count) {
    fprintf(stderr, "Invalid display index %i/%i\n", display_idx, display_count);
    free(displays);
    return -1;
  }

  /* */
  uint32_t mode_count = 0;
  vkGetDisplayModePropertiesKHR(
      vc->physical_device, displays[display_idx].display, &mode_count, NULL);
  if (!mode_count) {
    fprintf(stderr,
            "Not mode available for display %i (%s)\n",
            display_idx,
            displays[display_idx].displayName);
    free(displays);
    return -1;
  }
  VkDisplayModePropertiesKHR* modes =
      (VkDisplayModePropertiesKHR*)malloc(mode_count * sizeof(*modes));
  vkGetDisplayModePropertiesKHR(
      vc->physical_device, displays[display_idx].display, &mode_count, modes);
  if (display_mode_idx < 0) {
    fprintf(stdout, "display [%i] (%s) modes:\n", display_idx, displays[display_idx].displayName);
    for (uint32_t i = 0; i < mode_count; i++) {
      fprintf(stdout, "mode [%i]:\n", i);
      fprintf(stdout,
              "   visible region: %ux%u\n",
              modes[i].parameters.visibleRegion.width,
              modes[i].parameters.visibleRegion.height);
      fprintf(stdout, "   refresh rate: %u\n", modes[i].parameters.refreshRate);
    }
    free(displays);
    free(modes);
    return -1;
  } else if (display_mode_idx >= mode_count) {
    fprintf(stderr, "Invalid mode index %i/%i\n", display_mode_idx, mode_count);
    free(displays);
    free(modes);
    return -1;
  }

  /* */
  uint32_t plane_count = 0;
  vkGetPhysicalDeviceDisplayPlanePropertiesKHR(vc->physical_device, &plane_count, NULL);
  if (!plane_count) {
    fprintf(stderr,
            "Not plane available for display %i (%s)\n",
            display_idx,
            displays[display_idx].displayName);
    free(displays);
    free(modes);
    return -1;
  }

  VkDisplayPlanePropertiesKHR* planes =
      (VkDisplayPlanePropertiesKHR*)malloc(plane_count * sizeof(*planes));
  vkGetPhysicalDeviceDisplayPlanePropertiesKHR(vc->physical_device, &plane_count, planes);
  if (display_plane_idx < 0) {
    for (uint32_t i = 0; i < plane_count; i++) {
      fprintf(stdout,
              "display [%i] (%s) plane [%i]\n",
              display_idx,
              displays[display_idx].displayName,
              i);
      fprintf(stdout, "   current stack index: %u\n", planes[i].currentStackIndex);
      fprintf(stdout, "   displays supported:");
      uint32_t supported_display_count = 0;
      vkGetDisplayPlaneSupportedDisplaysKHR(vc->physical_device, i, &supported_display_count, NULL);
      VkDisplayKHR* supported_displays =
          (VkDisplayKHR*)malloc(supported_display_count * sizeof(*supported_displays));
      vkGetDisplayPlaneSupportedDisplaysKHR(
          vc->physical_device, i, &supported_display_count, supported_displays);
      for (uint32_t j = 0; j < supported_display_count; j++) {
        for (uint32_t k = 0; k < display_count; k++) {
          if (displays[k].display == supported_displays[j]) {
            fprintf(stdout, " %u", k);
            break;
          }
        }
      }
      fprintf(stdout, "\n");

      VkDisplayPlaneCapabilitiesKHR plane_caps;
      vkGetDisplayPlaneCapabilitiesKHR(
          vc->physical_device, modes[display_mode_idx].displayMode, i, &plane_caps);
      fprintf(stdout,
              "   src pos: %ux%u -> %ux%u\n",
              plane_caps.minSrcPosition.x,
              plane_caps.minSrcPosition.y,
              plane_caps.maxSrcPosition.x,
              plane_caps.maxSrcPosition.y);
      fprintf(stdout,
              "   src size: %ux%u -> %ux%u\n",
              plane_caps.minSrcExtent.width,
              plane_caps.minSrcExtent.height,
              plane_caps.maxSrcExtent.width,
              plane_caps.maxSrcExtent.height);
      fprintf(stdout,
              "   dst pos: %ux%u -> %ux%u\n",
              plane_caps.minDstPosition.x,
              plane_caps.minDstPosition.y,
              plane_caps.maxDstPosition.x,
              plane_caps.maxDstPosition.y);
      fprintf(stdout,
              "   dst size: %ux%u -> %ux%u\n",
              plane_caps.minDstExtent.width,
              plane_caps.minDstExtent.height,
              plane_caps.maxDstExtent.width,
              plane_caps.maxDstExtent.height);
    }
    free(displays);
    free(modes);
    free(planes);
    return -1;
  } else if (display_plane_idx >= plane_count) {
    fprintf(stderr, "Invalid plane index %i/%i\n", display_plane_idx, plane_count);
    free(displays);
    free(modes);
    free(planes);
    return -1;
  }

  VkDisplayModeCreateInfoKHR display_mode_create_info = {
      .sType = VK_STRUCTURE_TYPE_DISPLAY_MODE_CREATE_INFO_KHR,
      .parameters = modes[display_mode_idx].parameters,
  };
  VkResult result = vkCreateDisplayModeKHR(vc->physical_device,
                                           displays[display_idx].display,
                                           &display_mode_create_info,
                                           NULL,
                                           &vc->khr.display_mode);
  if (result != VK_SUCCESS) {
    fprintf(stderr, "Unable to create mode\n");
    free(displays);
    free(modes);
    free(planes);
    return -1;
  }

  VkDisplaySurfaceCreateInfoKHR display_plane_surface_create_info = {
      .sType = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
      .displayMode = vc->khr.display_mode,
      .planeIndex = display_plane_idx,
      .imageExtent = modes[display_mode_idx].parameters.visibleRegion,
  };
  result = vkCreateDisplayPlaneSurfaceKHR(
      vc->instance, &display_plane_surface_create_info, NULL, &vc->surface);

  vc->width = modes[display_mode_idx].parameters.visibleRegion.width;
  vc->height = modes[display_mode_idx].parameters.visibleRegion.height;

  init_vk_objects(vc);

  create_swapchain(vc);

  free(displays);
  free(modes);
  free(planes);

  return 0;
}

static void renderloop_khr(struct vkcube* vc) {
  while (1) {
    uint32_t index;
    VkResult result = vkAcquireNextImageKHR(
        vc->device, vc->swap_chain, UINT64_MAX, vc->semaphore, VK_NULL_HANDLE, &index);
    if (result != VK_SUCCESS) return;

    assert(index <= MAX_NUM_IMAGES);
    vc->model.render(vc, &vc->buffers[index], true);

    vkQueuePresentKHR(vc->queue,
                      &(VkPresentInfoKHR){
                          .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                          .swapchainCount = 1,
                          .pSwapchains =
                              (VkSwapchainKHR[]){
                                  vc->swap_chain,
                              },
                          .pImageIndices =
                              (uint32_t[]){
                                  index,
                              },
                          .pResults = &result,
                      });
    if (result != VK_SUCCESS) return;

    vkQueueWaitIdle(vc->queue);
  }
}

static bool display_mode_from_string(const char* s, enum display_mode* mode) {
  if (streq(s, "auto")) {
    *mode = DISPLAY_MODE_AUTO;
    return true;
  } else if (streq(s, "headless")) {
    *mode = DISPLAY_MODE_HEADLESS;
    return true;
  } else if (streq(s, "kms")) {
    *mode = DISPLAY_MODE_KMS;
    return true;
#if defined(ENABLE_XCB)
  } else if (streq(s, "xcb")) {
    *mode = DISPLAY_MODE_XCB;
    return true;
#endif
  } else if (streq(s, "khr")) {
    *mode = DISPLAY_MODE_KHR;
    return true;
  } else {
    return false;
  }
}

static void print_usage(FILE* f) {
  const char* usage =
      "usage: vkcube [-n] [-o <file>]\n"
      "\n"
      "  -n                      Don't initialize vt or kms, run headless. This\n"
      "                          option is equivalent to '-m headless'.\n"
      "\n"
      "  -m <mode>               Choose display backend, where <mode> is one of\n"
      "                          \"auto\" (the default), \"headless\", \"khr\",\n"
      "                          \"kms\", \"wayland\", or \"xcb\". This option is\n"
      "                          incompatible with '-n'.\n"
      "\n"
      "  -w                      Specify width.\n"
      "\n"
      "  -h                      Specify height.\n"
      "\n"
      "  -k <display:mode:plane> Select KHR configuration with 3 number separated\n"
      "                          by the column character. To display the item\n"
      "                          corresponding to those number, just omit the number.\n"
      "\n"
      "  -o <file>               Path to output image when running headless.\n"
      "                          Default is \"./cube.png\".\n"
      "\n"
      "  -p                      Attempt to use protected content (encrypted).\n";

  fprintf(f, "%s", usage);
}

static void printflike(1, 2) usage_error(const char* fmt, ...) {
  va_list va;

  fprintf(stderr, "usage error: ");
  va_start(va, fmt);
  vfprintf(stderr, fmt, va);
  va_end(va);
  fprintf(stderr, "\n\n");
  print_usage(stderr);
  exit(EXIT_FAILURE);
}

void parse_args(int argc, char* argv[]) {
  /* Setting '+' in the optstring is the same as setting POSIXLY_CORRECT in
   * the enviroment. It tells getopt to stop parsing argv when it encounters
   * the first non-option argument; it also prevents getopt from permuting
   * argv during parsing.
   *
   * The initial ':' in the optstring makes getopt return ':' when an option
   * is missing a required argument.
   */
  static const char* optstring = "+:nm:w:h:o:k:p";

  int opt;
  bool found_arg_headless = false;
  bool found_arg_display_mode = false;

  while ((opt = getopt(argc, argv, optstring)) != -1) {
    switch (opt) {
      // Width & Heigt
      case 'w':
        width = atoi(optarg);
        break;
      case 'h':
        height = atoi(optarg);
        break;

      // Alternate display modes
      case 'm':
        found_arg_display_mode = true;
        if (!display_mode_from_string(optarg, &display_mode))
          usage_error("option -m given bad display mode");
        break;
      case 'k': {  //
        char config[40], *saveptr, *t;
        snprintf(config, sizeof(config), "%s", optarg);
        if ((t = strtok_r(config, ":", &saveptr))) {
          display_idx = atoi(t);
          if ((t = strtok_r(NULL, ":", &saveptr))) {
            display_mode_idx = atoi(t);
            if ((t = strtok_r(NULL, ":", &saveptr))) display_plane_idx = atoi(t);
          }
        }
        break;
      }

      // Protected chain
      case 'p':
        protected_chain = true;
        break;

      // Troubleshoot
      case '?':
        usage_error("invalid option '-%c'", optopt);
        break;
      case ':':
        usage_error("option -%c requires an argument", optopt);
        break;
      default:
        assert(!"unreachable");
        break;
    }
  }

  if (found_arg_headless && found_arg_display_mode)
    usage_error("options -n and -m are mutually exclusive");

  if (optind != argc) usage_error("trailing args");
}

void init_display(struct vkcube* vc) {
  switch (display_mode) {
    case DISPLAY_MODE_AUTO:
      display_mode = DISPLAY_MODE_HEADLESS;
      if (init_headless(vc) == -1) {
        fprintf(stderr,
                "failed to initialize headless mode, falling "
                "back to xcb\n");

#if defined(ENABLE_XCB)
        display_mode = DISPLAY_MODE_XCB;
        if (init_xcb(vc) == -1) {
          fprintf(stderr,
                  "failed to initialize xcb, falling back "
                  "to kms\n");
#endif
          display_mode = DISPLAY_MODE_KMS;
          if (init_kms(vc) == -1) { fprintf(stderr, "failed to initialize kms\n"); }
#if defined(ENABLE_XCB)
        }
#endif
      } else {
        default_display = true;
      }
      break;
    case DISPLAY_MODE_HEADLESS:
      if (init_headless(vc) == -1) {
        fail("failed to initialize headless mode");
        default_display = true;
      }
      break;
    case DISPLAY_MODE_KHR:
      if (init_khr(vc) == -1) fail("fail to initialize khr");
      break;
    case DISPLAY_MODE_KMS:
      if (init_kms(vc) == -1) fail("failed to initialize kms");
      break;
#if defined(ENABLE_XCB)
    case DISPLAY_MODE_XCB:
      if (init_xcb(vc) == -1) fail("failed to initialize xcb");
      break;
#endif
  }
}

bool renderloop(struct vkcube* vc) {
  bool windowShouldClose = false;
  vc->model.render(vc, &vc->buffers[0], false);
  vkQueueWaitIdle(vc->queue);
  return windowShouldClose;
}

bool renderloop_alternate_display(struct vkcube* vc) {
  bool windowShouldClose = false;
  switch (display_mode) {
    case DISPLAY_MODE_AUTO:
      assert(!"display mode is unset");
      break;
#if defined(ENABLE_WAYLAND)
    case DISPLAY_MODE_WAYLAND:
      renderloop_wayland(vc);
      break;
#endif
#if defined(ENABLE_XCB)
    case DISPLAY_MODE_XCB:
      windowShouldClose = renderloop_xcb(vc);
      break;
#endif
    case DISPLAY_MODE_KMS:
      renderloop_vt(vc);
      break;
    case DISPLAY_MODE_KHR:
      renderloop_khr(vc);
      break;
  }
  return windowShouldClose;
}