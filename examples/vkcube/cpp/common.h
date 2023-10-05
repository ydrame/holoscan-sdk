#include <stdarg.h>
#include <stdbool.h>
#include <stdnoreturn.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>

// #include <drm_fourcc.h>
// #include <png.h>
// #include <xf86drm.h>
// #include <xf86drmMode.h>

#if defined(ENABLE_XCB)
#include <xcb/xcb.h>
#define VK_USE_PLATFORM_XCB_KHR
#endif

#define VK_PROTOTYPES
#include <vulkan/vulkan.h>

// #include <gbm.h>

#include "esUtil.h"

#define printflike(a, b) __attribute__((format(printf, (a), (b))))

#define MAX_NUM_IMAGES 5
static uint32_t width = 1024, height = 768;
static bool protected_chain = false;
static bool default_display = false;

struct vkcube_buffer {
  // struct gbm_bo* gbm_bo;
  VkDeviceMemory mem;
  VkImage image;
  VkImageView view;
  VkFramebuffer framebuffer;
  VkFence fence;
  VkCommandBuffer cmd_buffer;

  uint32_t fb;
  uint32_t stride;
};

struct vkcube;

struct model {
  void (*init)(struct vkcube* vc);
  void (*render)(struct vkcube* vc, struct vkcube_buffer* b, bool wait_semaphore);
};

struct vkcube {
  struct model model;

  bool protected_;

  int fd;
  // struct gbm_device* gbm_device;

#if defined(ENABLE_XCB)
  struct {
    xcb_connection_t* conn;
    xcb_window_t window;
    xcb_atom_t atom_wm_protocols;
    xcb_atom_t atom_wm_delete_window;
  } xcb;
#endif

  struct {
    VkDisplayModeKHR display_mode;
  } khr;

  VkSwapchainKHR swap_chain;

  // drmModeCrtc* crtc;
  // drmModeConnector* connector;
  uint32_t width, height;

  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkPhysicalDeviceMemoryProperties memory_properties;
  VkDevice device;
  VkRenderPass render_pass;
  VkQueue queue;
  VkPipelineLayout pipeline_layout;
  VkPipeline pipeline;
  VkDeviceMemory mem;
  VkBuffer buffer;
  VkDescriptorSet descriptor_set;
  VkSemaphore semaphore;
  VkCommandPool cmd_pool;

  void* map;
  uint32_t vertex_offset, colors_offset, normals_offset;

  struct timeval start_tv;
  VkSurfaceKHR surface;
  VkFormat image_format;
  struct vkcube_buffer buffers[MAX_NUM_IMAGES];
  uint32_t image_count;
  int current;
};

void failv(const char* format, va_list args);
void fail(const char* format, ...) printflike(1, 2);
void fail_if(int cond, const char* format, ...) printflike(2, 3);

static inline bool streq(const char* a, const char* b) {
  return strcmp(a, b) == 0;
}

void parse_args(int argc, char* argv[]);
void init_display(struct vkcube* vc);
bool renderloop(struct vkcube* vc);
bool renderloop_alternate_display(struct vkcube* vc);
void init_cube_model();

// struct model cube_model;
