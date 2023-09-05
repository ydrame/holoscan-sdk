/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "vulkan_app.hpp"

#include <nvmath/nvmath.h>
#include <unistd.h>
#include <vulkan/spv/geometry_color_shader.glsl.vert.h>
#include <vulkan/spv/geometry_shader.glsl.frag.h>
#include <vulkan/spv/geometry_shader.glsl.vert.h>
#include <vulkan/spv/geometry_text_shader.glsl.frag.h>
#include <vulkan/spv/geometry_text_shader.glsl.vert.h>
#include <vulkan/spv/image_shader.glsl.frag.h>
#include <vulkan/spv/image_shader.glsl.vert.h>
#include <vulkan/spv/vkcube.glsl.frag.h>
#include <vulkan/spv/vkcube.glsl.vert.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../cuda/convert.hpp"
#include "../cuda/cuda_service.hpp"

// include the push constants definition shared between GLSL and C++ code
#include "shaders/push_constants.hpp"

#include <fonts/roboto_bold_ttf.hpp>
#include <holoscan/logger/logger.hpp>
#include <nvh/fileoperations.hpp>
#include <nvvk/appbase_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/context_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>

#include "../layers/layer.hpp"
#include "framebuffer_sequence.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace holoscan::viz {

static void format_info(ImageFormat format, uint32_t* src_channels, uint32_t* dst_channels,
                        uint32_t* component_size) {
  switch (format) {
    case ImageFormat::R8_UINT:
    case ImageFormat::R8_UNORM:
      *src_channels = *dst_channels = 1u;
      *component_size = sizeof(uint8_t);
      break;
    case ImageFormat::R16_UINT:
    case ImageFormat::R16_UNORM:
    case ImageFormat::R16_SFLOAT:
      *src_channels = *dst_channels = 1u;
      *component_size = sizeof(uint16_t);
      break;
    case ImageFormat::R32_UINT:
      *src_channels = *dst_channels = 1u;
      *component_size = sizeof(uint32_t);
      break;
    case ImageFormat::R32_SFLOAT:
      *src_channels = *dst_channels = 1u;
      *component_size = sizeof(float);
      break;
    case ImageFormat::R8G8B8_UNORM:
      *src_channels = 3u;
      *dst_channels = 4u;
      *component_size = sizeof(uint8_t);
      break;
    case ImageFormat::B8G8R8_UNORM:
      *src_channels = 3u;
      *dst_channels = 4u;
      *component_size = sizeof(uint8_t);
      break;
    case ImageFormat::R8G8B8A8_UNORM:
    case ImageFormat::B8G8R8A8_UNORM:
      *src_channels = *dst_channels = 4u;
      *component_size = sizeof(uint8_t);
      break;
    case ImageFormat::R16G16B16A16_UNORM:
    case ImageFormat::R16G16B16A16_SFLOAT:
      *src_channels = *dst_channels = 4u;
      *component_size = sizeof(uint16_t);
      break;
    case ImageFormat::R32G32B32A32_SFLOAT:
      *src_channels = *dst_channels = 4u;
      *component_size = sizeof(uint32_t);
      break;
    case ImageFormat::D32_SFLOAT:
      *src_channels = *dst_channels = 1u;
      *component_size = sizeof(uint32_t);
      break;
    default:
      throw std::runtime_error("Unhandled image format.");
  }
}

static vk::Format to_vulkan_format(ImageFormat format) {
  vk::Format vk_format;

  switch (format) {
    case ImageFormat::R8_UINT:
      vk_format = vk::Format::eR8Uint;
      break;
    case ImageFormat::R8_UNORM:
      vk_format = vk::Format::eR8Unorm;
      break;
    case ImageFormat::R16_UINT:
      vk_format = vk::Format::eR16Uint;
      break;
    case ImageFormat::R16_UNORM:
      vk_format = vk::Format::eR16Unorm;
      break;
    case ImageFormat::R16_SFLOAT:
      vk_format = vk::Format::eR16Sfloat;
      break;
    case ImageFormat::R32_UINT:
      vk_format = vk::Format::eR32Uint;
      break;
    case ImageFormat::R32_SFLOAT:
      vk_format = vk::Format::eR32Sfloat;
      break;
    case ImageFormat::R8G8B8_UNORM:
      vk_format = vk::Format::eR8G8B8A8Unorm;
      break;
    case ImageFormat::B8G8R8_UNORM:
      vk_format = vk::Format::eB8G8R8A8Unorm;
      break;
    case ImageFormat::R8G8B8A8_UNORM:
      vk_format = vk::Format::eR8G8B8A8Unorm;
      break;
    case ImageFormat::B8G8R8A8_UNORM:
      vk_format = vk::Format::eB8G8R8A8Unorm;
      break;
    case ImageFormat::R16G16B16A16_UNORM:
      vk_format = vk::Format::eR16G16B16A16Unorm;
      break;
    case ImageFormat::R16G16B16A16_SFLOAT:
      vk_format = vk::Format::eR16G16B16A16Sfloat;
      break;
    case ImageFormat::R32G32B32A32_SFLOAT:
      vk_format = vk::Format::eR32G32B32A32Sfloat;
      break;
    case ImageFormat::D32_SFLOAT:
      vk_format = vk::Format::eD32Sfloat;
      break;
    default:
      throw std::runtime_error("Unhandled image format.");
  }

  return vk_format;
}

/// Resource base class. Can be shared between CUDA and Vulkan. Access to the resource is
/// synchronized with semaphores.
class Resource {
 public:
  explicit Resource(vk::Device* device, nvvk::ResourceAllocator* alloc)
      : device_(device), alloc_(alloc) {}
  Resource() = delete;
  ~Resource() { destroy(); }

  /**
   * Synchronize access to the resource before using it with Vulkan
   *
   * @param batch_submission command buffer to use for synchronization
   */
  void access_with_vulkan(nvvk::BatchSubmission& batch_submission) {
    if (external_mem_) {
      if (state_ == AccessState::CUDA) {
        // enqueue the semaphore signalled by CUDA to be waited on by rendering
        batch_submission.enqueueWait(cuda_access_wait_semaphore_.get(),
                                     vk::PipelineStageFlagBits::eAllCommands);
      }

      // also signal the render semapore which will be waited on by CUDA
      batch_submission.enqueueSignal(vulkan_access_signal_semaphore_.get());
      state_ = AccessState::VULKAN;
    }
  }

  /**
   * Start accessing the resource with CUDA
   *
   * @param stream CUDA stream to use for synchronization
   */
  void begin_access_with_cuda(CUstream stream) {
    if (state_ == AccessState::VULKAN) {
      CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS ext_wait_params{};
      const CUexternalSemaphore external_wait_semaphore = vulkan_access_wait_semaphore_.get();
      CudaCheck(
          cuWaitExternalSemaphoresAsync(&external_wait_semaphore, &ext_wait_params, 1, stream));
      state_ = AccessState::UNKNOWN;
    }
  }

  /**
   * End of resource access from CUDA
   *
   * @param stream CUDA stream to use for synchronization
   */
  void end_access_with_cuda(CUstream stream) {
    // signal the semaphore for the CUDA operation on the buffer
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS ext_signal_params{};
    const CUexternalSemaphore external_signal_semaphore = cuda_access_signal_semaphore_.get();
    CudaCheck(
        cuSignalExternalSemaphoresAsync(&external_signal_semaphore, &ext_signal_params, 1, stream));
    state_ = AccessState::CUDA;
  }

  /// access state
  enum class AccessState {
    /// not accessed yet
    UNKNOWN,
    /// last accessed by CUDA
    CUDA,
    /// last accessed by VULKAN
    VULKAN
  };
  AccessState state_ = AccessState::UNKNOWN;

  UniqueCUexternalMemory external_mem_;

  /// this semaphore is used to synchronize CUDA operations on the texture, it's signaled by CUDA
  /// after accessing the texture (for upload) and waited on by Vulkan before accessing (rendering)
  vk::UniqueSemaphore cuda_access_wait_semaphore_;
  UniqueCUexternalSemaphore cuda_access_signal_semaphore_;

  /// this semaphore is used to synchronize Vulkan operations on the texture, it's signaled by
  /// Vulkan after accessing the texture (for rendering) and waited on by CUDA before accessing
  /// (upload)
  vk::UniqueSemaphore vulkan_access_signal_semaphore_;
  UniqueCUexternalSemaphore vulkan_access_wait_semaphore_;

  /// last usage of the texture, need to sync before destroying memory
  vk::Fence fence_ = nullptr;

 protected:
  vk::Device* const device_;
  nvvk::ResourceAllocator* const alloc_;

  CudaService* cuda_service_ = nullptr;

  void destroy() {
    if (fence_) {
      // if the resource had been tagged with a fence, wait for it before freeing the memory
      const vk::Result result = device_->waitForFences(fence_, true, 100'000'000);
      if (result != vk::Result::eSuccess) {
        HOLOSCAN_LOG_WARN("Waiting for texture fence failed with {}", vk::to_string(result));
      }
      fence_ = nullptr;
    }

    // check if this resource had been imported to CUDA
    if (external_mem_) {
      const CudaService::ScopedPush cuda_context = cuda_service_->PushContext();

      external_mem_.reset();
      cuda_access_signal_semaphore_.reset();
      vulkan_access_wait_semaphore_.reset();

      cuda_access_wait_semaphore_.reset();
      vulkan_access_signal_semaphore_.reset();
    }
  }

  void import_to_cuda(const std::unique_ptr<CudaService>& cuda_service,
                      const nvvk::MemAllocator::MemInfo& mem_info) {
    cuda_service_ = cuda_service.get();

    vk::MemoryGetFdInfoKHR memory_get_fd_info;
    memory_get_fd_info.memory = mem_info.memory;
    memory_get_fd_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
    UniqueValue<int, decltype(&close), &close> file_handle;
    file_handle.reset(device_->getMemoryFdKHR(memory_get_fd_info));

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memory_handle_desc{};
    memory_handle_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    memory_handle_desc.handle.fd = file_handle.get();
    memory_handle_desc.size = mem_info.offset + mem_info.size;

    external_mem_.reset([&memory_handle_desc] {
      CUexternalMemory external_mem;
      CudaCheck(cuImportExternalMemory(&external_mem, &memory_handle_desc));
      return external_mem;
    }());
    // don't need to close the file handle if it had been successfully imported
    file_handle.release();

    // create the semaphores, one for waiting after CUDA access and one for signalling
    // Vulkan access
    vk::StructureChain<vk::SemaphoreCreateInfo, vk::ExportSemaphoreCreateInfoKHR> chain;
    vk::SemaphoreCreateInfo& semaphore_create_info = chain.get<vk::SemaphoreCreateInfo>();
    chain.get<vk::ExportSemaphoreCreateInfoKHR>().handleTypes =
        vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
    cuda_access_wait_semaphore_ = device_->createSemaphoreUnique(semaphore_create_info);
    vulkan_access_signal_semaphore_ = device_->createSemaphoreUnique(semaphore_create_info);

    // import the semaphore to CUDA
    cuda_access_signal_semaphore_ = import_semaphore_to_cuda(cuda_access_wait_semaphore_.get());
    vulkan_access_wait_semaphore_ = import_semaphore_to_cuda(vulkan_access_signal_semaphore_.get());
  }

 private:
  UniqueCUexternalSemaphore import_semaphore_to_cuda(vk::Semaphore semaphore) {
    vk::SemaphoreGetFdInfoKHR semaphore_get_fd_info;
    semaphore_get_fd_info.semaphore = semaphore;
    semaphore_get_fd_info.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;

    UniqueValue<int, decltype(&close), &close> file_handle;
    file_handle.reset(device_->getSemaphoreFdKHR(semaphore_get_fd_info));

    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC semaphore_handle_desc{};
    semaphore_handle_desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
    semaphore_handle_desc.handle.fd = file_handle.get();

    UniqueCUexternalSemaphore cuda_semaphore;
    cuda_semaphore.reset([&semaphore_handle_desc] {
      CUexternalSemaphore ext_semaphore;
      CudaCheck(cuImportExternalSemaphore(&ext_semaphore, &semaphore_handle_desc));
      return ext_semaphore;
    }());

    // don't need to close the file handle if it had been successfully imported
    file_handle.release();

    return cuda_semaphore;
  }
};

struct Vulkan::Texture : public Resource {
  explicit Texture(vk::Device* device, nvvk::ResourceAllocator* alloc, uint32_t width,
                   uint32_t height, ImageFormat format)
      : Resource(device, alloc), width_(width), height_(height), format_(format) {}
  Texture() = delete;

  void import_to_cuda(const std::unique_ptr<CudaService>& cuda_service) {
    const CudaService::ScopedPush cuda_context = cuda_service->PushContext();

    const nvvk::MemAllocator::MemInfo mem_info =
        alloc_->getMemoryAllocator()->getMemoryInfo(texture_.memHandle);

    // call the base class for creating the external mem and the semaphores
    Resource::import_to_cuda(cuda_service, mem_info);

    uint32_t src_channels, dst_channels, component_size;
    format_info(format_, &src_channels, &dst_channels, &component_size);

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipmapped_array_desc{};
    mipmapped_array_desc.arrayDesc.Width = width_;
    mipmapped_array_desc.arrayDesc.Height = height_;
    mipmapped_array_desc.arrayDesc.Depth = 0;
    switch (component_size) {
      case 1:
        mipmapped_array_desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        break;
      case 2:
        mipmapped_array_desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
        break;
      case 4:
        mipmapped_array_desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
        break;
      default:
        throw std::runtime_error("Unhandled component size");
    }
    mipmapped_array_desc.arrayDesc.NumChannels = dst_channels;
    mipmapped_array_desc.arrayDesc.Flags = CUDA_ARRAY3D_SURFACE_LDST;

    mipmapped_array_desc.numLevels = 1;
    mipmapped_array_desc.offset = mem_info.offset;

    mipmap_.reset([external_mem = external_mem_.get(), &mipmapped_array_desc] {
      CUmipmappedArray mipmaped_array;
      CudaCheck(cuExternalMemoryGetMappedMipmappedArray(
          &mipmaped_array, external_mem, &mipmapped_array_desc));
      return mipmaped_array;
    }());
  }

  ~Texture() {
    destroy();

    // check if this texture had been imported to CUDA
    if (mipmap_) {
      const CudaService::ScopedPush cuda_context = cuda_service_->PushContext();
      mipmap_.reset();
    }
    alloc_->destroy(texture_);
  }

  const uint32_t width_;
  const uint32_t height_;
  const ImageFormat format_;

  nvvk::Texture texture_{};
  UniqueCUmipmappedArray mipmap_;
};

class Vulkan::Buffer : public Resource {
 public:
  explicit Buffer(vk::Device* device, nvvk::ResourceAllocator* alloc, size_t size)
      : Resource(device, alloc), size_(size) {}
  Buffer() = delete;

  ~Buffer() {
    destroy();

    // check if this buffer had been imported to CUDA
    if (device_ptr_) {
      const CudaService::ScopedPush cuda_context = cuda_service_->PushContext();
      device_ptr_.reset();
    }
    alloc_->destroy(buffer_);
  }

  void import_to_cuda(const std::unique_ptr<CudaService>& cuda_service) {
    const CudaService::ScopedPush cuda_context = cuda_service->PushContext();

    const nvvk::MemAllocator::MemInfo mem_info =
        alloc_->getMemoryAllocator()->getMemoryInfo(buffer_.memHandle);

    // call the base class for creating the external mem and the semaphores
    Resource::import_to_cuda(cuda_service, mem_info);

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer_desc{};
    buffer_desc.size = size_;
    buffer_desc.offset = mem_info.offset;

    device_ptr_.reset([external_mem = external_mem_.get(), &buffer_desc] {
      CUdeviceptr device_ptr;
      CudaCheck(cuExternalMemoryGetMappedBuffer(&device_ptr, external_mem, &buffer_desc));
      return device_ptr;
    }());
  }

  const size_t size_;

  nvvk::Buffer buffer_{};
  UniqueCUdeviceptr device_ptr_;
};

class Vulkan::Impl {
 public:
  Impl() = default;
  virtual ~Impl();

  void setup(Window* window, const std::string& font_path, float font_size_in_pixels);

  Window* get_window() const;

  const timeval& get_start_tv() const;
  void set_start_tv(const struct timeval& tv);

  void begin_transfer_pass();
  void end_transfer_pass();
  void begin_render_pass();
  void end_render_pass();
  void cleanup_transfer_jobs();

  void prepare_frame();
  void submit_frame();
  uint32_t get_active_image_index() const { return fb_sequence_.get_active_image_index(); }
  const std::vector<vk::UniqueCommandBuffer>& get_command_buffers() { return command_buffers_; }

  void set_viewport(float x, float y, float width, float height);

  Texture* create_texture_for_cuda_interop(uint32_t width, uint32_t height, ImageFormat format,
                                           vk::Filter filter, bool normalized);
  Texture* create_texture(uint32_t width, uint32_t height, ImageFormat format, size_t data_size,
                          const void* data, vk::Filter filter, bool normalized,
                          bool export_allocation);

  void upload_to_texture(CUdeviceptr device_ptr, size_t row_pitch, Texture* texture,
                         CUstream stream);
  void upload_to_texture(const void* host_ptr, size_t row_pitch, Texture* texture);

  Buffer* create_buffer_for_cuda_interop(size_t data_size, vk::BufferUsageFlags usage);
  Buffer* create_buffer(size_t data_size, vk::BufferUsageFlags usage, const void* data = nullptr);

  void upload_to_buffer(size_t data_size, CUdeviceptr device_ptr, Buffer* buffer, size_t dst_offset,
                        CUstream stream);
  void upload_to_buffer(size_t data_size, const void* data, const Buffer* buffer);

  void draw_texture(Texture* texture, Texture* depth_texture, Texture* lut, float opacity,
                    const nvmath::mat4f& view_matrix);

  void draw(vk::PrimitiveTopology topology, uint32_t count, uint32_t first,
            const std::vector<Buffer*>& vertex_buffers, float opacity,
            const std::array<float, 4>& color, float point_size, float line_width,
            const struct ubo& ubo, const nvmath::mat4f& view_matrix);

  void draw_text_indexed(vk::DescriptorSet desc_set, Buffer* vertex_buffer, Buffer* index_buffer,
                         vk::IndexType index_type, uint32_t index_count, uint32_t first_index,
                         uint32_t vertex_offset, float opacity, const nvmath::mat4f& view_matrix);

  void draw_indexed(vk::Pipeline pipeline, vk::PipelineLayout pipeline_layout,
                    vk::DescriptorSet desc_set, const std::vector<Buffer*>& vertex_buffers,
                    Buffer* index_buffer, vk::IndexType index_type, uint32_t index_count,
                    uint32_t first_index, uint32_t vertex_offset, float opacity,
                    const std::array<float, 4>& color, float point_size, float line_width,
                    const nvmath::mat4f& view_matrix);

  void draw_indexed(vk::PrimitiveTopology topology, const std::vector<Buffer*>& vertex_buffers,
                    Buffer* index_buffer, vk::IndexType index_type, uint32_t index_count,
                    uint32_t first_index, uint32_t vertex_offset, float opacity,
                    const std::array<float, 4>& color, float point_size, float line_width,
                    const nvmath::mat4f& view_matrix);

  void read_framebuffer(ImageFormat fmt, uint32_t width, uint32_t height, size_t buffer_size,
                        CUdeviceptr device_ptr, CUstream stream);

 private:
  void init_im_gui(const std::string& font_path, float font_size_in_pixels);
  void create_framebuffer_sequence();
  void create_render_pass();

  /**
   * Create all the framebuffers in which the image will be rendered
   * - Swapchain need to be created before calling this
   */
  void create_frame_buffers();

  /**
   * Callback when the window is resized
   * - Destroy allocated frames, then rebuild them with the new size
   */
  void on_framebuffer_size(int w, int h);

  vk::CommandBuffer create_temp_cmd_buffer();
  void submit_temp_cmd_buffer(vk::CommandBuffer cmd_buffer);

  uint32_t get_memory_type(uint32_t typeBits, const vk::MemoryPropertyFlags& properties) const;

  vk::UniquePipeline create_pipeline(
      vk::PipelineLayout pipeline_layout, const uint32_t* vertex_shader, size_t vertex_shader_size,
      const uint32_t* fragment_shader, size_t fragment_shader_size, vk::PrimitiveTopology topology,
      const std::vector<vk::DynamicState>& dynamic_states,
      const std::vector<vk::VertexInputBindingDescription>& binding_descriptions,
      const std::vector<vk::VertexInputAttributeDescription>& attribute_descriptions);

  CUstream select_cuda_stream(CUstream stream);
  void sync_with_selected_stream(CUstream ext_stream, CUstream selected_stream);

  Window* window_ = nullptr;

  std::unique_ptr<CudaService> cuda_service_;
  UniqueCUstream cuda_stream_;
  struct timeval start_tv;

  /**
   * NVVK objects don't use destructors but init()/deinit(). To maintain the destructor calling
   * sequence with Vulkan HPP objects store all NVVK objects in this struct and deinit() on
   * destructor.
   */
  class NvvkObjects {
   public:
    ~NvvkObjects() {
      try {
        if (index_buffer_.buffer) { alloc_.destroy(index_buffer_); }
        if (vertex_buffer_.buffer) { alloc_.destroy(vertex_buffer_); }
        if (read_transfer_cmd_pool_initialized_) { read_transfer_cmd_pool_.deinit(); }
        if (transfer_cmd_pool_initialized_) { transfer_cmd_pool_.deinit(); }
        if (export_alloc_initialized_) { export_alloc_.deinit(); }
        if (alloc_initialized_) { alloc_.deinit(); }
        vk_ctx_.deinit();
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("NvvkObjects destructor failed with {}", e.what());
      }
    }

    nvvk::Context vk_ctx_;

    // allocators
    /// Allocator for buffer, images, acceleration structures
    nvvk::ResourceAllocatorDma alloc_;
    bool alloc_initialized_ = false;
    /// Allocator for allocations which can be exported
    nvvk::ExportResourceAllocator export_alloc_;
    bool export_alloc_initialized_ = false;

    nvvk::BatchSubmission batch_submission_;

    nvvk::CommandPool transfer_cmd_pool_;
    bool transfer_cmd_pool_initialized_ = false;

    nvvk::CommandPool read_transfer_cmd_pool_;
    bool read_transfer_cmd_pool_initialized_ = false;

    nvvk::Buffer vertex_buffer_{};
    nvvk::Buffer index_buffer_{};
  } nvvk_;

  // Vulkan low level
  vk::Instance instance_;
  vk::PhysicalDeviceLineRasterizationFeaturesEXT line_rasterization_feature_;
  vk::Device device_;
  vk::UniqueSurfaceKHR surface_;
  vk::PhysicalDevice physical_device_;
  vk::Queue queue_gct_;
  vk::Queue queue_t_;
  vk::UniqueCommandPool cmd_pool_;
  vk::UniqueDescriptorPool im_gui_desc_pool_;

  /// Drawing/Surface
  FramebufferSequence fb_sequence_;
  /// All framebuffers, correspond to the Swapchain
  std::vector<vk::UniqueFramebuffer> framebuffers_;
  /// Command buffer per nb element in Swapchain
  std::vector<vk::UniqueCommandBuffer> command_buffers_;
  /// Fences per nb element in Swapchain
  std::vector<vk::UniqueFence> wait_fences_;
  /// Base render pass
  vk::UniqueRenderPass render_pass_;
  /// Size of the window
  vk::Extent2D size_{0, 0};
  /// Cache for pipeline/shaders
  vk::UniquePipelineCache pipeline_cache_;

  class TransferJob {
   public:
    vk::CommandBuffer cmd_buffer_ = nullptr;
    vk::UniqueSemaphore semaphore_;
    vk::UniqueFence fence_;
    vk::Fence frame_fence_ = nullptr;
  };
  std::list<TransferJob> transfer_jobs_;

  enum ReadTransferType { COLOR, DEPTH, COUNT };
  std::array<TransferJob, ReadTransferType::COUNT> read_transfer_jobs_{};
  std::array<std::unique_ptr<Vulkan::Buffer>, ReadTransferType::COUNT> read_transfer_buffers_;

  vk::UniquePipelineLayout image_pipeline_layout_;
  vk::UniquePipelineLayout geometry_pipeline_layout_;
  vk::UniquePipelineLayout vkcube_pipeline_layout_;
  vk::UniquePipelineLayout geometry_text_pipeline_layout_;

  nvvk::DescriptorSetBindings desc_set_layout_bind_;
  vk::UniqueDescriptorSetLayout desc_set_layout_;

  nvvk::DescriptorSetBindings desc_set_layout_bind_text_;
  vk::UniqueDescriptorSetLayout desc_set_layout_text_;
  vk::UniqueDescriptorPool desc_pool_text_;
  vk::DescriptorSet desc_set_text_;
  vk::Sampler sampler_text_;

  vk::UniquePipeline image_pipeline_;

  vk::UniquePipeline geometry_point_pipeline_;
  vk::UniquePipeline geometry_line_pipeline_;
  vk::UniquePipeline geometry_line_strip_pipeline_;
  vk::UniquePipeline geometry_triangle_pipeline_;

  vk::UniquePipeline geometry_point_color_pipeline_;
  vk::UniquePipeline geometry_line_color_pipeline_;
  vk::UniquePipeline geometry_line_strip_color_pipeline_;
  vk::UniquePipeline geometry_triangle_color_pipeline_;
  vk::UniquePipeline geometry_triangle_strip_color_pipeline_;

  vk::UniquePipeline geometry_text_pipeline_;

  ImGuiContext* im_gui_context_ = nullptr;
};

Vulkan::Impl::~Impl() {
  try {
    if (device_) {
      device_.waitIdle();

      cleanup_transfer_jobs();

      nvvk_.alloc_.releaseSampler(sampler_text_);

      if (ImGui::GetCurrentContext() != nullptr) {
        ImGui_ImplVulkan_Shutdown();
        if (im_gui_context_) { ImGui::DestroyContext(im_gui_context_); }
      }
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Vulkan::Impl destructor failed with {}", e.what());
  }
}

void Vulkan::Impl::setup(Window* window, const std::string& font_path, float font_size_in_pixels) {
  window_ = window;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  set_start_tv(tv);
  // Initialize instance independent function pointers
  {
    vk::DynamicLoader dl;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
  }

#ifdef NDEBUG
  nvvk::ContextCreateInfo context_info;
#else
  nvvk::ContextCreateInfo context_info(true /*bUseValidation*/);
#endif

  context_info.setVersion(1, 2);  // Using Vulkan 1.2

  // Requesting Vulkan extensions and layers
  uint32_t count{0};
  const char** req_extensions = window_->get_required_instance_extensions(&count);
  for (uint32_t ext_id = 0; ext_id < count; ext_id++)
    context_info.addInstanceExtension(req_extensions[ext_id]);

  // Allow debug names
  context_info.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);
  context_info.addInstanceExtension(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);

  req_extensions = window_->get_required_device_extensions(&count);
  for (uint32_t ext_id = 0; ext_id < count; ext_id++)
    context_info.addDeviceExtension(req_extensions[ext_id]);

  context_info.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  context_info.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  context_info.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  context_info.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  context_info.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  line_rasterization_feature_.smoothLines = true;
  context_info.addDeviceExtension(
      VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME, true /*optional*/, &line_rasterization_feature_);

  // Creating Vulkan base application
  if (!nvvk_.vk_ctx_.initInstance(context_info)) {
    throw std::runtime_error("Failed to create the Vulkan instance.");
  }
  instance_ = nvvk_.vk_ctx_.m_instance;

  // Initialize instance specific function pointers
  VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);

  // Find all compatible devices
  const std::vector<uint32_t> compatible_devices = nvvk_.vk_ctx_.getCompatibleDevices(context_info);
  if (compatible_devices.empty()) { throw std::runtime_error("No Vulkan capable GPU present."); }

  // Build a list of compatible physical devices
  const std::vector<vk::PhysicalDevice> physical_devices = instance_.enumeratePhysicalDevices();
  std::vector<vk::PhysicalDevice> compatible_physical_devices;
  for (auto&& compatible_device : compatible_devices) {
    compatible_physical_devices.push_back(vk::PhysicalDevice(physical_devices[compatible_device]));
  }

  // Let the window select the device to use (e.g. the one connected to the display if we opened
  // a visible windows)
  const uint32_t device_index = window_->select_device(instance_, compatible_physical_devices);

  // Finally initialize the device
  nvvk_.vk_ctx_.initDevice(device_index, context_info);
  device_ = nvvk_.vk_ctx_.m_device;
  physical_device_ = nvvk_.vk_ctx_.m_physicalDevice;

  {
    auto properties =
        physical_device_
            .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();

    HOLOSCAN_LOG_INFO("Using device {}: {} (UUID {:x})",
                      device_index,
                      properties.get<vk::PhysicalDeviceProperties2>().properties.deviceName,
                      fmt::join(properties.get<vk::PhysicalDeviceIDProperties>().deviceUUID, ""));

    // CUDA initialization
    CUuuid cuda_uuid;
    std::copy(properties.get<vk::PhysicalDeviceIDProperties>().deviceUUID.begin(),
              properties.get<vk::PhysicalDeviceIDProperties>().deviceUUID.end(),
              cuda_uuid.bytes);
    cuda_service_ = std::make_unique<CudaService>(cuda_uuid);

    const CudaService::ScopedPush cuda_context = cuda_service_->PushContext();
    cuda_stream_.reset([] {
      CUstream stream;
      CudaCheck(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
      return stream;
    }());
  }

  // Initialize device-specific function pointers function pointers
  VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);

  // create a surface, headless windows don't have a surface
  surface_ = vk::UniqueSurfaceKHR(window_->create_surface(physical_device_, instance_), instance_);
  if (surface_) {
    if (!nvvk_.vk_ctx_.setGCTQueueWithPresent(surface_.get())) {
      throw std::runtime_error("Surface not supported by queue");
    }
  }

  queue_gct_ = device_.getQueue(nvvk_.vk_ctx_.m_queueGCT.familyIndex, 0);
  queue_t_ = device_.getQueue(nvvk_.vk_ctx_.m_queueT.familyIndex, 0);
  cmd_pool_ = device_.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer});
  pipeline_cache_ = device_.createPipelineCacheUnique({});

  nvvk_.alloc_.init(instance_, device_, physical_device_);
  nvvk_.alloc_initialized_ = true;
  nvvk_.export_alloc_.init(device_, physical_device_, nvvk_.alloc_.getMemoryAllocator());
  nvvk_.export_alloc_initialized_ = true;

  create_framebuffer_sequence();
  create_render_pass();
  create_frame_buffers();

  // init batch submission
  nvvk_.batch_submission_.init(nvvk_.vk_ctx_.m_queueGCT);

  // init command pool
  nvvk_.transfer_cmd_pool_.init(device_, nvvk_.vk_ctx_.m_queueT.familyIndex);
  nvvk_.transfer_cmd_pool_initialized_ = true;

  nvvk_.read_transfer_cmd_pool_.init(device_, nvvk_.vk_ctx_.m_queueGCT.familyIndex);
  nvvk_.read_transfer_cmd_pool_initialized_ = true;

  // allocate the vertex and index buffer for the image draw pass
  {
    nvvk::CommandPool cmd_buf_get(device_, nvvk_.vk_ctx_.m_queueGCT.familyIndex);
    vk::CommandBuffer cmd_buf = cmd_buf_get.createCommandBuffer();

    const std::vector<float> vertices{
        -1.0f, -1.0f, 0.f, 1.0f, -1.0f, 0.f, 1.0f, 1.0f, 0.f, -1.0f, 1.0f, 0.f};
    nvvk_.vertex_buffer_ =
        nvvk_.alloc_.createBuffer(cmd_buf, vertices, vk::BufferUsageFlagBits::eVertexBuffer);
    const std::vector<uint16_t> indices{0, 2, 1, 2, 0, 3};
    nvvk_.index_buffer_ =
        nvvk_.alloc_.createBuffer(cmd_buf, indices, vk::BufferUsageFlagBits::eIndexBuffer);

    cmd_buf_get.submitAndWait(cmd_buf);
    nvvk_.alloc_.finalizeAndReleaseStaging();
  }

  // create the descriptor sets
  desc_set_layout_bind_.addBinding(SAMPLE_BINDING_COLOR,
                                   VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                   1,
                                   VK_SHADER_STAGE_FRAGMENT_BIT);
  desc_set_layout_bind_.addBinding(SAMPLE_BINDING_COLOR_U,
                                   VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                   1,
                                   VK_SHADER_STAGE_FRAGMENT_BIT);
  desc_set_layout_bind_.addBinding(SAMPLE_BINDING_LUT,
                                   VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                   1,
                                   VK_SHADER_STAGE_FRAGMENT_BIT);
  desc_set_layout_bind_.addBinding(SAMPLE_BINDING_DEPTH,
                                   VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                   1,
                                   VK_SHADER_STAGE_FRAGMENT_BIT);
  desc_set_layout_ = vk::UniqueDescriptorSetLayout(
      desc_set_layout_bind_.createLayout(device_,
                                         VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR),
      device_);

  {
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = VK_FILTER_LINEAR;
    info.minFilter = VK_FILTER_LINEAR;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.minLod = -1000;
    info.maxLod = 1000;
    info.maxAnisotropy = 1.0f;
    sampler_text_ = nvvk_.alloc_.acquireSampler(info);
  }
  desc_set_layout_bind_text_.addBinding(SAMPLE_BINDING_COLOR,
                                        vk::DescriptorType::eCombinedImageSampler,
                                        1,
                                        vk::ShaderStageFlagBits::eFragment,
                                        &sampler_text_);
  desc_set_layout_text_ =
      vk::UniqueDescriptorSetLayout(desc_set_layout_bind_text_.createLayout(device_), device_);
  desc_pool_text_ =
      vk::UniqueDescriptorPool(desc_set_layout_bind_text_.createPool(device_), device_);
  desc_set_text_ =
      nvvk::allocateDescriptorSet(device_, desc_pool_text_.get(), desc_set_layout_text_.get());

  // Push constants
  vk::PushConstantRange push_constant_ranges[2];
  push_constant_ranges[0].stageFlags = vk::ShaderStageFlagBits::eVertex;
  push_constant_ranges[0].offset = 0;
  push_constant_ranges[0].size = sizeof(PushConstantVertex);
  push_constant_ranges[1].stageFlags = vk::ShaderStageFlagBits::eFragment;
  push_constant_ranges[1].offset = sizeof(PushConstantVertex);
  push_constant_ranges[1].size = sizeof(PushConstantFragment);

  // Push constants
  vk::PushConstantRange vkcube_push_constant_ranges[1];
  vkcube_push_constant_ranges[0].stageFlags = vk::ShaderStageFlagBits::eVertex;
  vkcube_push_constant_ranges[0].offset = 0;
  vkcube_push_constant_ranges[0].size = sizeof(VKcubePushConstantVertex);

  // create the pipeline layout for images
  {
    // Creating the Pipeline Layout
    vk::PipelineLayoutCreateInfo create_info;
    create_info.setLayoutCount = 1;
    create_info.pSetLayouts = &desc_set_layout_.get();
    create_info.pushConstantRangeCount = 2;
    create_info.pPushConstantRanges = push_constant_ranges;
    image_pipeline_layout_ = device_.createPipelineLayoutUnique(create_info);
  }

  const std::vector<vk::VertexInputBindingDescription> binding_description_float_3{
      {0, sizeof(float) * 3, vk::VertexInputRate::eVertex}};
  const std::vector<vk::VertexInputAttributeDescription> attribute_description_float_3{
      {0, 0, vk::Format::eR32G32B32Sfloat, 0}};

  // Create the Pipeline
  image_pipeline_ =
      create_pipeline(image_pipeline_layout_.get(),
                      image_shader_glsl_vert,
                      sizeof(image_shader_glsl_vert) / sizeof(image_shader_glsl_vert[0]),
                      image_shader_glsl_frag,
                      sizeof(image_shader_glsl_frag) / sizeof(image_shader_glsl_frag[0]),
                      vk::PrimitiveTopology::eTriangleList,
                      {},
                      binding_description_float_3,
                      attribute_description_float_3);

  // create the pipeline layout for geometry
  {
    vk::PipelineLayoutCreateInfo create_info;
    create_info.pushConstantRangeCount = 2;
    create_info.pPushConstantRanges = push_constant_ranges;
    geometry_pipeline_layout_ = device_.createPipelineLayoutUnique(create_info);
  }

  {
    vk::PipelineLayoutCreateInfo create_info2;
    create_info2.pushConstantRangeCount = 1;
    create_info2.pPushConstantRanges = vkcube_push_constant_ranges;
    vkcube_pipeline_layout_ = device_.createPipelineLayoutUnique(create_info2);
  }

  geometry_point_pipeline_ =
      create_pipeline(geometry_pipeline_layout_.get(),
                      geometry_shader_glsl_vert,
                      sizeof(geometry_shader_glsl_vert) / sizeof(geometry_shader_glsl_vert[0]),
                      geometry_shader_glsl_frag,
                      sizeof(geometry_shader_glsl_frag) / sizeof(geometry_shader_glsl_frag[0]),
                      vk::PrimitiveTopology::ePointList,
                      {},
                      binding_description_float_3,
                      attribute_description_float_3);
  geometry_line_pipeline_ =
      create_pipeline(geometry_pipeline_layout_.get(),
                      geometry_shader_glsl_vert,
                      sizeof(geometry_shader_glsl_vert) / sizeof(geometry_shader_glsl_vert[0]),
                      geometry_shader_glsl_frag,
                      sizeof(geometry_shader_glsl_frag) / sizeof(geometry_shader_glsl_frag[0]),
                      vk::PrimitiveTopology::eLineList,
                      {vk::DynamicState::eLineWidth},
                      binding_description_float_3,
                      attribute_description_float_3);
  geometry_line_strip_pipeline_ =
      create_pipeline(geometry_pipeline_layout_.get(),
                      geometry_shader_glsl_vert,
                      sizeof(geometry_shader_glsl_vert) / sizeof(geometry_shader_glsl_vert[0]),
                      geometry_shader_glsl_frag,
                      sizeof(geometry_shader_glsl_frag) / sizeof(geometry_shader_glsl_frag[0]),
                      vk::PrimitiveTopology::eLineStrip,
                      {vk::DynamicState::eLineWidth},
                      binding_description_float_3,
                      attribute_description_float_3);
  geometry_triangle_pipeline_ =
      create_pipeline(geometry_pipeline_layout_.get(),
                      geometry_shader_glsl_vert,
                      sizeof(geometry_shader_glsl_vert) / sizeof(geometry_shader_glsl_vert[0]),
                      geometry_shader_glsl_frag,
                      sizeof(geometry_shader_glsl_frag) / sizeof(geometry_shader_glsl_frag[0]),
                      vk::PrimitiveTopology::eTriangleList,
                      {},
                      binding_description_float_3,
                      attribute_description_float_3);

  const std::vector<vk::VertexInputBindingDescription> binding_description_float_3_uint8_4{
      {0, sizeof(float) * 3, vk::VertexInputRate::eVertex},
      {1, sizeof(uint8_t) * 4, vk::VertexInputRate::eVertex}};
  const std::vector<vk::VertexInputAttributeDescription> attribute_description_float_3_uint8_4{
      {0, 0, vk::Format::eR32G32B32Sfloat, 0}, {1, 1, vk::Format::eR8G8B8A8Unorm, 0}};

  struct Vertex {
    float pos[3];
    float color[3];
    float normal[3];
  };
  const std::vector<vk::VertexInputBindingDescription> vkcube_binding_description_float_3_uint8_4{
      {0, sizeof(Vertex), vk::VertexInputRate::eVertex}};
  const std::vector<vk::VertexInputAttributeDescription>
      vkcube_attribute_description_float_3_uint8_4{
          {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)},
          {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)},
          {2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)}};

  geometry_point_color_pipeline_ = create_pipeline(
      geometry_pipeline_layout_.get(),
      geometry_color_shader_glsl_vert,
      sizeof(geometry_color_shader_glsl_vert) / sizeof(geometry_color_shader_glsl_vert[0]),
      geometry_shader_glsl_frag,
      sizeof(geometry_shader_glsl_frag) / sizeof(geometry_shader_glsl_frag[0]),
      vk::PrimitiveTopology::ePointList,
      {},
      binding_description_float_3_uint8_4,
      attribute_description_float_3_uint8_4);
  geometry_line_color_pipeline_ = create_pipeline(
      geometry_pipeline_layout_.get(),
      geometry_color_shader_glsl_vert,
      sizeof(geometry_color_shader_glsl_vert) / sizeof(geometry_color_shader_glsl_vert[0]),
      geometry_shader_glsl_frag,
      sizeof(geometry_shader_glsl_frag) / sizeof(geometry_shader_glsl_frag[0]),
      vk::PrimitiveTopology::eLineList,
      {vk::DynamicState::eLineWidth},
      binding_description_float_3_uint8_4,
      attribute_description_float_3_uint8_4);
  geometry_line_strip_color_pipeline_ = create_pipeline(
      geometry_pipeline_layout_.get(),
      geometry_color_shader_glsl_vert,
      sizeof(geometry_color_shader_glsl_vert) / sizeof(geometry_color_shader_glsl_vert[0]),
      geometry_shader_glsl_frag,
      sizeof(geometry_shader_glsl_frag) / sizeof(geometry_shader_glsl_frag[0]),
      vk::PrimitiveTopology::eLineStrip,
      {vk::DynamicState::eLineWidth},
      binding_description_float_3_uint8_4,
      attribute_description_float_3_uint8_4);
  geometry_triangle_color_pipeline_ = create_pipeline(
      geometry_pipeline_layout_.get(),
      geometry_color_shader_glsl_vert,
      sizeof(geometry_color_shader_glsl_vert) / sizeof(geometry_color_shader_glsl_vert[0]),
      geometry_shader_glsl_frag,
      sizeof(geometry_shader_glsl_frag) / sizeof(geometry_shader_glsl_frag[0]),
      vk::PrimitiveTopology::eTriangleList,
      {},
      binding_description_float_3_uint8_4,
      attribute_description_float_3_uint8_4);
  geometry_triangle_strip_color_pipeline_ =
      create_pipeline(vkcube_pipeline_layout_.get(),
                      vkcube_glsl_vert,
                      sizeof(vkcube_glsl_vert) / sizeof(vkcube_glsl_vert[0]),
                      vkcube_glsl_frag,
                      sizeof(vkcube_glsl_frag) / sizeof(vkcube_glsl_frag[0]),
                      vk::PrimitiveTopology::eTriangleStrip,
                      {},
                      vkcube_binding_description_float_3_uint8_4,
                      vkcube_attribute_description_float_3_uint8_4);
  // create the pipeline layout for text geometry
  {
    vk::PipelineLayoutCreateInfo create_info;
    create_info.setLayoutCount = 1;
    create_info.pSetLayouts = &desc_set_layout_text_.get();
    create_info.pushConstantRangeCount = 2;
    create_info.pPushConstantRanges = push_constant_ranges;
    geometry_text_pipeline_layout_ = device_.createPipelineLayoutUnique(create_info);
  }

  const std::vector<vk::VertexInputBindingDescription> binding_description_imgui{
      {0, sizeof(ImDrawVert), vk::VertexInputRate::eVertex}};
  const std::vector<vk::VertexInputAttributeDescription> attribute_description_imgui{
      {0, 0, vk::Format::eR32G32Sfloat, IM_OFFSETOF(ImDrawVert, pos)},
      {1, 0, vk::Format::eR32G32Sfloat, IM_OFFSETOF(ImDrawVert, uv)},
      {2, 0, vk::Format::eR8G8B8A8Unorm, IM_OFFSETOF(ImDrawVert, col)}};

  geometry_text_pipeline_ = create_pipeline(
      geometry_text_pipeline_layout_.get(),
      geometry_text_shader_glsl_vert,
      sizeof(geometry_text_shader_glsl_vert) / sizeof(geometry_text_shader_glsl_vert[0]),
      geometry_text_shader_glsl_frag,
      sizeof(geometry_text_shader_glsl_frag) / sizeof(geometry_text_shader_glsl_frag[0]),
      vk::PrimitiveTopology::eTriangleList,
      {},
      binding_description_imgui,
      attribute_description_imgui);

  // ImGui initialization
  init_im_gui(font_path, font_size_in_pixels);
  window_->setup_callbacks(
      [this](int width, int height) { this->on_framebuffer_size(width, height); });
  window_->init_im_gui();
}

Window* Vulkan::Impl::get_window() const {
  return window_;
}

const struct timeval& Vulkan::Impl::get_start_tv() const {
  return start_tv;
}
void Vulkan::Impl::set_start_tv(const struct timeval& tv) {
  start_tv = tv;
}

void Vulkan::Impl::init_im_gui(const std::string& font_path, float font_size_in_pixels) {
  // if the app did not specify a context, create our own
  if (!ImGui::GetCurrentContext()) { im_gui_context_ = ImGui::CreateContext(); }

  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;  // Avoiding the INI file
  io.LogFilename = nullptr;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // Enable Docking

  std::vector<vk::DescriptorPoolSize> pool_size{{vk::DescriptorType::eSampler, 1},
                                                {vk::DescriptorType::eCombinedImageSampler, 1}};
  vk::DescriptorPoolCreateInfo pool_info;
  pool_info.maxSets = 2;
  pool_info.poolSizeCount = 2;
  pool_info.pPoolSizes = pool_size.data();
  im_gui_desc_pool_ = device_.createDescriptorPoolUnique(pool_info);

  // Setup Platform/Renderer back ends
  ImGui_ImplVulkan_InitInfo init_info{};
  init_info.Instance = instance_;
  init_info.PhysicalDevice = physical_device_;
  init_info.Device = device_;
  init_info.QueueFamily = nvvk_.vk_ctx_.m_queueGCT.familyIndex;
  init_info.Queue = queue_gct_;
  init_info.PipelineCache = VK_NULL_HANDLE;
  init_info.DescriptorPool = im_gui_desc_pool_.get();
  init_info.Subpass = 0;
  init_info.MinImageCount = 2;
  init_info.ImageCount = static_cast<int>(fb_sequence_.get_image_count());
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  init_info.CheckVkResultFn = nullptr;
  init_info.Allocator = nullptr;

  if (!ImGui_ImplVulkan_Init(&init_info, render_pass_.get())) {
    throw std::runtime_error("Failed to initialize ImGui vulkan backend.");
  }

  // set the font, if the user provided a font path, use this, else use the default font
  ImFont* font = nullptr;
  if (!font_path.empty()) {
    // ImGui asserts in debug when the file does not exist resulting in program termination,
    // therefore first check if the file is there.
    if (std::filesystem::exists(font_path)) {
      font = io.Fonts->AddFontFromFileTTF(font_path.c_str(), font_size_in_pixels);
    }
    if (!font) {
      const std::string err = "Failed to load font " + font_path;
      throw std::runtime_error(err.c_str());
    }
  } else {
    // by default the font data will be deleted by ImGui, since the font data is a static array
    // avoid this
    ImFontConfig font_config;
    font_config.FontDataOwnedByAtlas = false;
    // add the Roboto Bold fond as the default font
    font_size_in_pixels = 25.f;
    font = io.Fonts->AddFontFromMemoryTTF(
        roboto_bold_ttf, sizeof(roboto_bold_ttf), font_size_in_pixels, &font_config);
    if (!font) { throw std::runtime_error("Failed to add default font."); }
  }

  // the size of the ImGui default font is 13 pixels, set the global font scale so that the
  // GUI text has the same size as with the default font.
  io.FontGlobalScale = 13.f / font_size_in_pixels;

  // build the font atlast
  if (!io.Fonts->Build()) { throw std::runtime_error("Failed to build font atlas."); }
  ImGui::SetCurrentFont(font);

  // Upload Fonts
  vk::CommandBuffer cmd_buf = create_temp_cmd_buffer();
  if (!ImGui_ImplVulkan_CreateFontsTexture(VkCommandBuffer(cmd_buf))) {
    throw std::runtime_error("Failed to create fonts texture.");
  }
  submit_temp_cmd_buffer(cmd_buf);
}

void Vulkan::Impl::begin_transfer_pass() {
  // create a new transfer job and a command buffer
  transfer_jobs_.emplace_back();
  TransferJob& transfer_job = transfer_jobs_.back();

  transfer_job.cmd_buffer_ = nvvk_.transfer_cmd_pool_.createCommandBuffer();
}

void Vulkan::Impl::end_transfer_pass() {
  if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_)) {
    throw std::runtime_error("Not in transfer pass.");
  }

  TransferJob& transfer_job = transfer_jobs_.back();

  // end the command buffer for this job
  transfer_job.cmd_buffer_.end();

  // create the fence and semaphore needed for submission
  transfer_job.semaphore_ = device_.createSemaphoreUnique({});
  transfer_job.fence_ = device_.createFenceUnique({});

  // finalize the staging job for later cleanup of resources
  // associates all current staging resources with the transfer fence
  nvvk_.alloc_.finalizeStaging(transfer_job.fence_.get());

  // submit staged transfers
  vk::SubmitInfo submit_info;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &transfer_job.cmd_buffer_;
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &transfer_job.semaphore_.get();
  queue_t_.submit(submit_info, transfer_job.fence_.get());

  // next graphics submission must wait for transfer completion
  nvvk_.batch_submission_.enqueueWait(transfer_job.semaphore_.get(),
                                      VK_PIPELINE_STAGE_TRANSFER_BIT);
}

void Vulkan::Impl::begin_render_pass() {
  // Acquire the next image
  prepare_frame();

  // Get the command buffer for the frame. There are n command buffers equal to the number of
  // in-flight frames.
  const uint32_t cur_frame = get_active_image_index();
  const vk::CommandBuffer cmd_buf = command_buffers_[cur_frame].get();

  vk::CommandBufferBeginInfo begin_info;
  begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  cmd_buf.begin(begin_info);

  // Clearing values
  std::array<vk::ClearValue, 2> clear_values;
  clear_values[0].color = vk::ClearColorValue(std::array<float, 4>({0.f, 0.f, 0.f, 0.f}));
  clear_values[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

  // Begin rendering
  vk::RenderPassBeginInfo render_pass_begin_info;
  render_pass_begin_info.clearValueCount = 2;
  render_pass_begin_info.pClearValues = clear_values.data();
  render_pass_begin_info.renderPass = render_pass_.get();
  render_pass_begin_info.framebuffer = framebuffers_[cur_frame].get();
  render_pass_begin_info.renderArea = {{0, 0}, size_};
  cmd_buf.beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);

  // set the dynamic viewport
  vk::Viewport viewport{
      0.0f, 0.0f, static_cast<float>(size_.width), static_cast<float>(size_.height), 0.0f, 1.0f};
  cmd_buf.setViewport(0, viewport);

  vk::Rect2D scissor{{0, 0}, {size_.width, size_.height}};
  cmd_buf.setScissor(0, scissor);
}

void Vulkan::Impl::end_render_pass() {
  const vk::CommandBuffer cmd_buf = command_buffers_[get_active_image_index()].get();

  // End rendering
  cmd_buf.endRenderPass();

  // Submit for display
  cmd_buf.end();
  submit_frame();
}

void Vulkan::Impl::cleanup_transfer_jobs() {
  for (auto it = transfer_jobs_.begin(); it != transfer_jobs_.end();) {
    if (it->fence_) {
      // check if the upload fence was triggered, that means the copy has completed and cmd
      // buffer can be destroyed
      const vk::Result result = device_.getFenceStatus(it->fence_.get());
      if (result == vk::Result::eSuccess) {
        nvvk_.transfer_cmd_pool_.destroy(it->cmd_buffer_);
        it->cmd_buffer_ = nullptr;

        // before destroying the fence release all staging buffers using that fence
        nvvk_.alloc_.releaseStaging();

        it->fence_.reset();
      } else if (result != vk::Result::eNotReady) {
        vk::resultCheck(result, "Failed to get upload fence status");
      }
    }

    if (!it->fence_) {
      if (it->frame_fence_) {
        // check if the frame fence was triggered, that means the job can be destroyed
        const vk::Result result = device_.getFenceStatus(it->frame_fence_);
        if (result == vk::Result::eSuccess) {
          it->semaphore_.reset();
          /// @todo instead of allocating and destroying semaphore and fences, move to
          /// unused list and reuse (call 'device_.resetFences(1, &it->fence_);' to reuse)
          it = transfer_jobs_.erase(it);
          continue;
        } else if (result != vk::Result::eNotReady) {
          vk::resultCheck(result, "Failed to get frame fence status");
        }
      } else {
        // this is a stale transfer buffer (no end_transfer_pass()?), remove it
        it = transfer_jobs_.erase(it);
        continue;
      }
    }
    ++it;
  }
}

void Vulkan::Impl::prepare_frame() {
  if (!transfer_jobs_.empty() && (!transfer_jobs_.back().fence_)) {
    throw std::runtime_error("Transfer pass is active!");
  }

  // Acquire the next image from the framebuffer sequence
  fb_sequence_.acquire();

  // Use a fence to wait until the command buffer has finished execution before using it again
  const uint32_t image_index = get_active_image_index();

  vk::Result result{vk::Result::eSuccess};
  do {
    result = device_.waitForFences(wait_fences_[image_index].get(), true, 1'000'000);
  } while (result == vk::Result::eTimeout);

  if (result != vk::Result::eSuccess) {
    // This allows Aftermath to do things and exit below
    usleep(1000);
    vk::resultCheck(result, "Failed to wait for frame fences");
    exit(-1);
  }

  // reset the fence to be re-used
  device_.resetFences(wait_fences_[image_index].get());

  // if there is a pending transfer job assign the frame fence of the frame which is about to be
  // rendered.
  if (!transfer_jobs_.empty() && !transfer_jobs_.back().frame_fence_) {
    transfer_jobs_.back().frame_fence_ = wait_fences_[image_index].get();
  }

  // try to free previous transfer jobs
  cleanup_transfer_jobs();
}

void Vulkan::Impl::submit_frame() {
  const uint32_t image_index = get_active_image_index();

  nvvk_.batch_submission_.enqueue(command_buffers_[image_index].get());

  // wait for the previous frame's semaphore
  if (fb_sequence_.get_active_read_semaphore()) {
    nvvk_.batch_submission_.enqueueWait(fb_sequence_.get_active_read_semaphore(),
                                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
  }
  // and signal this frames semaphore on completion
  nvvk_.batch_submission_.enqueueSignal(fb_sequence_.get_active_written_semaphore());

  const vk::Result result =
      vk::Result(nvvk_.batch_submission_.execute(wait_fences_[image_index].get(), 0b0000'0001));
  vk::resultCheck(result, "Failed to execute bach submission");

  // Presenting frame
  fb_sequence_.present(queue_gct_);
}

void Vulkan::Impl::create_framebuffer_sequence() {
  window_->get_framebuffer_size(&size_.width, &size_.height);

  fb_sequence_.init(&nvvk_.alloc_,
                    device_,
                    physical_device_,
                    queue_gct_,
                    nvvk_.vk_ctx_.m_queueGCT.familyIndex,
                    surface_.get());

  fb_sequence_.update(size_.width, size_.height, &size_);

  // Create Synchronization Primitives
  vk::FenceCreateInfo fence_create_info;
  fence_create_info.flags = vk::FenceCreateFlagBits::eSignaled;
  for (uint32_t index = 0; index < fb_sequence_.get_image_count(); ++index) {
    wait_fences_.push_back(device_.createFenceUnique(fence_create_info));
  }

  // Command buffers store a reference to the frame buffer inside their render pass info
  // so for static usage without having to rebuild them each frame, we use one per frame buffer
  vk::CommandBufferAllocateInfo allocate_info;
  allocate_info.commandPool = cmd_pool_.get();
  allocate_info.commandBufferCount = fb_sequence_.get_image_count();
  allocate_info.level = vk::CommandBufferLevel::ePrimary;
  command_buffers_ = device_.allocateCommandBuffersUnique(allocate_info);

  const vk::CommandBuffer cmd_buffer = create_temp_cmd_buffer();
  fb_sequence_.cmd_update_barriers(cmd_buffer);
  submit_temp_cmd_buffer(cmd_buffer);

#ifdef _DEBUG
  for (size_t i = 0; i < command_buffers_.size(); i++) {
    const std::string name = std::string("Holoviz") + std::to_string(i);
    vk::DebugUtilsObjectNameInfoEXT name_info;
    name_info.objectHandle = (uint64_t)VkCommandBuffer(command_buffers_[i].get());
    name_info.objectType = vk::ObjectType::eCommandBuffer;
    name_info.pObjectName = name.c_str();
    device_.setDebugUtilsObjectNameEXT(name_info);
  }
#endif  // _DEBUG
}

void Vulkan::Impl::create_render_pass() {
  render_pass_.reset();

  std::array<vk::AttachmentDescription, 2> attachments;
  // Color attachment
  attachments[0].format = fb_sequence_.get_color_format();
  attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
  attachments[0].finalLayout =
      surface_ ? vk::ImageLayout::ePresentSrcKHR : vk::ImageLayout::eColorAttachmentOptimal;
  attachments[0].samples = vk::SampleCountFlagBits::e1;

  // Depth attachment
  attachments[1].format = fb_sequence_.get_depth_format();
  attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
  attachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
  attachments[1].samples = vk::SampleCountFlagBits::e1;

  // One color, one depth
  const vk::AttachmentReference color_reference{0, vk::ImageLayout::eColorAttachmentOptimal};
  const vk::AttachmentReference depth_reference{1, vk::ImageLayout::eDepthStencilAttachmentOptimal};

  std::array<vk::SubpassDependency, 1> subpass_dependencies;
  // Transition from final to initial (VK_SUBPASS_EXTERNAL refers to all commands executed outside
  // of the actual renderpass)
  subpass_dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  subpass_dependencies[0].dstSubpass = 0;
  subpass_dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
  subpass_dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  subpass_dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
  subpass_dependencies[0].dstAccessMask =
      vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
  subpass_dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

  vk::SubpassDescription subpass_description;
  subpass_description.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
  subpass_description.colorAttachmentCount = 1;
  subpass_description.pColorAttachments = &color_reference;
  subpass_description.pDepthStencilAttachment = &depth_reference;

  vk::RenderPassCreateInfo render_pass_info;
  render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
  render_pass_info.pAttachments = attachments.data();
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass_description;
  render_pass_info.dependencyCount = static_cast<uint32_t>(subpass_dependencies.size());
  render_pass_info.pDependencies = subpass_dependencies.data();

  render_pass_ = device_.createRenderPassUnique(render_pass_info);

#ifdef _DEBUG
  vk::DebugUtilsObjectNameInfoEXT name_info;
  name_info.objectHandle = (uint64_t)VkRenderPass(render_pass_.get());
  name_info.objectType = vk::ObjectType::eRenderPass;
  name_info.pObjectName = R"(Holoviz)";
  device_.setDebugUtilsObjectNameEXT(name_info);
#endif  // _DEBUG
}

void Vulkan::Impl::create_frame_buffers() {
  // Recreate the frame buffers
  framebuffers_.clear();

  // Array of attachment (color, depth)
  std::array<vk::ImageView, 2> attachments;

  // Create frame buffers for every swap chain image
  vk::FramebufferCreateInfo framebuffer_create_info;
  framebuffer_create_info.renderPass = render_pass_.get();
  framebuffer_create_info.attachmentCount = 2;
  framebuffer_create_info.width = size_.width;
  framebuffer_create_info.height = size_.height;
  framebuffer_create_info.layers = 1;
  framebuffer_create_info.pAttachments = attachments.data();

  // Create frame buffers for every swap chain image
  for (uint32_t i = 0; i < fb_sequence_.get_image_count(); i++) {
    attachments[0] = fb_sequence_.get_color_image_view(i);
    attachments[1] = fb_sequence_.get_depth_image_view(i);
    framebuffers_.push_back(device_.createFramebufferUnique(framebuffer_create_info));
  }

#ifdef _DEBUG
  for (size_t i = 0; i < framebuffers_.size(); i++) {
    const std::string name = std::string("Holoviz") + std::to_string(i);
    vk::DebugUtilsObjectNameInfoEXT name_info;
    name_info.objectHandle = (uint64_t)VkFramebuffer(framebuffers_[i].get());
    name_info.objectType = vk::ObjectType::eFramebuffer;
    name_info.pObjectName = name.c_str();
    device_.setDebugUtilsObjectNameEXT(name_info);
  }
#endif  // _DEBUG
}

void Vulkan::Impl::on_framebuffer_size(int w, int h) {
  if ((w == 0) || (h == 0)) { return; }

  // Update imgui
  if (ImGui::GetCurrentContext() != nullptr) {
    auto& imgui_io = ImGui::GetIO();
    imgui_io.DisplaySize = ImVec2(static_cast<float>(w), static_cast<float>(h));
  }

  // Wait to finish what is currently drawing
  device_.waitIdle();
  queue_gct_.waitIdle();

  // Request new swapchain image size
  fb_sequence_.update(w, h, &size_);
  {
    const vk::CommandBuffer cmd_buffer = create_temp_cmd_buffer();
    fb_sequence_.cmd_update_barriers(cmd_buffer);  // Make them presentable
    submit_temp_cmd_buffer(cmd_buffer);
  }

  // Recreating other resources
  create_frame_buffers();
}

vk::CommandBuffer Vulkan::Impl::create_temp_cmd_buffer() {
  // Create an image barrier to change the layout from undefined to DepthStencilAttachmentOptimal
  vk::CommandBufferAllocateInfo allocate_info;
  allocate_info.commandBufferCount = 1;
  allocate_info.commandPool = cmd_pool_.get();
  allocate_info.level = vk::CommandBufferLevel::ePrimary;
  const vk::CommandBuffer cmd_buffer = device_.allocateCommandBuffers(allocate_info)[0];

  vk::CommandBufferBeginInfo begin_info;
  begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  cmd_buffer.begin(begin_info);
  return cmd_buffer;
}

void Vulkan::Impl::submit_temp_cmd_buffer(vk::CommandBuffer cmd_buffer) {
  cmd_buffer.end();

  vk::SubmitInfo submit_info;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd_buffer;
  queue_gct_.submit(submit_info);
  queue_gct_.waitIdle();
  device_.freeCommandBuffers(cmd_pool_.get(), 1, &cmd_buffer);
}

uint32_t Vulkan::Impl::get_memory_type(uint32_t typeBits,
                                       const vk::MemoryPropertyFlags& properties) const {
  const vk::PhysicalDeviceMemoryProperties memory_properties =
      physical_device_.getMemoryProperties();

  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
    if (((typeBits & (1 << i)) > 0) &&
        (memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
      return i;
  }
  std::string err = "Unable to find memory type " + vk::to_string(properties);
  HOLOSCAN_LOG_ERROR("{}", err.c_str());
  return ~0u;
}

vk::UniquePipeline Vulkan::Impl::create_pipeline(
    vk::PipelineLayout pipeline_layout, const uint32_t* vertex_shader, size_t vertex_shader_size,
    const uint32_t* fragment_shader, size_t fragment_shader_size, vk::PrimitiveTopology topology,
    const std::vector<vk::DynamicState>& dynamic_states,
    const std::vector<vk::VertexInputBindingDescription>& binding_descriptions,
    const std::vector<vk::VertexInputAttributeDescription>& attribute_descriptions) {
  nvvk::GraphicsPipelineState state;

  state.depthStencilState.depthTestEnable = true;
  state.depthStencilState.depthWriteEnable = true;
  state.depthStencilState.depthCompareOp = vk::CompareOp::eLessOrEqual;
  state.depthStencilState.depthBoundsTestEnable = false;
  state.depthStencilState.stencilTestEnable = false;

  state.inputAssemblyState.topology = topology;

  state.addBindingDescriptions(binding_descriptions);
  state.addAttributeDescriptions(attribute_descriptions);

  // disable culling
  state.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;

  // enable smooth line rasterization if available
  vk::PipelineRasterizationLineStateCreateInfoEXT line_state;
  if (nvvk_.vk_ctx_.hasDeviceExtension(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME)) {
    line_state.lineRasterizationMode = vk::LineRasterizationModeEXT::eRectangularSmooth;
    state.rasterizationState.pNext = &line_state;
  }

  // enable blending
  vk::PipelineColorBlendAttachmentState color_blend_attachment_state;
  color_blend_attachment_state.colorWriteMask =
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
  color_blend_attachment_state.blendEnable = true;
  color_blend_attachment_state.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
  color_blend_attachment_state.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
  color_blend_attachment_state.colorBlendOp = vk::BlendOp::eAdd;
  color_blend_attachment_state.srcAlphaBlendFactor = vk::BlendFactor::eOne;
  color_blend_attachment_state.dstAlphaBlendFactor = vk::BlendFactor::eZero;
  color_blend_attachment_state.alphaBlendOp = vk::BlendOp::eAdd;
  // remove the default blend attachment state
  state.clearBlendAttachmentStates();
  state.addBlendAttachmentState(color_blend_attachment_state);

  for (auto&& dynamic_state : dynamic_states) { state.addDynamicStateEnable(dynamic_state); }

  state.update();

  nvvk::GraphicsPipelineGenerator generator(device_, pipeline_layout, render_pass_.get(), state);

  std::vector<uint32_t> code;
  code.assign(vertex_shader, vertex_shader + vertex_shader_size);
  generator.addShader(code, vk::ShaderStageFlagBits::eVertex);
  code.assign(fragment_shader, fragment_shader + fragment_shader_size);
  generator.addShader(code, vk::ShaderStageFlagBits::eFragment);

  generator.update();

  return device_.createGraphicsPipelineUnique(pipeline_cache_.get(), generator.createInfo).value;
}

CUstream Vulkan::Impl::select_cuda_stream(CUstream stream) {
  // on single GPU or with special streams we can use the provided stream
  if (!cuda_service_->IsMultiGPU() || (stream == 0) || (stream == CU_STREAM_LEGACY) ||
      (stream == CU_STREAM_PER_THREAD)) {
    return stream;
  }

  // On MGPU we need to use the internal stream

  // Synchronize with the external stream by recording an event and synchronizing our internal
  // stream with the event
  UniqueCUevent event;

  // When recording an event, the event needs to be created with the same context it's recorded
  // with. Therefore get the context from the stream passed in.
  CUcontext stream_context;
  CudaCheck(cuStreamGetCtx(stream, &stream_context));

  {
    // make the stream context current
    const CudaService::ScopedPush pushed_stream_context =
        cuda_service_->PushContext(stream_context);

    // and create the event with the stream context
    event.reset([] {
      CUevent event;
      CudaCheck(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
      return event;
    }());

    CudaCheck(cuEventRecord(event.get(), stream));
  }

  // no wait on the event on the internal CUDA stream
  CudaCheck(cuStreamWaitEvent(cuda_stream_.get(), event.get(), CU_EVENT_WAIT_DEFAULT));

  return cuda_stream_.get();
}

void Vulkan::Impl::sync_with_selected_stream(CUstream ext_stream, CUstream selected_stream) {
  // nothing to do if the external stream had been selected
  if (ext_stream == selected_stream) { return; }

  // synchronize the external stream with our selected stream, this time we don't need to get
  // the stream context since we record the event on our internal stream.
  UniqueCUevent event;
  event.reset([] {
    CUevent event;
    CudaCheck(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
    return event;
  }());

  CudaCheck(cuEventRecord(event.get(), selected_stream));
  CudaCheck(cuStreamWaitEvent(ext_stream, event.get(), CU_EVENT_WAIT_DEFAULT));
}

Vulkan::Texture* Vulkan::Impl::create_texture(uint32_t width, uint32_t height, ImageFormat format,
                                              size_t data_size, const void* data, vk::Filter filter,
                                              bool normalized, bool export_allocation) {
  if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_)) {
    throw std::runtime_error(
        "Transfer command buffer not set. Calls to create_texture() need to be enclosed by "
        "begin_transfer_pass() and end_transfer_pass()");
  }

  const vk::Format vk_format = to_vulkan_format(format);
  uint32_t src_channels, dst_channels, component_size;
  format_info(format, &src_channels, &dst_channels, &component_size);

  const vk::ImageCreateInfo image_create_info =
      nvvk::makeImage2DCreateInfo(vk::Extent2D{width, height}, vk_format);
  nvvk::Image image;
  nvvk::ResourceAllocator* allocator;
  if (export_allocation) {
    allocator = &nvvk_.export_alloc_;
  } else {
    allocator = &nvvk_.alloc_;
  }
  if (data) {
    if (data_size != width * height * src_channels * component_size) {
      throw std::runtime_error("The size of the data array is wrong");
    }
    image = allocator->createImage(
        transfer_jobs_.back().cmd_buffer_, data_size, data, image_create_info);
  } else {
    // the VkExternalMemoryImageCreateInfoKHR struct is appended by nvvk::ExportResourceAllocator
    image = allocator->createImage(image_create_info, vk::MemoryPropertyFlagBits::eDeviceLocal);
  }

  // create the texture
  std::unique_ptr<Texture> texture =
      std::make_unique<Texture>(&device_, allocator, width, height, format);

  // create the Vulkan texture
  vk::SamplerCreateInfo sampler_create_info;
  sampler_create_info.minFilter = filter;
  sampler_create_info.magFilter = filter;
  sampler_create_info.mipmapMode = vk::SamplerMipmapMode::eNearest;
  sampler_create_info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
  sampler_create_info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
  sampler_create_info.addressModeW = vk::SamplerAddressMode::eClampToEdge;
  sampler_create_info.maxLod = normalized ? FLT_MAX : 0;
  sampler_create_info.unnormalizedCoordinates = normalized ? false : true;

  const vk::ImageViewCreateInfo image_view_info =
      nvvk::makeImageViewCreateInfo(image.image, image_create_info);
  texture->texture_ = allocator->createTexture(image, image_view_info, sampler_create_info);

  // transition to shader layout
  /// @todo I don't know if this is defined. Should the old layout be
  /// vk::ImageLayout::eTransferDstOptimal, like it would be if we uploaded using Vulkan?
  nvvk::cmdBarrierImageLayout(transfer_jobs_.back().cmd_buffer_,
                              image.image,
                              image_create_info.initialLayout,
                              vk::ImageLayout::eShaderReadOnlyOptimal);

  return texture.release();
}

void Vulkan::Impl::set_viewport(float x, float y, float width, float height) {
  uint32_t fb_width, fb_height;
  window_->get_framebuffer_size(&fb_width, &fb_height);

  const vk::CommandBuffer cmd_buf = command_buffers_[get_active_image_index()].get();

  vk::Viewport viewport{
      x * fb_width, y * fb_height, width * fb_width, height * fb_height, 0.0f, 1.0f};
  cmd_buf.setViewport(0, viewport);

  // height can be negative to flip the rendering, but scissor needs to be positive.
  if (height < 0) {
    height = -height;
    y -= height;
  }
  vk::Rect2D scissor{{std::max(0, static_cast<int32_t>(x * fb_width + .5f)),
                      std::max(0, static_cast<int32_t>(y * fb_height + .5f))},
                     {static_cast<uint32_t>(width * fb_width + .5f),
                      static_cast<uint32_t>(height * fb_height + .5f)}};
  cmd_buf.setScissor(0, scissor);
}

Vulkan::Texture* Vulkan::Impl::create_texture_for_cuda_interop(uint32_t width, uint32_t height,
                                                               ImageFormat format,
                                                               vk::Filter filter, bool normalized) {
  if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_)) {
    throw std::runtime_error(
        "Transfer command buffer not set. Calls to create_texture_for_cuda_interop() "
        "need to be enclosed by "
        "begin_transfer_pass() and "
        "end_transfer_pass()");
  }

  std::unique_ptr<Texture> texture;
  texture.reset(create_texture(
      width, height, format, 0, nullptr, filter, normalized, true /*export_allocation*/));

  texture->import_to_cuda(cuda_service_);

  return texture.release();
}

void Vulkan::Impl::upload_to_texture(CUdeviceptr device_ptr, size_t row_pitch, Texture* texture,
                                     CUstream ext_stream) {
  if (!texture->mipmap_) {
    throw std::runtime_error("Texture had not been imported to CUDA, can't upload data.");
  }

  const CudaService::ScopedPush cuda_context = cuda_service_->PushContext();

  // select the stream to be used by CUDA operations
  const CUstream stream = select_cuda_stream(ext_stream);

  // start accessing the texture with CUDA
  texture->begin_access_with_cuda(stream);

  CUarray array;
  CudaCheck(cuMipmappedArrayGetLevel(&array, texture->mipmap_.get(), 0));

  uint32_t src_channels, dst_channels, component_size;
  format_info(texture->format_, &src_channels, &dst_channels, &component_size);
  size_t src_pitch = row_pitch != 0 ? row_pitch : texture->width_ * src_channels * component_size;

  if (src_channels != dst_channels) {
    // three channel texture data is not hardware natively supported, convert to four channel
    if ((src_channels != 3) || (dst_channels != 4) || (component_size != 1)) {
      throw std::runtime_error("Unhandled conversion.");
    }

    // if the source CUDA memory is on a different device, allocate temporary memory, copy from
    // the source memory to the temporary memory and start the convert kernel using the temporary
    // memory
    UniqueAsyncCUdeviceptr tmp_device_ptr;
    if (!cuda_service_->IsMemOnDevice(device_ptr)) {
      const size_t tmp_pitch = texture->width_ * src_channels * component_size;

      // allocate temporary memory, note this is using the stream ordered memory allocator which
      // is not syncing globally like the normal `cuMemAlloc`
      tmp_device_ptr.reset([tmp_pitch, texture, stream] {
        CUdeviceptr device_ptr;
        CudaCheck(cuMemAllocAsync(&device_ptr, tmp_pitch * texture->height_, stream));
        return std::pair<CUdeviceptr, CUstream>(device_ptr, stream);
      }());

      CUDA_MEMCPY2D memcpy_2d{};
      memcpy_2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      memcpy_2d.srcDevice = device_ptr;
      memcpy_2d.srcPitch = src_pitch;
      memcpy_2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      memcpy_2d.dstDevice = tmp_device_ptr.get().first;
      memcpy_2d.dstPitch = tmp_pitch;
      memcpy_2d.WidthInBytes = tmp_pitch;
      memcpy_2d.Height = texture->height_;
      CudaCheck(cuMemcpy2DAsync(&memcpy_2d, stream));

      device_ptr = tmp_device_ptr.get().first;
      src_pitch = tmp_pitch;
    }

    ConvertR8G8B8ToR8G8B8A8(
        texture->width_, texture->height_, device_ptr, src_pitch, array, stream);
  } else {
    // else just copy
    CUDA_MEMCPY2D memcpy_2d{};
    memcpy_2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    memcpy_2d.srcDevice = device_ptr;
    memcpy_2d.srcPitch = src_pitch;
    memcpy_2d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    memcpy_2d.dstArray = array;
    memcpy_2d.WidthInBytes = texture->width_ * dst_channels * component_size;
    memcpy_2d.Height = texture->height_;
    CudaCheck(cuMemcpy2DAsync(&memcpy_2d, stream));
  }

  // indicate that the texture had been used by CUDA
  texture->end_access_with_cuda(stream);

  sync_with_selected_stream(ext_stream, stream);
}

void Vulkan::Impl::upload_to_texture(const void* host_ptr, size_t row_pitch, Texture* texture) {
  if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_)) {
    throw std::runtime_error(
        "Transfer command buffer not set. Calls to upload_to_texture() need to be enclosed by "
        "begin_transfer_pass() and end_transfer_pass()");
  }

  if ((texture->state_ != Texture::AccessState::VULKAN) &&
      (texture->state_ != Texture::AccessState::UNKNOWN)) {
    throw std::runtime_error(
        "When uploading to texture, the texture should be in Vulkan "
        "or unknown state");
  }

  const vk::CommandBuffer cmd_buf = transfer_jobs_.back().cmd_buffer_;

  // Copy buffer to image
  vk::ImageSubresourceRange subresource_range;
  subresource_range.aspectMask = vk::ImageAspectFlagBits::eColor;
  subresource_range.baseArrayLayer = 0;
  subresource_range.baseMipLevel = 0;
  subresource_range.layerCount = 1;
  subresource_range.levelCount = 1;

  nvvk::cmdBarrierImageLayout(cmd_buf,
                              texture->texture_.image,
                              vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eTransferDstOptimal,
                              subresource_range);

  vk::Offset3D offset;
  vk::ImageSubresourceLayers subresource;
  subresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  subresource.layerCount = 1;

  uint32_t src_channels, dst_channels, component_size;
  format_info(texture->format_, &src_channels, &dst_channels, &component_size);

  const uint32_t src_pitch = texture->width_ * src_channels * component_size;
  const uint32_t dst_pitch = texture->width_ * dst_channels * component_size;
  const vk::DeviceSize data_size = dst_pitch * texture->height_;

  void* mapping =
      nvvk_.alloc_.getStaging()->cmdToImage(cmd_buf,
                                            texture->texture_.image,
                                            VkOffset3D(offset),
                                            VkExtent3D{texture->width_, texture->height_, 1},
                                            VkImageSubresourceLayers(subresource),
                                            data_size,
                                            nullptr);

  if (src_channels != dst_channels) {
    // three channel texture data is not hardware natively supported, convert to four channel
    if ((src_channels != 3) || (dst_channels != 4) || (component_size != 1)) {
      throw std::runtime_error("Unhandled conversion.");
    }
    const uint8_t* src = reinterpret_cast<const uint8_t*>(host_ptr);
    uint32_t* dst = reinterpret_cast<uint32_t*>(mapping);
    for (uint32_t y = 0; y < texture->height_; ++y) {
      for (uint32_t x = 0; x < texture->width_; ++x) {
        const uint8_t data[4]{src[0], src[1], src[2], 0xFF};
        *dst = *reinterpret_cast<const uint32_t*>(&data);
        src += 3;
        ++dst;
      }
      if (row_pitch != 0) { src += row_pitch - src_pitch; }
    }
  } else {
    if ((row_pitch == 0) || (row_pitch == dst_pitch)) {
      // contiguous copy
      memcpy(mapping, host_ptr, data_size);
    } else {
      // source and destination pitch is different, copy row by row
      const uint8_t* src = reinterpret_cast<const uint8_t*>(host_ptr);
      uint8_t* dst = reinterpret_cast<uint8_t*>(mapping);
      for (uint32_t y = 0; y < texture->height_; ++y) {
        memcpy(dst, src, dst_pitch);
        src += row_pitch;
        dst += dst_pitch;
      }
    }
  }

  // Setting final image layout
  nvvk::cmdBarrierImageLayout(cmd_buf,
                              texture->texture_.image,
                              vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::eShaderReadOnlyOptimal);

  // no need to set the texture state here, the transfer command buffer submission is
  // always synchronized to the render command buffer submission.
}

Vulkan::Buffer* Vulkan::Impl::create_buffer(size_t data_size, vk::BufferUsageFlags usage,
                                            const void* data) {
  std::unique_ptr<Buffer> buffer(new Buffer(&device_, &nvvk_.alloc_, data_size));
  if (data) {
    if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_)) {
      throw std::runtime_error(
          "Transfer command buffer not set. Calls to create_buffer() with data need to be "
          "enclosed by begin_transfer_pass() and end_transfer_pass()");
    }

    buffer->buffer_ = nvvk_.alloc_.createBuffer(
        transfer_jobs_.back().cmd_buffer_, static_cast<vk::DeviceSize>(data_size), data, usage);
  } else {
    buffer->buffer_ = nvvk_.alloc_.createBuffer(
        static_cast<vk::DeviceSize>(data_size), usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
  }

  return buffer.release();
}

Vulkan::Buffer* Vulkan::Impl::create_buffer_for_cuda_interop(size_t data_size,
                                                             vk::BufferUsageFlags usage) {
  std::unique_ptr<Vulkan::Buffer> buffer;
  buffer.reset(create_buffer(data_size, usage));

  // import buffer to CUDA
  buffer->import_to_cuda(cuda_service_);

  return buffer.release();
}

void Vulkan::Impl::upload_to_buffer(size_t data_size, CUdeviceptr device_ptr, Buffer* buffer,
                                    size_t dst_offset, CUstream stream) {
  if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_)) {
    throw std::runtime_error(
        "Transfer command buffer not set. Calls to upload_to_buffer() need to be enclosed by "
        "begin_transfer_pass() and end_transfer_pass()");
  }

  if (!buffer->device_ptr_) {
    throw std::runtime_error("Buffer had not been imported to CUDA, can't upload data.");
  }

#if 1
  /// @brief @TODO workaround for uploading from device memory: download to host and then upload.
  ///  When uploading using cuMemcpy the data is corrupted, probably vertex cache.
  ///  - tried memory barrier before rendering:
  ///      vk::MemoryBarrier memory_barrier;
  ///      memory_barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
  ///      memory_barrier.dstAccessMask = vk::AccessFlagBits::eVertexAttributeRead;
  ///      cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
  ///                              vk::PipelineStageFlagBits::eVertexInput,
  ///                              vk::DependencyFlags(),
  ///                              memory_barrier,
  ///                              nullptr,
  ///                              nullptr);
  ///    with added self-dependency subpass
  ///      // self-dependency
  ///      subpass_dependencies[1].srcSubpass = 0;
  ///      subpass_dependencies[1].dstSubpass = 0;
  ///      subpass_dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eTopOfPipe;
  ///      subpass_dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eVertexInput;
  ///      subpass_dependencies[1].srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
  ///      subpass_dependencies[1].dstAccessMask = vk::AccessFlagBits::eVertexAttributeRead;
  ///      subpass_dependencies[1].dependencyFlags = vk::DependencyFlags();
  ///  - tried CUDA and vulkan syncs
  std::unique_ptr<uint8_t> host_data(new uint8_t[data_size]);
  CudaCheck(
      cuMemcpyDtoHAsync(reinterpret_cast<void*>(host_data.get()), device_ptr, data_size, stream));
  CudaCheck(cuStreamSynchronize(stream));
  upload_to_buffer(data_size, reinterpret_cast<const void*>(host_data.get()), buffer);
#else
  const CudaService::ScopedPush cuda_context = cuda_service_->PushContext();

  // start accessing the buffer with CUDA
  buffer->begin_access_with_cuda(stream);

  CudaCheck(
      cuMemcpyDtoDAsync(buffer->device_ptr_.get() + dst_offset, device_ptr, data_size, stream));

  // indicate that the texture had been used by CUDA
  buffer->end_access_with_cuda(stream);
#endif
}

void Vulkan::Impl::upload_to_buffer(size_t data_size, const void* data, const Buffer* buffer) {
  if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_)) {
    throw std::runtime_error(
        "Transfer command buffer not set. Calls to upload_to_buffer() need to be enclosed by "
        "begin_transfer_pass() and end_transfer_pass()");
  }

  const vk::CommandBuffer cmd_buf = transfer_jobs_.back().cmd_buffer_;
  nvvk::StagingMemoryManager* const staging = nvvk_.alloc_.getStaging();

  nvvk_.alloc_.getStaging()->cmdToBuffer(cmd_buf, buffer->buffer_.buffer, 0, data_size, data);
}

void Vulkan::Impl::draw_texture(Texture* texture, Texture* depth_texture, Texture* lut,
                                float opacity, const nvmath::mat4f& view_matrix) {
  const vk::CommandBuffer cmd_buf = command_buffers_[get_active_image_index()].get();

  PushConstantFragment push_constants;
  push_constants.flags = 0;

  // update descriptor sets
  std::vector<vk::WriteDescriptorSet> writes;
  uint32_t color_sample_binding = SAMPLE_BINDING_COLOR;
  if (lut) {
    lut->access_with_vulkan(nvvk_.batch_submission_);

    if ((texture->format_ == ImageFormat::R8_UINT) || (texture->format_ == ImageFormat::R16_UINT) ||
        (texture->format_ == ImageFormat::R32_UINT)) {
      color_sample_binding = SAMPLE_BINDING_COLOR_U;
      push_constants.flags |= PUSH_CONSTANT_FRAGMENT_FLAG_LUT_U;
    } else {
      push_constants.flags |= PUSH_CONSTANT_FRAGMENT_FLAG_LUT;
    }
    writes.emplace_back(
        desc_set_layout_bind_.makeWrite(nullptr, SAMPLE_BINDING_LUT, &lut->texture_.descriptor));
  } else {
    push_constants.flags |= PUSH_CONSTANT_FRAGMENT_FLAG_COLOR;
  }

  texture->access_with_vulkan(nvvk_.batch_submission_);
  writes.emplace_back(desc_set_layout_bind_.makeWrite(
      nullptr, color_sample_binding, &texture->texture_.descriptor));

  if (depth_texture) {
    depth_texture->access_with_vulkan(nvvk_.batch_submission_);

    writes.emplace_back(desc_set_layout_bind_.makeWrite(
        nullptr, SAMPLE_BINDING_DEPTH, &depth_texture->texture_.descriptor));
    push_constants.flags |= PUSH_CONSTANT_FRAGMENT_FLAG_DEPTH;
  }

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, image_pipeline_.get());

  cmd_buf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics,
                               image_pipeline_layout_.get(),
                               0,
                               static_cast<uint32_t>(writes.size()),
                               writes.data());

  // push the constants
  push_constants.opacity = opacity;
  cmd_buf.pushConstants(image_pipeline_layout_.get(),
                        vk::ShaderStageFlagBits::eFragment,
                        sizeof(PushConstantVertex),
                        sizeof(PushConstantFragment),
                        &push_constants);

  PushConstantVertex push_constant_vertex;
  push_constant_vertex.matrix = view_matrix;
  cmd_buf.pushConstants(image_pipeline_layout_.get(),
                        vk::ShaderStageFlagBits::eVertex,
                        0,
                        sizeof(PushConstantVertex),
                        &push_constant_vertex);

  // bind the buffers
  const vk::DeviceSize offset{0};
  const vk::Buffer buffer(nvvk_.vertex_buffer_.buffer);
  cmd_buf.bindVertexBuffers(0, 1, &buffer, &offset);
  cmd_buf.bindIndexBuffer(nvvk_.index_buffer_.buffer, 0, vk::IndexType::eUint16);

  // draw
  cmd_buf.drawIndexed(6, 1, 0, 0, 0);

  // tag the textures with the current fence
  texture->fence_ = wait_fences_[get_active_image_index()].get();
  if (depth_texture) { depth_texture->fence_ = wait_fences_[get_active_image_index()].get(); }
  if (lut) { lut->fence_ = wait_fences_[get_active_image_index()].get(); }
}

void Vulkan::Impl::draw(vk::PrimitiveTopology topology, uint32_t count, uint32_t first,
                        const std::vector<Buffer*>& vertex_buffers, float opacity,
                        const std::array<float, 4>& color, float point_size, float line_width,
                        const struct ubo& ubo, const nvmath::mat4f& view_matrix) {
  const vk::CommandBuffer cmd_buf = command_buffers_[get_active_image_index()].get();

  switch (topology) {
    case vk::PrimitiveTopology::ePointList:
      if (vertex_buffers.size() == 1) {
        cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, geometry_point_pipeline_.get());
      } else {
        cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             geometry_point_color_pipeline_.get());
      }
      break;
    case vk::PrimitiveTopology::eLineList:
      if (vertex_buffers.size() == 1) {
        cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, geometry_line_pipeline_.get());
      } else {
        cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, geometry_line_color_pipeline_.get());
      }
      cmd_buf.setLineWidth(line_width);
      break;
    case vk::PrimitiveTopology::eLineStrip:
      if (vertex_buffers.size() == 1) {
        cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, geometry_line_strip_pipeline_.get());
      } else {
        cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             geometry_line_strip_color_pipeline_.get());
      }
      cmd_buf.setLineWidth(line_width);
      break;
    case vk::PrimitiveTopology::eTriangleList:
      if (vertex_buffers.size() == 1) {
        cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, geometry_triangle_pipeline_.get());
      } else {
        cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             geometry_triangle_color_pipeline_.get());
      }
      break;
    case vk::PrimitiveTopology::eTriangleStrip:
      cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics,
                           geometry_triangle_strip_color_pipeline_.get());

      break;
    default:
      throw std::runtime_error("Unhandled primitive type");
  }

  if (!(topology == vk::PrimitiveTopology::eTriangleStrip)) {
    // push the constants
    PushConstantFragment push_constants_fragment;
    push_constants_fragment.opacity = opacity;
    cmd_buf.pushConstants(geometry_pipeline_layout_.get(),
                          vk::ShaderStageFlagBits::eFragment,
                          sizeof(PushConstantVertex),
                          sizeof(PushConstantFragment),
                          &push_constants_fragment);

    PushConstantVertex push_constant_vertex;
    push_constant_vertex.matrix = view_matrix;
    push_constant_vertex.point_size = point_size;
    push_constant_vertex.color = color;
    cmd_buf.pushConstants(geometry_pipeline_layout_.get(),
                          vk::ShaderStageFlagBits::eVertex,
                          0,
                          sizeof(PushConstantVertex),
                          &push_constant_vertex);
  } else {
    VKcubePushConstantVertex push_constant_vertex;
    push_constant_vertex.modelview = ubo.modelview;
    push_constant_vertex.modelviewprojection = ubo.modelviewprojection;
    memcpy(push_constant_vertex.normal, &ubo.normal, sizeof(ubo.normal));

    cmd_buf.pushConstants(vkcube_pipeline_layout_.get(),
                          vk::ShaderStageFlagBits::eVertex,
                          0,
                          sizeof(VKcubePushConstantVertex),
                          &push_constant_vertex);
  }
  // bind the buffers
  std::vector<vk::DeviceSize> offsets(vertex_buffers.size());
  std::vector<vk::Buffer> buffers(vertex_buffers.size());
  for (size_t index = 0; index < vertex_buffers.size(); ++index) {
    offsets[index] = 0;
    buffers[index] = vertex_buffers[index]->buffer_.buffer;
  }
  cmd_buf.bindVertexBuffers(0, vertex_buffers.size(), buffers.data(), offsets.data());

  // draw
  cmd_buf.draw(count, 1, first, 0);

  // tag the buffers with the current fence
  const vk::Fence fence = wait_fences_[get_active_image_index()].get();
  for (size_t index = 0; index < vertex_buffers.size(); ++index) {
    vertex_buffers[index]->fence_ = fence;
  }
}

void Vulkan::Impl::draw_text_indexed(vk::DescriptorSet desc_set, Buffer* vertex_buffer,
                                     Buffer* index_buffer, vk::IndexType index_type,
                                     uint32_t index_count, uint32_t first_index,
                                     uint32_t vertex_offset, float opacity,
                                     const nvmath::mat4f& view_matrix) {
  draw_indexed(geometry_text_pipeline_.get(),
               geometry_text_pipeline_layout_.get(),
               desc_set,
               {vertex_buffer},
               index_buffer,
               index_type,
               index_count,
               first_index,
               vertex_offset,
               opacity,
               std::array<float, 4>({1.f, 1.f, 1.f, 1.f}),
               1.f,
               0.f,
               view_matrix);
}

void Vulkan::Impl::draw_indexed(vk::Pipeline pipeline, vk::PipelineLayout pipeline_layout,
                                vk::DescriptorSet desc_set,
                                const std::vector<Buffer*>& vertex_buffers, Buffer* index_buffer,
                                vk::IndexType index_type, uint32_t index_count,
                                uint32_t first_index, uint32_t vertex_offset, float opacity,
                                const std::array<float, 4>& color, float point_size,
                                float line_width, const nvmath::mat4f& view_matrix) {
  const vk::CommandBuffer cmd_buf = command_buffers_[get_active_image_index()].get();

  for (auto&& vertex_buffer : vertex_buffers) {
    vertex_buffer->access_with_vulkan(nvvk_.batch_submission_);
  }

  if (line_width > 0.f) { cmd_buf.setLineWidth(line_width); }

  if (desc_set) {
    cmd_buf.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &desc_set, 0, nullptr);
  }

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

  // push the constants
  PushConstantFragment push_constants;
  push_constants.opacity = opacity;
  cmd_buf.pushConstants(pipeline_layout,
                        vk::ShaderStageFlagBits::eFragment,
                        sizeof(PushConstantVertex),
                        sizeof(PushConstantFragment),
                        &push_constants);

  PushConstantVertex push_constant_vertex;
  push_constant_vertex.matrix = view_matrix;
  push_constant_vertex.point_size = point_size;
  push_constant_vertex.color = color;
  cmd_buf.pushConstants(pipeline_layout,
                        vk::ShaderStageFlagBits::eVertex,
                        0,
                        sizeof(PushConstantVertex),
                        &push_constant_vertex);

  // bind the buffers
  std::vector<vk::DeviceSize> offsets(vertex_buffers.size());
  std::vector<vk::Buffer> buffers(vertex_buffers.size());
  for (size_t index = 0; index < vertex_buffers.size(); ++index) {
    offsets[index] = 0;
    buffers[index] = vertex_buffers[index]->buffer_.buffer;
  }
  cmd_buf.bindVertexBuffers(0, vertex_buffers.size(), buffers.data(), offsets.data());
  cmd_buf.bindIndexBuffer(index_buffer->buffer_.buffer, 0, index_type);

  // draw
  cmd_buf.drawIndexed(index_count, 1, first_index, vertex_offset, 0);

  // tag the buffers with the current fence
  const vk::Fence fence = wait_fences_[get_active_image_index()].get();
  for (size_t index = 0; index < vertex_buffers.size(); ++index) {
    vertex_buffers[index]->fence_ = fence;
  }
  index_buffer->fence_ = fence;
}

void Vulkan::Impl::draw_indexed(vk::PrimitiveTopology topology,
                                const std::vector<Buffer*>& vertex_buffers, Buffer* index_buffer,
                                vk::IndexType index_type, uint32_t index_count,
                                uint32_t first_index, uint32_t vertex_offset, float opacity,
                                const std::array<float, 4>& color, float point_size,
                                float line_width, const nvmath::mat4f& view_matrix) {
  vk::Pipeline pipeline;
  switch (topology) {
    case vk::PrimitiveTopology::ePointList:
      if (vertex_buffers.size() == 1) {
        pipeline = geometry_point_pipeline_.get();
      } else {
        pipeline = geometry_point_color_pipeline_.get();
      }
      break;
    case vk::PrimitiveTopology::eLineList:
      if (vertex_buffers.size() == 1) {
        pipeline = geometry_line_pipeline_.get();
      } else {
        pipeline = geometry_line_color_pipeline_.get();
      }
      break;
    case vk::PrimitiveTopology::eTriangleStrip:
      if (vertex_buffers.size() == 1) {
        pipeline = geometry_line_strip_pipeline_.get();
      } else {
        pipeline = geometry_line_strip_color_pipeline_.get();
      }
      break;
    case vk::PrimitiveTopology::eTriangleList:
      if (vertex_buffers.size() == 1) {
        pipeline = geometry_triangle_pipeline_.get();
      } else {
        pipeline = geometry_triangle_color_pipeline_.get();
      }
      break;
    default:
      throw std::runtime_error("Unhandled primitive type");
  }
  draw_indexed(pipeline,
               geometry_pipeline_layout_.get(),
               nullptr,
               vertex_buffers,
               index_buffer,
               index_type,
               index_count,
               first_index,
               vertex_offset,
               opacity,
               color,
               point_size,
               line_width,
               view_matrix);
}

void Vulkan::Impl::read_framebuffer(ImageFormat fmt, uint32_t width, uint32_t height,
                                    size_t buffer_size, CUdeviceptr device_ptr,
                                    CUstream ext_stream) {
  ReadTransferType transfer_type;
  vk::Image image;
  vk::Format image_format;
  vk::ImageAspectFlags image_aspect;
  vk::ImageLayout image_layout;

  if (fmt == ImageFormat::R8G8B8A8_UNORM) {
    transfer_type = ReadTransferType::COLOR;
    image = fb_sequence_.get_active_color_image();
    image_format = fb_sequence_.get_color_format();
    image_aspect = vk::ImageAspectFlagBits::eColor;
    image_layout =
        surface_ ? vk::ImageLayout::ePresentSrcKHR : vk::ImageLayout::eColorAttachmentOptimal;
  } else if (fmt == ImageFormat::D32_SFLOAT) {
    transfer_type = ReadTransferType::DEPTH;
    image = fb_sequence_.get_active_depth_image();
    image_format = fb_sequence_.get_depth_format();
    image_aspect = vk::ImageAspectFlagBits::eDepth;
    image_layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
  } else {
    throw std::runtime_error(
        "Unsupported image format, supported formats: R8G8B8A8_UNORM, D32_SFLOAT.");
  }

  TransferJob& read_job = read_transfer_jobs_[transfer_type];

  const vk::Format out_vk_format = to_vulkan_format(fmt);
  uint32_t src_channels, dst_channels, component_size;
  format_info(fmt, &src_channels, &dst_channels, &component_size);

  // limit size to actual framebuffer size
  const uint32_t read_width = std::min(size_.width, width);
  const uint32_t read_height = std::min(size_.height, height);

  const size_t data_size = read_width * read_height * dst_channels * component_size;
  if (buffer_size < data_size) { throw std::runtime_error("The size of the buffer is too small"); }

  // allocate the transfer buffer if needed
  const size_t src_data_size = read_width * read_height * src_channels * component_size;
  if (!read_transfer_buffers_[transfer_type] ||
      (read_transfer_buffers_[transfer_type]->size_ < src_data_size)) {
    read_transfer_buffers_[transfer_type].reset(
        create_buffer_for_cuda_interop(src_data_size, vk::BufferUsageFlagBits::eTransferDst));
  }
  Vulkan::Buffer* read_transfer_buffer = read_transfer_buffers_[transfer_type].get();

  // create a fence for the job, or wait on the previous job's fence if already created
  if (!read_job.fence_) {
    read_job.fence_ = device_.createFenceUnique({});
  } else {
    vk::Result result{vk::Result::eSuccess};
    do {
      result = device_.waitForFences(read_job.fence_.get(), true, 1'000'000);
    } while (result == vk::Result::eTimeout);

    if (result != vk::Result::eSuccess) {
      // This allows Aftermath to do things and exit below
      usleep(1000);
      vk::resultCheck(result, "Failed to wait for frame fences");
      exit(-1);
    }

    // reset the fence to be re-used
    device_.resetFences(read_job.fence_.get());
  }

  // create a command buffer, destroying the previous one if present
  if (read_job.cmd_buffer_) { nvvk_.read_transfer_cmd_pool_.destroy(read_job.cmd_buffer_); }
  read_job.cmd_buffer_ = nvvk_.read_transfer_cmd_pool_.createCommandBuffer();

  // Make the image layout vk::ImageLayout::eTransferSrcOptimal to copy to buffer
  vk::ImageSubresourceRange subresource_range;
  subresource_range.aspectMask = image_aspect;
  subresource_range.levelCount = 1;
  subresource_range.layerCount = 1;
  nvvk::cmdBarrierImageLayout(read_job.cmd_buffer_,
                              image,
                              image_layout,
                              vk::ImageLayout::eTransferSrcOptimal,
                              subresource_range);

  // Copy the image to the buffer
  vk::BufferImageCopy copy_region;
  copy_region.imageSubresource.aspectMask = image_aspect;
  copy_region.imageSubresource.layerCount = 1;
  copy_region.imageExtent.width = read_width;
  copy_region.imageExtent.height = read_height;
  copy_region.imageExtent.depth = 1;
  read_job.cmd_buffer_.copyImageToBuffer(image,
                                         vk::ImageLayout::eTransferSrcOptimal,
                                         read_transfer_buffer->buffer_.buffer,
                                         1,
                                         &copy_region);

  // Put back the image as it was
  nvvk::cmdBarrierImageLayout(read_job.cmd_buffer_,
                              image,
                              vk::ImageLayout::eTransferSrcOptimal,
                              image_layout,
                              subresource_range);

  read_job.cmd_buffer_.end();

  nvvk_.batch_submission_.enqueue(read_job.cmd_buffer_);

  read_transfer_buffer->access_with_vulkan(nvvk_.batch_submission_);

  // assign the fence to the transfer buffer, the fence will be checked on buffer destroy
  read_transfer_buffer->fence_ = read_job.fence_.get();

  // submit the command buffer
  const vk::Result result =
      vk::Result(nvvk_.batch_submission_.execute(read_job.fence_.get(), 0b0000'0001));
  vk::resultCheck(result, "Failed to execute bach submission");

  // copy the buffer to CUDA memory
  {
    const CudaService::ScopedPush cuda_context = cuda_service_->PushContext();

    // select the stream to be used by CUDA operations
    const CUstream stream = select_cuda_stream(ext_stream);

    // synchronize with the Vulkan copy
    read_transfer_buffer->begin_access_with_cuda(stream);

    if ((image_format == vk::Format::eB8G8R8A8Unorm) &&
        (out_vk_format == vk::Format::eR8G8B8A8Unorm)) {
      // if the destination CUDA memory is on a different device, allocate temporary memory, convert
      // from the read transfer buffer memory to the temporary memory and copy from temporary memory
      // to destination memory
      UniqueAsyncCUdeviceptr tmp_device_ptr;
      CUdeviceptr dst_device_ptr;
      size_t dst_pitch;
      if (!cuda_service_->IsMemOnDevice(device_ptr)) {
        // allocate temporary memory, note this is using the stream ordered memory allocator which
        // is not syncing globally like the normal `cuMemAlloc`
        dst_pitch = width * dst_channels * component_size;
        tmp_device_ptr.reset([dst_pitch, read_height, stream] {
          CUdeviceptr device_ptr;
          CudaCheck(cuMemAllocAsync(&device_ptr, dst_pitch * read_height, stream));
          return std::pair<CUdeviceptr, CUstream>(device_ptr, stream);
        }());
        dst_device_ptr = tmp_device_ptr.get().first;
      } else {
        dst_pitch = width * dst_channels * component_size;
        dst_device_ptr = device_ptr;
      }

      ConvertB8G8R8A8ToR8G8B8A8(read_width,
                                read_height,
                                read_transfer_buffer->device_ptr_.get(),
                                read_width * src_channels * component_size,
                                dst_device_ptr,
                                dst_pitch,
                                stream);

      if (tmp_device_ptr) {
        // copy from temporary memory to destination memory
        CUDA_MEMCPY2D memcpy_2d{};
        memcpy_2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        memcpy_2d.srcDevice = tmp_device_ptr.get().first;
        memcpy_2d.srcPitch = dst_pitch;
        memcpy_2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        memcpy_2d.dstDevice = device_ptr;
        memcpy_2d.dstPitch = width * dst_channels * component_size;
        memcpy_2d.WidthInBytes = width * dst_channels * component_size;
        memcpy_2d.Height = read_height;
        CudaCheck(cuMemcpy2DAsync(&memcpy_2d, stream));
      }
    } else if (image_format == out_vk_format) {
      CUDA_MEMCPY2D memcpy2d{};
      memcpy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      memcpy2d.srcDevice = read_transfer_buffer->device_ptr_.get();
      memcpy2d.srcPitch = read_width * src_channels * component_size;
      memcpy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      memcpy2d.dstDevice = device_ptr;
      memcpy2d.dstPitch = width * dst_channels * component_size;
      memcpy2d.WidthInBytes = memcpy2d.srcPitch;
      memcpy2d.Height = read_height;
      CudaCheck(cuMemcpy2DAsync(&memcpy2d, stream));
    } else {
      throw std::runtime_error("Unhandled framebuffer format.");
    }

    read_transfer_buffer->end_access_with_cuda(stream);

    sync_with_selected_stream(ext_stream, stream);
  }
}

Vulkan::Vulkan() : impl_(new Vulkan::Impl) {}

Vulkan::~Vulkan() {}

void Vulkan::setup(Window* window, const std::string& font_path, float font_size_in_pixels) {
  impl_->setup(window, font_path, font_size_in_pixels);
}

Window* Vulkan::get_window() const {
  return impl_->get_window();
}

const struct timeval& Vulkan::get_start_tv() const {
  return impl_->get_start_tv();
}

void Vulkan::set_start_tv(const struct timeval& tv) {
  impl_->set_start_tv(tv);
}

void Vulkan::begin_transfer_pass() {
  impl_->begin_transfer_pass();
}

void Vulkan::end_transfer_pass() {
  impl_->end_transfer_pass();
}

void Vulkan::begin_render_pass() {
  impl_->begin_render_pass();
}

void Vulkan::end_render_pass() {
  impl_->end_render_pass();
}

vk::CommandBuffer Vulkan::get_command_buffer() {
  return impl_->get_command_buffers()[impl_->get_active_image_index()].get();
}

void Vulkan::set_viewport(float x, float y, float width, float height) {
  impl_->set_viewport(x, y, width, height);
}

Vulkan::Texture* Vulkan::create_texture_for_cuda_interop(uint32_t width, uint32_t height,
                                                         ImageFormat format, vk::Filter filter,
                                                         bool normalized) {
  return impl_->create_texture_for_cuda_interop(width, height, format, filter, normalized);
}

Vulkan::Texture* Vulkan::create_texture(uint32_t width, uint32_t height, ImageFormat format,
                                        size_t data_size, const void* data, vk::Filter filter,
                                        bool normalized) {
  return impl_->create_texture(
      width, height, format, data_size, data, filter, normalized, false /*export_allocation*/);
}

void Vulkan::destroy_texture(Texture* texture) {
  delete texture;
}

void Vulkan::upload_to_texture(CUdeviceptr device_ptr, size_t row_pitch, Texture* texture,
                               CUstream stream) {
  impl_->upload_to_texture(device_ptr, row_pitch, texture, stream);
}

void Vulkan::upload_to_texture(const void* host_ptr, size_t row_pitch, Texture* texture) {
  impl_->upload_to_texture(host_ptr, row_pitch, texture);
}

Vulkan::Buffer* Vulkan::create_buffer(size_t data_size, const void* data,
                                      vk::BufferUsageFlags usage) {
  return impl_->create_buffer(data_size, usage, data);
}

Vulkan::Buffer* Vulkan::create_buffer_for_cuda_interop(size_t data_size,
                                                       vk::BufferUsageFlags usage) {
  return impl_->create_buffer_for_cuda_interop(data_size, usage);
}

void Vulkan::upload_to_buffer(size_t data_size, CUdeviceptr device_ptr, Buffer* buffer,
                              size_t dst_offset, CUstream stream) {
  return impl_->upload_to_buffer(data_size, device_ptr, buffer, dst_offset, stream);
}

void Vulkan::upload_to_buffer(size_t data_size, const void* data, const Buffer* buffer) {
  return impl_->upload_to_buffer(data_size, data, buffer);
}

void Vulkan::destroy_buffer(Buffer* buffer) {
  delete buffer;
}

void Vulkan::draw_texture(Texture* texture, Texture* depth_texture, Texture* lut, float opacity,
                          const nvmath::mat4f& view_matrix) {
  impl_->draw_texture(texture, depth_texture, lut, opacity, view_matrix);
}

void Vulkan::draw(vk::PrimitiveTopology topology, uint32_t count, uint32_t first,
                  const std::vector<Buffer*>& vertex_buffers, float opacity,
                  const std::array<float, 4>& color, float point_size, float line_width,
                  const struct ubo& ubo, const nvmath::mat4f& view_matrix) {
  impl_->draw(topology,
              count,
              first,
              vertex_buffers,
              opacity,
              color,
              point_size,
              line_width,
              ubo,
              view_matrix);
}

void Vulkan::draw_text_indexed(vk::DescriptorSet desc_set, Buffer* vertex_buffer,
                               Buffer* index_buffer, vk::IndexType index_type, uint32_t index_count,
                               uint32_t first_index, uint32_t vertex_offset, float opacity,
                               const nvmath::mat4f& view_matrix) {
  impl_->draw_text_indexed(desc_set,
                           vertex_buffer,
                           index_buffer,
                           index_type,
                           index_count,
                           first_index,
                           vertex_offset,
                           opacity,
                           view_matrix);
}

void Vulkan::draw_indexed(vk::PrimitiveTopology topology,
                          const std::vector<Buffer*>& vertex_buffers, Buffer* index_buffer,
                          vk::IndexType index_type, uint32_t index_count, uint32_t first_index,
                          uint32_t vertex_offset, float opacity, const std::array<float, 4>& color,
                          float point_size, float line_width, const nvmath::mat4f& view_matrix) {
  impl_->draw_indexed(topology,
                      vertex_buffers,
                      index_buffer,
                      index_type,
                      index_count,
                      first_index,
                      vertex_offset,
                      opacity,
                      color,
                      point_size,
                      line_width,
                      view_matrix);
}

void Vulkan::read_framebuffer(ImageFormat fmt, uint32_t width, uint32_t height, size_t buffer_size,
                              CUdeviceptr buffer, CUstream stream) {
  impl_->read_framebuffer(fmt, width, height, buffer_size, buffer, stream);
}

}  // namespace holoscan::viz
