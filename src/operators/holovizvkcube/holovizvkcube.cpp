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

#include "holoscan/operators/holovizvkcube/holovizvkcube.hpp"

#include <cuda_runtime.h>

#include <sys/time.h>
#include <algorithm>
#include <array>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/codecs.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/operators/holoviz/codecs.hpp"

#include "gxf/multimedia/video.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/tensor.hpp"
#include "holoviz/holoviz.hpp"  // holoviz module

#define CUDA_TRY(stmt)                                                                          \
  ({                                                                                            \
    cudaError_t _holoscan_cuda_err = stmt;                                                      \
    if (cudaSuccess != _holoscan_cuda_err) {                                                    \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).\n", \
                         #stmt,                                                                 \
                         __LINE__,                                                              \
                         __FILE__,                                                              \
                         cudaGetErrorString(_holoscan_cuda_err),                                \
                         _holoscan_cuda_err);                                                   \
    }                                                                                           \
    _holoscan_cuda_err;                                                                         \
  })

namespace viz = holoscan::viz;

/// Buffer information, can be initialized either with a tensor or a video buffer
struct BufferInfo {
  /**
   * Initialize with tensor
   *
   * @returns error code
   */
  gxf_result_t init(const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor) {
    rank = tensor->rank();
    shape = tensor->shape();
    element_type = tensor->element_type();
    name = tensor.name();
    buffer_ptr = tensor->pointer();
    storage_type = tensor->storage_type();
    bytes_size = tensor->bytes_size();
    for (uint32_t i = 0; i < rank; ++i) { stride[i] = tensor->stride(i); }

    return GXF_SUCCESS;
  }

  /**
   * Initialize with video buffer
   *
   * @returns error code
   */
  gxf_result_t init(const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>& video) {
    // NOTE: VideoBuffer::moveToTensor() converts VideoBuffer instance to the Tensor instance
    // with an unexpected shape:  [width, height] or [width, height, num_planes].
    // And, if we use moveToTensor() to convert VideoBuffer to Tensor, we may lose the original
    // video buffer when the VideoBuffer instance is used in other places. For that reason, we
    // directly access internal data of VideoBuffer instance to access Tensor data.
    const auto& buffer_info = video->video_frame_info();

    int32_t channels;
    switch (buffer_info.color_format) {
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        channels = 1;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned16;
        channels = 1;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned32;
        channels = 1;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        channels = 3;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        channels = 4;
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unsupported input format: {}\n",
                           static_cast<int64_t>(buffer_info.color_format));
        return GXF_FAILURE;
    }

    rank = 3;
    shape = nvidia::gxf::Shape{static_cast<int32_t>(buffer_info.height),
                               static_cast<int32_t>(buffer_info.width),
                               channels};
    name = video.name();
    buffer_ptr = video->pointer();
    storage_type = video->storage_type();
    bytes_size = video->size();
    stride[0] = buffer_info.color_planes[0].stride;
    stride[1] = channels;
    stride[2] = PrimitiveTypeSize(element_type);

    return GXF_SUCCESS;
  }

  uint32_t rank;
  nvidia::gxf::Shape shape;
  nvidia::gxf::PrimitiveType element_type;
  std::string name;
  const nvidia::byte* buffer_ptr;
  nvidia::gxf::MemoryStorageType storage_type;
  uint64_t bytes_size;
  nvidia::gxf::Tensor::stride_array_t stride;
};

namespace {

/// table to convert input type to string
static const std::array<std::pair<holoscan::ops::HolovizOp::InputType, std::string>, 20>
    kInputTypeToStr{{{holoscan::ops::HolovizOp::InputType::UNKNOWN, "unknown"},
                     {holoscan::ops::HolovizOp::InputType::COLOR, "color"},
                     {holoscan::ops::HolovizOp::InputType::COLOR_LUT, "color_lut"},
                     {holoscan::ops::HolovizOp::InputType::POINTS, "points"},
                     {holoscan::ops::HolovizOp::InputType::LINES, "lines"},
                     {holoscan::ops::HolovizOp::InputType::LINE_STRIP, "line_strip"},
                     {holoscan::ops::HolovizOp::InputType::TRIANGLES, "triangles"},
                     {holoscan::ops::HolovizOp::InputType::CROSSES, "crosses"},
                     {holoscan::ops::HolovizOp::InputType::RECTANGLES, "rectangles"},
                     {holoscan::ops::HolovizOp::InputType::OVALS, "ovals"},
                     {holoscan::ops::HolovizOp::InputType::TEXT, "text"},
                     {holoscan::ops::HolovizOp::InputType::DEPTH_MAP, "depth_map"},
                     {holoscan::ops::HolovizOp::InputType::DEPTH_MAP_COLOR, "depth_map_color"},
                     {holoscan::ops::HolovizOp::InputType::POINTS_3D, "points_3d"},
                     {holoscan::ops::HolovizOp::InputType::LINES_3D, "lines_3d"},
                     {holoscan::ops::HolovizOp::InputType::LINE_STRIP_3D, "line_strip_3d"},
                     {holoscan::ops::HolovizOp::InputType::TRIANGLES_3D, "triangles_3d"},
                     {holoscan::ops::HolovizOp::InputType::VBO, "vbo"},
                     {holoscan::ops::HolovizOp::InputType::UPDATE, "update"},
                     {holoscan::ops::HolovizOp::InputType::TRANSFORM, "transforms"}}};

/**
 * Convert a string to a input type enum
 *
 * @param string input type string
 * @return input type enum
 */
static nvidia::gxf::Expected<holoscan::ops::HolovizOp::InputType> inputTypeFromString(
    const std::string& string) {
  const auto it = std::find_if(std::cbegin(kInputTypeToStr),
                               std::cend(kInputTypeToStr),
                               [&string](const auto& v) { return v.second == string; });
  if (it != std::cend(kInputTypeToStr)) { return it->first; }

  HOLOSCAN_LOG_ERROR("Unsupported input type '{}'", string);
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/**
 * Convert a input type enum to a string
 *
 * @param input_type input type enum
 * @return input type string
 */
static std::string inputTypeToString(holoscan::ops::HolovizOp::InputType input_type) {
  const auto it = std::find_if(std::cbegin(kInputTypeToStr),
                               std::cend(kInputTypeToStr),
                               [&input_type](const auto& v) { return v.first == input_type; });
  if (it != std::cend(kInputTypeToStr)) { return it->second; }

  return "invalid";
}

/// table to convert depth map render mode to string
static const std::array<std::pair<holoscan::ops::HolovizOp::DepthMapRenderMode, std::string>, 3>
    kDepthMapRenderModeToStr{
        {{holoscan::ops::HolovizOp::DepthMapRenderMode::POINTS, "points"},
         {holoscan::ops::HolovizOp::DepthMapRenderMode::LINES, "lines"},
         {holoscan::ops::HolovizOp::DepthMapRenderMode::TRIANGLES, "triangles"}}};

/**
 * Convert a string to a depth map render mode enum
 *
 * @param string depth map render mode string
 * @return depth map render mode enum
 */
static nvidia::gxf::Expected<holoscan::ops::HolovizOp::DepthMapRenderMode>
depthMapRenderModeFromString(const std::string& string) {
  const auto it = std::find_if(std::cbegin(kDepthMapRenderModeToStr),
                               std::cend(kDepthMapRenderModeToStr),
                               [&string](const auto& v) { return v.second == string; });
  if (it != std::cend(kDepthMapRenderModeToStr)) { return it->first; }

  HOLOSCAN_LOG_ERROR("Unsupported depth map render mode '{}'", string);
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/**
 * Convert a depth map render mode enum to a string
 *
 * @param depth_map_render_mode depth map render mode enum
 * @return depth map render mode string
 */
static std::string depthMapRenderModeToString(
    holoscan::ops::HolovizOp::DepthMapRenderMode depth_map_render_mode) {
  const auto it = std::find_if(
      std::cbegin(kDepthMapRenderModeToStr),
      std::cend(kDepthMapRenderModeToStr),
      [&depth_map_render_mode](const auto& v) { return v.first == depth_map_render_mode; });
  if (it != std::cend(kDepthMapRenderModeToStr)) { return it->second; }

  return "invalid";
}

}  // namespace

/**
 * Custom YAML parser for InputSpec class
 */
template <>
struct YAML::convert<holoscan::ops::HolovizOp::InputSpec> {
  static Node encode(const holoscan::ops::HolovizOp::InputSpec& input_spec) {
    Node node;
    node["type"] = inputTypeToString(input_spec.type_);
    node["name"] = input_spec.tensor_name_;
    node["opacity"] = std::to_string(input_spec.opacity_);
    node["priority"] = std::to_string(input_spec.priority_);
    switch (input_spec.type_) {
      case holoscan::ops::HolovizOp::InputType::POINTS:
      case holoscan::ops::HolovizOp::InputType::LINES:
      case holoscan::ops::HolovizOp::InputType::LINE_STRIP:
      case holoscan::ops::HolovizOp::InputType::TRIANGLES:
      case holoscan::ops::HolovizOp::InputType::CROSSES:
      case holoscan::ops::HolovizOp::InputType::RECTANGLES:
      case holoscan::ops::HolovizOp::InputType::OVALS:
      case holoscan::ops::HolovizOp::InputType::POINTS_3D:
      case holoscan::ops::HolovizOp::InputType::LINES_3D:
      case holoscan::ops::HolovizOp::InputType::LINE_STRIP_3D:
      case holoscan::ops::HolovizOp::InputType::TRIANGLES_3D:
        node["color"] = input_spec.color_;
        node["line_width"] = std::to_string(input_spec.line_width_);
        node["point_size"] = std::to_string(input_spec.point_size_);
        break;
      case holoscan::ops::HolovizOp::InputType::TEXT:
        node["color"] = input_spec.color_;
        node["text"] = input_spec.text_;
        break;
      case holoscan::ops::HolovizOp::InputType::DEPTH_MAP:
        node["depth_map_render_mode"] =
            depthMapRenderModeToString(input_spec.depth_map_render_mode_);
        break;
      default:
        break;
    }
    for (auto&& view : input_spec.views_) { node["views"].push_back(view); }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::HolovizOp::InputSpec& input_spec) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    // YAML is using exceptions, catch them
    try {
      const auto maybe_input_type = inputTypeFromString(node["type"].as<std::string>());
      if (!maybe_input_type) { return false; }

      input_spec.tensor_name_ = node["name"].as<std::string>();
      input_spec.type_ = maybe_input_type.value();
      input_spec.opacity_ = node["opacity"].as<float>(input_spec.opacity_);
      input_spec.priority_ = node["priority"].as<int32_t>(input_spec.priority_);
      switch (input_spec.type_) {
        case holoscan::ops::HolovizOp::InputType::LINES:
        case holoscan::ops::HolovizOp::InputType::LINE_STRIP:
        case holoscan::ops::HolovizOp::InputType::TRIANGLES:
        case holoscan::ops::HolovizOp::InputType::CROSSES:
        case holoscan::ops::HolovizOp::InputType::RECTANGLES:
        case holoscan::ops::HolovizOp::InputType::OVALS:
        case holoscan::ops::HolovizOp::InputType::POINTS_3D:
        case holoscan::ops::HolovizOp::InputType::LINES_3D:
        case holoscan::ops::HolovizOp::InputType::LINE_STRIP_3D:
        case holoscan::ops::HolovizOp::InputType::TRIANGLES_3D:
          input_spec.color_ = node["color"].as<std::vector<float>>(input_spec.color_);
          input_spec.line_width_ = node["line_width"].as<float>(input_spec.line_width_);
          input_spec.point_size_ = node["point_size"].as<float>(input_spec.point_size_);
          break;
        case holoscan::ops::HolovizOp::InputType::TEXT:
          input_spec.color_ = node["color"].as<std::vector<float>>(input_spec.color_);
          input_spec.text_ = node["text"].as<std::vector<std::string>>(input_spec.text_);
          break;
        case holoscan::ops::HolovizOp::InputType::DEPTH_MAP:
          if (node["depth_map_render_mode"]) {
            const auto maybe_depth_map_render_mode =
                depthMapRenderModeFromString(node["depth_map_render_mode"].as<std::string>());
            if (maybe_depth_map_render_mode) {
              input_spec.depth_map_render_mode_ = maybe_depth_map_render_mode.value();
            }
          }
          break;
        default:
          break;
      }

      if (node["views"]) {
        input_spec.views_ =
            node["views"].as<std::vector<holoscan::ops::HolovizOp::InputSpec::View>>();
      }

      return true;
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
  }
};

/**
 * Custom YAML parser for InputSpec::View class
 */
template <>
struct YAML::convert<holoscan::ops::HolovizOp::InputSpec::View> {
  static Node encode(const holoscan::ops::HolovizOp::InputSpec::View& view) {
    Node node;
    node["offset_x"] = view.offset_x_;
    node["offset_y"] = view.offset_y_;
    node["width"] = view.width_;
    node["height"] = view.height_;
    if (view.matrix_.has_value()) { node["matrix"] = view.matrix_.value(); }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::HolovizOp::InputSpec::View& view) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    // YAML is using exceptions, catch them
    try {
      view.offset_x_ = node["offset_x"].as<float>(view.offset_x_);
      view.offset_y_ = node["offset_y"].as<float>(view.offset_y_);
      view.width_ = node["width"].as<float>(view.width_);
      view.height_ = node["height"].as<float>(view.height_);
      if (node["matrix"]) { view.matrix_ = node["matrix"].as<std::array<float, 16>>(); }

      return true;
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
  }
};

namespace holoscan::ops {

void HolovizOp::setup(OperatorSpec& spec) {
  constexpr uint32_t DEFAULT_WIDTH = 1920;
  constexpr uint32_t DEFAULT_HEIGHT = 1080;
  constexpr float DEFAULT_FRAMERATE = 60.f;
  static const std::string DEFAULT_WINDOW_TITLE(" Holoviz ");
  static const std::string DEFAULT_DISPLAY_NAME("DP-0");
  constexpr bool DEFAULT_EXCLUSIVE_DISPLAY = false;
  constexpr bool DEFAULT_FULLSCREEN = false;
  constexpr bool DEFAULT_HEADLESS = false;

  spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});

  spec.input<std::any>("input_specs").condition(ConditionType::kNone);
  // spec.input<std::any>("input_transform_specs").condition(ConditionType::kNone);

  auto& render_buffer_input =
      spec.input<gxf::Entity>("render_buffer_input").condition(ConditionType::kNone);
  spec.param(render_buffer_input_,
             "render_buffer_input",
             "RenderBufferInput",
             "Input for an empty render buffer.",
             &render_buffer_input);
  auto& render_buffer_output =
      spec.output<gxf::Entity>("render_buffer_output").condition(ConditionType::kNone);
  spec.param(render_buffer_output_,
             "render_buffer_output",
             "RenderBufferOutput",
             "Output for a filled render buffer. If an input render buffer is specified it is "
             "using that one, else it allocates a new buffer.",
             &render_buffer_output);

  auto& camera_pose_output =
      spec.output<std::array<float, 16>>("camera_pose_output").condition(ConditionType::kNone);
  spec.param(camera_pose_output_,
             "camera_pose_output",
             "CameraPoseOutput",
             "Output the camera pose. The camera parameters are returned in a 4x4 row major "
             "projection matrix.",
             &camera_pose_output);

  spec.param(
      tensors_,
      "tensors",
      "Input Tensors",
      "List of input tensors. 'name' is required, 'type' is optional (unknown, color, color_lut, "
      "points, lines, line_strip, triangles, crosses, rectangles, ovals, text, points_3d, "
      "lines_3d, line_strip_3d, triangles_3d, vbo).",
      std::vector<InputSpec>());

  spec.param(color_lut_,
             "color_lut",
             "ColorLUT",
             "Color lookup table for tensors of type 'color_lut'",
             {});

  spec.param(window_title_,
             "window_title",
             "Window title",
             "Title on window canvas",
             DEFAULT_WINDOW_TITLE);
  spec.param(display_name_,
             "display_name",
             "Display name",
             "In exclusive mode, name of display to use as shown with xrandr.",
             DEFAULT_DISPLAY_NAME);
  spec.param(width_,
             "width",
             "Width",
             "Window width or display resolution width if in exclusive or fullscreen mode.",
             DEFAULT_WIDTH);
  spec.param(height_,
             "height",
             "Height",
             "Window height or display resolution height if in exclusive or fullscreen mode.",
             DEFAULT_HEIGHT);
  spec.param(framerate_,
             "framerate",
             "Framerate",
             "Display framerate in Hz if in exclusive mode.",
             DEFAULT_FRAMERATE);
  spec.param(use_exclusive_display_,
             "use_exclusive_display",
             "Use exclusive display",
             "Enable exclusive display",
             DEFAULT_EXCLUSIVE_DISPLAY);
  spec.param(fullscreen_,
             "fullscreen",
             "Use fullscreen window",
             "Enable fullscreen window",
             DEFAULT_FULLSCREEN);
  spec.param(headless_,
             "headless",
             "Headless",
             "Enable headless mode. No window is opened, the render buffer is output to "
             "‘render_buffer_output’.",
             DEFAULT_HEADLESS);
  spec.param(window_close_scheduling_term_,
             "window_close_scheduling_term",
             "WindowCloseSchedulingTerm",
             "BooleanSchedulingTerm to stop the codelet from ticking when the window is closed.");

  spec.param(
      allocator_, "allocator", "Allocator", "Allocator used to allocate render buffer output.");

  spec.param(font_path_,
             "font_path",
             "FontPath",
             "File path for the font used for rendering text",
             std::string());

  cuda_stream_handler_.defineParams(spec);
}

void HolovizOp::initialize() {
  HolovizOp::initialize();
}

void HolovizOp::start() {
  viz::Begin();
  // set the font to be used
  if (!font_path_.get().empty()) { viz::SetFont(font_path_.get().c_str(), 25.f); }

  // initialize Holoviz
  viz::InitFlags init_flags = viz::InitFlags::NONE;
  if (fullscreen_ && headless_) {
    throw std::runtime_error("Headless and fullscreen are mutually exclusive.");
  }
  if (fullscreen_) { init_flags = viz::InitFlags::FULLSCREEN; }
  if (headless_) { init_flags = viz::InitFlags::HEADLESS; }

  if (use_exclusive_display_) {
    viz::Init(
        display_name_.get().c_str(), width_, height_, uint32_t(framerate_ * 1000.f), init_flags);
  } else {
    viz::Init(width_, height_, window_title_.get().c_str(), init_flags);
  }

  // get the color lookup table
  const auto& color_lut = color_lut_.get();
  lut_.reserve(color_lut.size() * 4);
  for (auto&& color : color_lut) {
    if (color.size() != 4) {
      std::string msg = fmt::format(
          "Expected four components in color lookup table element, but got {}", color.size());
      throw std::runtime_error(msg);
    }
    lut_.insert(lut_.end(), color.begin(), color.end());
  }

  // cast Condition to BooleanCondition
  window_close_scheduling_term_->enable_tick();

  // Copy the user defined input spec list to the internal input spec list. If there is no user
  // defined input spec it will be generated from the first messages received.
  if (!tensors_.get().empty()) {
    initial_input_spec_.reserve(tensors_.get().size());
    initial_input_spec_.insert(
        initial_input_spec_.begin(), tensors_.get().begin(), tensors_.get().end());
  }
  gettimeofday(&start_tv, NULL);

  viz::BeginGeometryLayer();
  {
    std::vector<std::array<float, 3>> vertices = {
        // front
        {-1.0f, -1.0f, +1.0f},  // point blue
        {+1.0f, -1.0f, +1.0f},  // point magenta
        {-1.0f, +1.0f, +1.0f},  // point cyan
        {+1.0f, +1.0f, +1.0f},  // point white
        // back
        {+1.0f, -1.0f, -1.0f},  // point red
        {-1.0f, -1.0f, -1.0f},  // point black
        {+1.0f, +1.0f, -1.0f},  // point yellow
        {-1.0f, +1.0f, -1.0f},  // point green
        // right
        {+1.0f, -1.0f, +1.0f},  // point magenta
        {+1.0f, -1.0f, -1.0f},  // point red
        {+1.0f, +1.0f, +1.0f},  // point white
        {+1.0f, +1.0f, -1.0f},  // point yellow
        // left
        {-1.0f, -1.0f, -1.0f},  // point black
        {-1.0f, -1.0f, +1.0f},  // point blue
        {-1.0f, +1.0f, -1.0f},  // point green
        {-1.0f, +1.0f, +1.0f},  // point cyan
        // top
        {-1.0f, +1.0f, +1.0f},  // point cyan
        {+1.0f, +1.0f, +1.0f},  // point white
        {-1.0f, +1.0f, -1.0f},  // point green
        {+1.0f, +1.0f, -1.0f},  // point yellow
        // bottom
        {-1.0f, -1.0f, -1.0f},  // point black
        {+1.0f, -1.0f, -1.0f},  // point red
        {-1.0f, -1.0f, +1.0f},  // point blue
        {+1.0f, -1.0f, +1.0f},  // point magenta
    };

    viz::Primitive(
        viz::PrimitiveTopology::VBO, vertices.size() / 4, vertices.size(), vertices.data()->data());

    std::vector<std::array<float, 3>> colors =
        {                     // front
         {0.0f, 0.0f, 1.0f},  // blue
         {1.0f, 0.0f, 1.0f},  // magenta
         {0.0f, 1.0f, 1.0f},  // cyan
         {1.0f, 1.0f, 1.0f},  // white
                              // back
         {1.0f, 0.0f, 0.0f},  // red
         {0.0f, 0.0f, 0.0f},  // black
         {1.0f, 1.0f, 0.0f},  // yellow
         {0.0f, 1.0f, 0.0f},  // green
                              // right
         {1.0f, 0.0f, 1.0f},  // magenta
         {1.0f, 0.0f, 0.0f},  // red
         {1.0f, 1.0f, 1.0f},  // white
         {1.0f, 1.0f, 0.0f},  // yellow
                              // left
         {0.0f, 0.0f, 0.0f},  // black
         {0.0f, 0.0f, 1.0f},  // blue
         {0.0f, 1.0f, 0.0f},  // green
         {0.0f, 1.0f, 1.0f},  // cyan
                              // top
         {0.0f, 1.0f, 1.0f},  // cyan
         {1.0f, 1.0f, 1.0f},  // white
         {0.0f, 1.0f, 0.0f},  // green
         {1.0f, 1.0f, 0.0f},  // yellow
                              // bottom
         {0.0f, 0.0f, 0.0f},  // black
         {1.0f, 0.0f, 0.0f},  // red
         {0.0f, 0.0f, 1.0f},  // blue
         {1.0f, 0.0f, 1.0f}}  // magenta
    ;

    viz::Colors((const float*)colors.data(), colors.size());

    std::vector<std::array<float, 3>> normals = {
        {+0.0f, +0.0f, +1.0f},  // forward
        {+0.0f, +0.0f, +1.0f},  // forward
        {+0.0f, +0.0f, +1.0f},  // forward
        {+0.0f, +0.0f, +1.0f},  // forward
                                // back
        {+0.0f, +0.0f, -1.0f},  // backbard
        {+0.0f, +0.0f, -1.0f},  // backbard
        {+0.0f, +0.0f, -1.0f},  // backbard
        {+0.0f, +0.0f, -1.0f},  // backbard
                                // right
        {+1.0f, +0.0f, +0.0f},  // right
        {+1.0f, +0.0f, +0.0f},  // right
        {+1.0f, +0.0f, +0.0f},  // right
        {+1.0f, +0.0f, +0.0f},  // right
                                // left
        {-1.0f, +0.0f, +0.0f},  // left
        {-1.0f, +0.0f, +0.0f},  // left
        {-1.0f, +0.0f, +0.0f},  // left
        {-1.0f, +0.0f, +0.0f},  // left
                                // top
        {+0.0f, +1.0f, +0.0f},  // up
        {+0.0f, +1.0f, +0.0f},  // up
        {+0.0f, +1.0f, +0.0f},  // up
        {+0.0f, +1.0f, +0.0f},  // up
                                // bottom
        {+0.0f, -1.0f, +0.0f},  // down
        {+0.0f, -1.0f, +0.0f},  // down
        {+0.0f, -1.0f, +0.0f},  // down
        {+0.0f, -1.0f, +0.0f}   // down
    };
    holoscan::viz::Normals((const float*)normals.data(), normals.size());

    holoscan::viz::Light(2.0, 2.0, 20.0, 0.0);

    struct timeval tv;
    uint64_t t;

    gettimeofday(&tv, NULL);

    t = ((tv.tv_sec * 1000 + tv.tv_usec / 1000) -
         (start_tv.tv_sec * 1000 + start_tv.tv_usec / 1000)) /
        5;
    // std::cout << "time factor" << t << std::endl;
    std::vector<std::array<float, 4>> rotations = {{45.0f + (0.25f * t), 1.0f, 0.0f, 0.0f},
                                                   {45.0f - (0.5f * t), 0.0f, 1.0f, 0.0f},
                                                   {10.0f + (0.15f * t), 0.0f, 0.0f, 1.0f}};
    std::vector<std::array<float, 3>> translations = {{0.0f, 0.0f, -8.0f}};

    holoscan::viz::Translations((const float*)translations.data()->data(), translations.size());
    holoscan::viz::Rotations((const float*)rotations.data()->data(), rotations.size());
    holoscan::viz::Frustum(-2.8f, +2.8f, -2.8f, +2.8f, 6.0f, 10.0f);
  }
  viz::EndLayer();
}

void HolovizOp::stop() {
  viz::Shutdown();
}

void HolovizOp::compute(InputContext& op_input, OutputContext& op_output,
                        ExecutionContext& context) {
  viz::BeginGeometryLayer();

  struct timeval tv;
  uint64_t t;

  gettimeofday(&tv, NULL);

  t = ((tv.tv_sec * 1000 + tv.tv_usec / 1000) -
       (start_tv.tv_sec * 1000 + start_tv.tv_usec / 1000)) /
      5;
  // std::cout << "time factor" << t << std::endl;
  std::vector<std::array<float, 4>> rotations = {{45.0f + (0.25f * t), 1.0f, 0.0f, 0.0f},
                                                 {45.0f - (0.5f * t), 0.0f, 1.0f, 0.0f},
                                                 {10.0f + (0.15f * t), 0.0f, 0.0f, 1.0f}};
  std::vector<std::array<float, 3>> translations = {{0.0f, 0.0f, -8.0f}};

  holoscan::viz::Translations((const float*)translations.data()->data(), translations.size());
  holoscan::viz::Rotations((const float*)rotations.data()->data(), rotations.size());
  holoscan::viz::Frustum(-2.8f, +2.8f, -2.8f, +2.8f, 6.0f, 10.0f);

  viz::EndLayer();
  viz::End();

  // check if the render buffer should be output
  if (render_buffer_output_enabled_) { read_frame_buffer(op_input, op_output, context); }
}

}  // namespace holoscan::ops
