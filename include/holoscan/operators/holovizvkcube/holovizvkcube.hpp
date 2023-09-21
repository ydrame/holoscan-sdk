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

#ifndef HOLOSCAN_OPERATORS_HOLOVIZVKCUBE_HOLOVIZ_HPP
#define HOLOSCAN_OPERATORS_HOLOVIZVKCUBE_HOLOVIZ_HPP

#include "holoscan/operators/holoviz/holoviz.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class for data visualization.
 *
 * This high-speed viewer handles compositing, blending, and visualization of RGB or RGBA images,
 * masks, geometric primitives, text and depth maps. The operator can auto detect the format of the
 * input tensors when only the `receivers` parameter list is specified. Else the input specification
 * can be set at creation time using the `tensors` parameter or at runtime when passing input
 * specifications to the `input_specs` port.
 *
 * 1. Parameters
 *
 *    - **`receivers`**: List of input queues to component accepting `gxf::Tensor` or
 *      `gxf::VideoBuffer`
 *      - type: `std::vector<gxf::Handle<gxf::Receiver>>`
 *    - **`enable_render_buffer_input`**: Enable `render_buffer_input`, (default: `false`)
 *      - type: `bool`
 *    - **`render_buffer_input`**: Input for an empty render buffer, type `gxf::VideoBuffer`
 *      - type: `gxf::Handle<gxf::Receiver>`
 *    - **`enable_render_buffer_output`**: Enable `render_buffer_output`, (default: `false`)
 *      - type: `bool`
 *    - **`render_buffer_output`**: Output for a filled render buffer. If an input render buffer is
 *      specified at `render_buffer_input` it uses that one, otherwise it allocates a new buffer.
 *      - type: `gxf::Handle<gxf::Transmitter>`
 *    - **`enable_camera_pose_output`**: Enable `camera_pose_output`, (default: `false`)
 *      - type: `bool`
 *    - **`camera_pose_output`**: Output the camera pose. The camera parameters are returned in a
 *      4x4 row major projection matrix.
 *      - type: `std::array<float, 16>`
 *    - **`tensors`**: List of input tensor specifications (default: `[]`)
 *      - type: `std::vector<InputSpec>`
 *        - **`name`**: name of the tensor containing the input data to display
 *          - type: `std::string`
 *        - **`type`**: input type (default `"unknown"`)
 *          - type: `std::string`
 *          - possible values:
 *            **`unknown`**: unknown type, the operator tries to guess the type by inspecting the
 *            tensor
 *            **`color`**: RGB or RGBA color 2d image
 *            **`color_lut`**: single channel 2d image, color is looked up
 *            **`points`**: point primitives, one coordinate (x, y) per primitive
 *            **`lines`**: line primitives, two coordinates (x0, y0) and (x1, y1) per primitive
 *            **`line_strip`**: line strip primitive, a line primitive i is defined by each
 *            coordinate (xi, yi) and the following (xi+1, yi+1)
 *            **`triangles`**: triangle primitive, three coordinates (x0, y0), (x1, y1) and (x2, y2)
 *            per primitive
 *            **`crosses`**: cross primitive, a cross is defined by the center coordinate and the
 *            size (xi, yi, si)
 *            **`rectangles`**: axis aligned rectangle primitive, each rectangle is defined by two
 *            coordinates (xi, yi) and (xi+1, yi+1)
 *            **`ovals`**: oval primitive, an oval primitive is defined by the center coordinate and
 *            the axis sizes (xi, yi, sxi, syi)
 *            **`text`**: text is defined by the top left coordinate and the size (x, y, s) per
 *            string, text strings are defined by InputSpec member **`text`**
 *            **`depth_map`**: single channel 2d array where each element represents a depth value.
 *            The data is rendered as a 3d object using points, lines or triangles. The color for
 *            the elements can be specified through `depth_map_color`. Supported format: 8-bit
 *            unsigned normalized format that has a single 8-bit depth component
 *            **`depth_map_color`**: RGBA 2d image, same size as the depth map. One color value for
 *            each element of the depth map grid. Supported format: 32-bit unsigned normalized
 *            format that has an 8-bit R component in byte 0, an 8-bit G component in byte 1, an
 *            8-bit B component in byte 2, and an 8-bit A component in byte 3.
 *        - **`opacity`**: layer opacity, 1.0 is fully opaque, 0.0 is fully transparent (default:
 *          `1.0`)
 *          - type: `float`
 *        - **`priority`**: layer priority, determines the render order, layers with higher priority
 *            values are rendered on top of layers with lower priority values (default: `0`)
 *          - type: `int32_t`
 *        - **`color`**: RGBA color of rendered geometry (default: `[1.f, 1.f, 1.f, 1.f]`)
 *          - type: `std::vector<float>`
 *        - **`line_width`**: line width for geometry made of lines (default: `1.0`)
 *          - type: `float`
 *        - **`point_size`**: point size for geometry made of points (default: `1.0`)
 *          - type: `float`
 *        - **`text`**: array of text strings, used when `type` is text. (default: `[]`)
 *          - type: `std::vector<std::string>`
 *        - **`depth_map_render_mode`**: depth map render mode (default: `points`)
 *          -type `std::string`
 *          - possible values:
 *            **`points`**: render as points
 *            **`lines`**: render as lines
 *            **`triangles`**: render as triangles
 *    - **`color_lut`**: Color lookup table for tensors of type 'color_lut', vector of four float
 *      RGBA values
 *      - type: `std::vector<std::vector<float>>`
 *    - **`window_title`**: Title on window canvas (default: `Holoviz`)
 *      - type: `std::string`
 *    - **`display_name`**: In exclusive mode, name of display to use as shown with xrandr (default:
 *      `DP-0`)
 *      - type: `std::string`
 *    - **`width`**: Window width or display resolution width if in exclusive or fullscreen mode
 *      (default: `1920`)
 *      - type: `uint32_t`
 *    - **`height`**: Window height or display resolution height if in exclusive or fullscreen mode
 *      (default: `1080`)
 *      - type: `uint32_t`
 *    - **`framerate`**: Display framerate if in exclusive mode (default: `60`)
 *      - type: `uint32_t`
 *    - **`use_exclusive_display`**: Enable exclusive display (default: `false`)
 *      - type: `bool`
 *    - **`fullscreen`**: Enable fullscreen window (default: `false`)
 *      - type: `bool`
 *    - **`headless`**: Enable headless mode. No window is opened, the render buffer is output to
 *      `render_buffer_output`. (default: `false`)
 *      - type: `bool`
 *    - **`window_close_scheduling_term`**: BooleanSchedulingTerm to stop the codelet from ticking
 *      when the window is closed
 *      - type: `gxf::Handle<gxf::BooleanSchedulingTerm>`
 *    - **`allocator`**: Allocator used to allocate memory for `render_buffer_output`
 *      - type: `gxf::Handle<gxf::Allocator>`
 *    - **`font_path`**: File path for the font used for rendering text.
 *      - type: `std::string`
 *    - **`cuda_stream_pool`**: Instance of gxf::CudaStreamPool
 *      - type: `gxf::Handle<gxf::CudaStreamPool>`
 *
 * 2. Displaying Color Images
 *
 *    Image data can either be on host or device (GPU). Multiple image formats are supported
 *    - R 8 bit unsigned
 *    - R 16 bit unsigned
 *    - R 16 bit float
 *    - R 32 bit unsigned
 *    - R 32 bit float
 *    - RGB 8 bit unsigned
 *    - BGR 8 bit unsigned
 *    - RGBA 8 bit unsigned
 *    - BGRA 8 bit unsigned
 *    - RGBA 16 bit unsigned
 *    - RGBA 16 bit float
 *    - RGBA 32 bit float
 *
 *    When the `type` parameter is set to `color_lut` the final color is looked up using the values
 *    from the `color_lut` parameter. For color lookups these image formats are supported
 *    - R 8 bit unsigned
 *    - R 16 bit unsigned
 *    - R 32 bit unsigned
 *
 * 3. Drawing Geometry
 *
 *    In all cases, `x` and `y` are normalized coordinates in the range `[0, 1]`. The `x` and `y`
 *    correspond to the horizontal and vertical axes of the display, respectively. The origin `(0,
 *    0)` is at the top left of the display. All coordinates should be defined using a single
 *    precision float data type. Geometric primitives outside of the visible area are clipped.
 *    Coordinate arrays are expected to have the shape `(1, N, C)` where `N` is the coordinate count
 *    and `C` is the component count for each coordinate.
 *
 *    - Points are defined by a `(x, y)` coordinate pair.
 *    - Lines are defined by a set of two `(x, y)` coordinate pairs.
 *    - Lines strips are defined by a sequence of `(x, y)` coordinate pairs. The first two
 *      coordinates define the first line, each additional coordinate adds a line connecting to the
 *      previous coordinate.
 *    - Triangles are defined by a set of three `(x, y)` coordinate pairs.
 *    - Crosses are defined by `(x, y, size)` tuples. `size` specifies the size of the cross in the
 *      `x` direction and is optional, if omitted it's set to `0.05`. The size in the `y` direction
 *      is calculated using the aspect ratio of the window to make the crosses square.
 *    - Rectangles (bounding boxes) are defined by a pair of 2-tuples defining the upper-left and
 *      lower-right coordinates of a box: `(x1, y1), (x2, y2)`.
 *    - Ovals are defined by `(x, y, size_x, size_y)` tuples. `size_x` and `size_y` are optional, if
 *      omitted they are set to `0.05`.
 *    - Texts are defined by `(x, y, size)` tuples. `size` specifies the size of the text in `y`
 *      direction and is optional, if omitted it's set to `0.05`. The size in the `x` direction is
 *      calculated using the aspect ratio of the window. The index of each coordinate references a
 *      text string from the `text` parameter and the index is clamped to the size of the text
 *      array. For example, if there is one item set for the `text` parameter, e.g.
 *      `text=['my_text']` and three coordinates, then `my_text` is rendered three times. If
 *      `text=['first text', 'second text']` and three coordinates are specified, then `first text`
 *      is rendered at the first coordinate, `second text` at the second coordinate and then `second
 *      text` again at the third coordinate. The `text` string array is fixed and can't be changed
 *      after initialization. To hide text which should not be displayed, specify coordinates
 *      greater than `(1.0, 1.0)` for the text item, the text is then clipped away.
 *    - 3D Points are defined by a `(x, y, z)` coordinate tuple.
 *    - 3D Lines are defined by a set of two `(x, y, z)` coordinate tuples.
 *    - 3D Lines strips are defined by a sequence of `(x, y, z)` coordinate tuples. The first two
 *      coordinates define the first line, each additional coordinate adds a line connecting to the
 *      previous coordinate.
 *    - 3D Triangles are defined by a set of three `(x, y, z)` coordinate tuples.
 *
 * 4. Displaying Depth Maps
 *
 *    When `type` is `depth_map` the provided data is interpreted as a rectangular array of depth
 *    values. Additionally a 2d array with a color value for each point in the grid can be specified
 *    by setting `type` to `depth_map_color`.
 *
 *    The type of geometry drawn can be selected by setting `depth_map_render_mode`.
 *
 *    Depth maps are rendered in 3D and support camera movement. The camera is controlled using the
 *    mouse:
 *    - Orbit        (LMB)
 *    - Pan          (LMB + CTRL  | MMB)
 *    - Dolly        (LMB + SHIFT | RMB | Mouse wheel)
 *    - Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)
 *    - Zoom         (Mouse wheel + SHIFT)
 *
 * 5. Output
 *
 *    By default a window is opened to display the rendering, but the extension can also be run in
 *    headless mode with the `headless` parameter.
 *
 *    Using a display in exclusive mode is also supported with the `use_exclusive_display`
 *    parameter. This reduces the latency by avoiding the desktop compositor.
 *
 *    The rendered framebuffer can be output to `render_buffer_output`.
 *
 */
class HolovizOpVkCube : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(HolovizOpVkCube, holoscan::ops::HolovizOp);
  HolovizOpVkCube() = default;
  // template <typename ArgT, typename... ArgsT>
  // HolovizOpVkCube(ArgT&& arg, ArgsT&&... args)
  //   : HolovizOp(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
};
}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_HOLOVIZ_HOLOVIZ_HPP */
