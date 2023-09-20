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

#include "holoviz/holoviz.hpp"

#include <imgui.h>

#include <iostream>

#include "context.hpp"
#include "layers/geometry_layer.hpp"
#include "layers/image_layer.hpp"
#include "window.hpp"

namespace holoscan::viz {

void Init(GLFWwindow* window, InitFlags flags) {
  Context::get().init(window, flags);
}

void Init(uint32_t width, uint32_t height, const char* title, InitFlags flags) {
  Context::get().init(width, height, title, flags);
}

void Init(const char* displayName, uint32_t width, uint32_t height, uint32_t refreshRate,
          InitFlags flags) {
  Context::get().init(displayName, width, height, refreshRate, flags);
}

void ImGuiSetCurrentContext(ImGuiContext* context) {
  Context::get().im_gui_set_current_context(context);
}

void SetCudaStream(CUstream stream) {
  Context::get().set_cuda_stream(stream);
}

void SetFont(const char* path, float size_in_pixels) {
  Context::get().set_font(path, size_in_pixels);
}

bool WindowShouldClose() {
  return Context::get().get_window()->should_close();
}

bool WindowIsMinimized() {
  return Context::get().get_window()->is_minimized();
}

void Shutdown() {
  Context::get().shutdown();
}

void Begin() {
  Context::get().begin();
}

void End() {
  Context::get().end();
}

void BeginImageLayer() {
  Context::get().begin_image_layer();
}

void ImageCudaDevice(uint32_t w, uint32_t h, ImageFormat fmt, CUdeviceptr device_ptr,
                     size_t row_pitch) {
  Context::get().get_active_image_layer()->image_cuda_device(w, h, fmt, device_ptr, row_pitch);
}

void ImageCudaArray(ImageFormat fmt, CUarray array) {
  Context::get().get_active_image_layer()->image_cuda_array(fmt, array);
}

void ImageHost(uint32_t w, uint32_t h, ImageFormat fmt, const void* data, size_t row_pitch) {
  Context::get().get_active_image_layer()->image_host(w, h, fmt, data, row_pitch);
}

void LUT(uint32_t size, ImageFormat fmt, size_t data_size, const void* data, bool normalized) {
  Context::get().get_active_image_layer()->lut(size, fmt, data_size, data, normalized);
}

void BeginImGuiLayer() {
  Context::get().begin_im_gui_layer();
}

void BeginGeometryLayer() {
  Context::get().begin_geometry_layer();
}

void Color(float r, float g, float b, float a) {
  Context::get().get_active_geometry_layer()->color(r, g, b, a);
}

void Colors(const float* colors, int size) {
  Context::get().get_active_geometry_layer()->colors(colors, size);
}
void Normals(const float* normals, int size) {
  Context::get().get_active_geometry_layer()->normals(normals, size);
}

void Light(float r, float g, float b, float a) {
  Context::get().get_active_geometry_layer()->light(r, g, b, a);
}

void Translations(const float* translations, int size) {
  Context::get().get_active_geometry_layer()->translations(translations, size);
}
void Rotations(const float* rotations, int size) {
  Context::get().get_active_geometry_layer()->rotations(rotations, size);
}

void Frustum(float left, float right, float bottom, float top, float nearZ, float farZ) {
  Context::get().get_active_geometry_layer()->frustum(left, right, bottom, top, nearZ, farZ);
}

void LineWidth(float width) {
  Context::get().get_active_geometry_layer()->line_width(width);
}

void PointSize(float size) {
  Context::get().get_active_geometry_layer()->point_size(size);
}

void Text(float x, float y, float size, const char* text) {
  Context::get().get_active_geometry_layer()->text(x, y, size, text);
}

void Primitive(PrimitiveTopology topology, uint32_t primitive_count, size_t data_size,
               const float* data) {
  Context::get().get_active_geometry_layer()->primitive(topology, primitive_count, data_size, data);
}

void DepthMap(DepthMapRenderMode render_mode, uint32_t width, uint32_t height,
              ImageFormat depth_fmt, CUdeviceptr depth_device_ptr, ImageFormat color_fmt,
              CUdeviceptr color_device_ptr) {
  Context::get().get_active_geometry_layer()->depth_map(
      render_mode, width, height, depth_fmt, depth_device_ptr, color_fmt, color_device_ptr);
}

void LayerOpacity(float opacity) {
  Context::get().get_active_layer()->set_opacity(opacity);
}

void LayerPriority(int32_t priority) {
  Context::get().get_active_layer()->set_priority(priority);
}

void LayerAddView(float offset_x, float offset_y, float width, float height, const float* matrix) {
  Layer::View view;
  view.offset_x = offset_x;
  view.offset_y = offset_y;
  view.width = width;
  view.height = height;
  if (matrix) {
    // nvmath::mat4f is column major, the incoming matrix is row major, transpose while copying
    view.matrix = nvmath::mat4f(matrix[0],
                                matrix[4],
                                matrix[8],
                                matrix[12],
                                matrix[1],
                                matrix[5],
                                matrix[9],
                                matrix[13],
                                matrix[2],
                                matrix[6],
                                matrix[10],
                                matrix[14],
                                matrix[3],
                                matrix[7],
                                matrix[11],
                                matrix[15]);
  }
  Context::get().get_active_layer()->add_view(view);
}

void EndLayer() {
  Context::get().end_layer();
}

void ReadFramebuffer(ImageFormat fmt, uint32_t width, uint32_t height, size_t buffer_size,
                     CUdeviceptr device_ptr) {
  Context::get().read_framebuffer(fmt, width, height, buffer_size, device_ptr);
}

void GetCameraPose(size_t size, float* matrix) {
  if (size != 16) { throw std::invalid_argument("Size of the matrix array should be 16"); }
  if (matrix == nullptr) { throw std::invalid_argument("Pointer to matrix should not be nullptr"); }

  nvmath::mat4f view_matrix;
  Context::get().get_window()->get_view_matrix(&view_matrix);

  // nvmath::mat4f is column major, the outgoing matrix is row major, transpose while copying
  for (uint32_t row = 0; row < 4; ++row) {
    for (uint32_t col = 0; col < 4; ++col) { matrix[row * 4 + col] = view_matrix(row, col); }
  }
}

}  // namespace holoscan::viz
