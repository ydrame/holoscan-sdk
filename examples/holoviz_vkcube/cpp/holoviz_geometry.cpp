/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <getopt.h>

#include <sys/time.h>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include <gxf/std/tensor.hpp>

#include "util/esUtil.h"

namespace holoscan::ops {

/**
 * Example of an operator generating geometric primitives to be displayed by the HolovizOp
 *
 *  This operator has:
 *       outputs: "output_tensor"
 *       output_specs: "output_specs"
 *       transform_specs : "transform_specs"
 */
class GeometrySourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GeometrySourceOp)

  GeometrySourceOp() = default;

  void initialize() override {
    // Create an allocator for the operator
    allocator_ = fragment()->make_resource<UnboundedAllocator>("pool");
    // Add the allocator to the operator so that it is initialized
    add_arg(allocator_);

    // Call the base class initialize function
    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override {
    spec.output<gxf::Entity>("outputs");
    // spec.output<std::vector<HolovizOp::InputSpec>>("output_specs");
    // spec.output<std::vector<HolovizOp::InputSpec>>("transform_specs");

    gettimeofday(&start_tv, NULL);
  }

  /**
   * Helper function to add a tensor with data to an entity.
   */
  template <std::size_t N, std::size_t C>
  void add_data(gxf::Entity& entity, const char* name,
                const std::array<std::array<float, C>, N>& data, ExecutionContext& context) {
    // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());
    // add a tensor
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name).value();
    // reshape the tensor to the size of the data
    tensor->reshape<float>(
        nvidia::gxf::Shape({1, N, C}), nvidia::gxf::MemoryStorageType::kHost, allocator.value());
    // copy the data to the tensor
    std::memcpy(tensor->pointer(), data.data(), N * C * sizeof(float));
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto entity = gxf::Entity::New(&context);
    auto specs = std::vector<HolovizOp::InputSpec>();

    // if (count_ == 0) {
    add_data<72, 3>(entity,
                    "vbo",
                    {{
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
                        {1.0f, 0.0f, 1.0f},  // magenta

                        // front
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
                    }},
                    context);
    /*add_data<2, 2>(entity, "transforms", {{{0.f, 0.f}}}, context);
    add_data<1, 1>(entity, "update", {{{1.0f}}}, context);
    */

    op_output.emit(entity, "outputs");

    /*HolovizOp::InputSpec spec;
    spec.tensor_name_ = "transforms";
    spec.type_ = HolovizOp::InputType::TRANSFORM;

    struct timeval tv;
    uint64_t t;

    gettimeofday(&tv, NULL);

    t = ((tv.tv_sec * 1000 + tv.tv_usec / 1000) -
         (start_tv.tv_sec * 1000 + start_tv.tv_usec / 1000)) /
        5;

    // spec.text_.push_back(std::string("Frame ") + std::to_string(count_));
    spec.transform_names.push_back(std::string("translation"));
    spec.transform_values.push_back({0.0f, 0.0f, -8.0f});

    spec.transform_names.push_back(std::string("rotation"));
    spec.transform_values.push_back({45.0f + (0.25f * t), 1.0f, 0.0f, 0.0f});

    spec.transform_names.push_back(std::string("rotation"));
    spec.transform_values.push_back({45.0f - (0.5f * t), 0.0f, 1.0f, 0.0f});

    spec.transform_names.push_back(std::string("rotation"));
    spec.transform_values.push_back({10.0f + (0.15f * t), 0.0f, 0.0f, 1.0f});

    spec.transform_names.push_back(std::string("frustum"));
    spec.transform_values.push_back({2.8f, +2.8f, -2.8f, +2.8f, 6.0f, 10.0f});

    specs.push_back(spec);

    // emit the transformspecs
    op_output.emit(specs, "transform_specs");*/

    count_++;
  }
  struct ubo {
    ESMatrix modelview;
    ESMatrix modelviewprojection;
    float normal[12];
  };

 private:
  std::shared_ptr<UnboundedAllocator> allocator_;
  uint32_t count_ = 0;
  struct timeval start_tv;
  struct ubo ubo;
};

}  // namespace holoscan::ops

/**
 * Example of an application that uses the operators defined above.
 *
 * This application has the following operators:
 *
 * - VideoStreamReplayerOp
 * - GeometrySourceOp
 * - HolovizOp
 *
 * The VideoStreamReplayerOp reads a video file and sends the frames to the ImageProcessingOp.
 * The GeometrySourceOp creates geometric primitives and sends it to the HolovizOp.
 * The HolovizOp displays the processed frames and geometry.
 */
class HolovizGeometryApp : public holoscan::Application {
 public:
  /**
   * @brief Construct a new HolovizGeometryApp object
   *
   * @param count Limits the number of frames to show before the application ends.
   *   Set to 0 by default. The video stream will not automatically stop.
   *   Any positive integer will limit on the number of frames displayed.
   */
  explicit HolovizGeometryApp(uint64_t count) : count_(count) {}

  void compose() override {
    using namespace holoscan;

    // Define the replayer, geometry source and holoviz operators
    /*auto replayer = make_operator<ops::VideoStreamReplayerOp>(
        "replayer",
        Arg("directory", std::string("../data/endoscopy/video")),
        Arg("basename", std::string("surgical_video")),
        Arg("frame_rate", 0.f),
        Arg("repeat", true),
        Arg("realtime", true),
        Arg("count", count_));
        */

    auto source = make_operator<ops::GeometrySourceOp>("source");

    // build the input spec list
    std::vector<ops::HolovizOp::InputSpec> input_spec;
    int32_t priority = 0;

    auto& vbo_spec =
        input_spec.emplace_back(ops::HolovizOp::InputSpec("vbo", ops::HolovizOp::InputType::VBO));
    vbo_spec.color_ = {1.0f, 1.0f, 1.0f, 1.0f};
    vbo_spec.priority_ = priority++;

    // Parameters defining the vbo primitives
    auto& update_spec = input_spec.emplace_back(
        ops::HolovizOp::InputSpec("update", ops::HolovizOp::InputType::UPDATE));
    update_spec.priority_ = priority++;

    // Parameters defining the vbo primitives
    auto& tranform_spec = input_spec.emplace_back(
        ops::HolovizOp::InputSpec("transforms", ops::HolovizOp::InputType::TRANSFORM));
    tranform_spec.priority_ = priority++;
    /*auto& video_spec =
      input_spec.emplace_back(ops::HolovizOp::InputSpec("", ops::HolovizOp::InputType::COLOR));
  video_spec.line_width_ = 2.f;
  video_spec.opacity_ = 0.5f;
  video_spec.priority_ = priority++;

  // Parameters defining the rectangle primitives
  auto& boxes_spec = input_spec.emplace_back(
      ops::HolovizOp::InputSpec("boxes", ops::HolovizOp::InputType::RECTANGLES));
  boxes_spec.line_width_ = 2.f;
  boxes_spec.color_ = {1.0f, 0.0f, 1.0f, 0.5f};
  boxes_spec.priority_ = priority++;

  // line strip reuses the rectangle coordinates. This will make
  // a connected set of line segments through the diagonals of
  // each box.
  auto& line_strip_spec = input_spec.emplace_back(
      ops::HolovizOp::InputSpec("boxes", ops::HolovizOp::InputType::LINE_STRIP));
  line_strip_spec.line_width_ = 3.f;
  line_strip_spec.color_ = {0.4f, 0.4f, 1.0f, 0.7f};
  line_strip_spec.priority_ = priority++;

  // Lines also reuses the boxes coordinates so will plot a set of
  // disconnected line segments along the box diagonals.
  auto& lines_spec = input_spec.emplace_back(
      ops::HolovizOp::InputSpec("boxes", ops::HolovizOp::InputType::LINES));
  lines_spec.line_width_ = 3.f;
  lines_spec.color_ = {0.4f, 1.0f, 0.4f, 0.7f};
  lines_spec.priority_ = priority++;

  // Parameters defining the triangle primitives
  auto& triangles_spec = input_spec.emplace_back(
      ops::HolovizOp::InputSpec("triangles", ops::HolovizOp::InputType::TRIANGLES));
  triangles_spec.color_ = {1.0f, 0.0f, 0.0f, 0.5f};
  triangles_spec.priority_ = priority++;

  // Parameters defining the crosses primitives
  auto& crosses_spec = input_spec.emplace_back(
      ops::HolovizOp::InputSpec("crosses", ops::HolovizOp::InputType::CROSSES));
  crosses_spec.line_width_ = 3.f;
  crosses_spec.color_ = {0.0f, 1.0f, 0.0f, 1.0f};
  crosses_spec.priority_ = priority++;

  // Parameters defining the ovals primitives
  auto& ovals_spec = input_spec.emplace_back(
      ops::HolovizOp::InputSpec("ovals", ops::HolovizOp::InputType::OVALS));
  ovals_spec.opacity_ = 0.5f;
  ovals_spec.line_width_ = 2.f;
  ovals_spec.color_ = {1.0f, 1.0f, 1.0f, 1.0f};
  ovals_spec.priority_ = priority++;

  // Parameters defining the points primitives
  auto& points_spec = input_spec.emplace_back(
      ops::HolovizOp::InputSpec("points", ops::HolovizOp::InputType::POINTS));
  points_spec.point_size_ = 4.f;
  points_spec.color_ = {1.0f, 1.0f, 1.0f, 1.0f};
  points_spec.priority_ = priority++;

  // Parameters defining the label_coords primitives
  auto& label_coords_spec = input_spec.emplace_back(
      ops::HolovizOp::InputSpec("label_coords", ops::HolovizOp::InputType::TEXT));
  label_coords_spec.color_ = {1.0f, 1.0f, 1.0f, 1.0f};
  label_coords_spec.text_ = {"label_1", "label_2"};
  label_coords_spec.priority_ = priority++;
  */

    auto visualizer = make_operator<ops::HolovizOp>(
        "holoviz", Arg("width", 854u), Arg("height", 480u), Arg("tensors", input_spec));

    // Define the workflow: source -> holoviz
    add_flow(source, visualizer, {{"outputs", "receivers"}});
    // add_flow(source, visualizer, {{"transform_specs", "input_transform_specs"}});
    // add_flow(source, visualizer, {{"output_specs", "input_specs"}});
    // add_flow(replayer, visualizer, {{"output", "receivers"}});
  }

 private:
  uint64_t count_ = 0;
};

int main(int argc, char** argv) {
  // Parse args
  struct option long_options[] = {
      {"help", no_argument, 0, 'h'}, {"count", required_argument, 0, 'c'}, {0, 0, 0, 0}};
  uint64_t count;
  while (true) {
    int option_index = 0;
    const int c = getopt_long(argc, argv, "hc:", long_options, &option_index);

    if (c == -1) { break; }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                  << "Options:" << std::endl
                  << "  -h, --help     display this information" << std::endl
                  << "  -c, --count    Set the number of frames to display the video" << std::endl
                  << std::endl;
        return EXIT_SUCCESS;

      case 'c':
        count = std::stoull(argument);
        break;
      default:
        throw std::runtime_error("Unhandled option ");
    }
  }

  auto app = holoscan::make_application<HolovizGeometryApp>(count);
  app->run();

  return 0;
}
