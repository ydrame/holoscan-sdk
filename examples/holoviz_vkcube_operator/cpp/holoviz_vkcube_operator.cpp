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
#include <holoscan/operators/holovizvkcube/holovizvkcube.hpp>
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

  void setup(OperatorSpec& spec) override { spec.output<gxf::Entity>("outputs"); }

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

    op_output.emit(entity, "outputs");

    count_++;
  }

 private:
  std::shared_ptr<UnboundedAllocator> allocator_;
  uint32_t count_ = 0;
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

    auto source = make_operator<ops::GeometrySourceOp>("source");

    auto visualizer =
        make_operator<ops::HolovizOpVkCube>("holoviz", Arg("width", 854u), Arg("height", 480u));

    // Define the workflow: source -> holoviz
    add_flow(source, visualizer, {{"outputs", "receivers"}});
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
