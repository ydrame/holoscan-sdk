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

#include <holoscan/holoscan.hpp>
#include <iostream>

extern "C" {
#include "common.h"
}

namespace holoscan::ops {

class VkCubeOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VkCubeOp)

  VkCubeOp() = default;

  void setup(OperatorSpec& spec) override {}

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    std::cout << std::endl;
    std::cout << "Hello World!" << std::endl;
    std::cout << std::endl;
  }
};

}  // namespace holoscan::ops

class VkCubeApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto hello = make_operator<ops::VkCubeOp>("hello", make_condition<CountCondition>(1));

    // Define the one-operator workflow
    add_operator(hello);
  }
};

extern struct model cube_model;

int main(int argc, char** argv) {
  /*auto app = holoscan::make_application<VkCubeApp>();
  app->run();
*/
  struct vkcube vc;

  parse_args(argc, argv);

  vc.model = cube_model;
  // vc.gbm_device = NULL;
#if defined(ENABLE_XCB)
  vc.xcb.window = XCB_NONE;
#endif
  vc.width = width;
  vc.height = height;
  vc.protected_ = protected_chain;
  gettimeofday(&vc.start_tv, NULL);

  init_display(&vc);
  mainloop(&vc);
  return 0;
}
