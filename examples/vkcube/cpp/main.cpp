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
extern struct model cube_model;

namespace holoscan::ops {

class VkCubeOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VkCubeOp)

  VkCubeOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(window_close_scheduling_term_,
               "window_close_scheduling_term",
               "WindowCloseSchedulingTerm",
               "BooleanSchedulingTerm to stop the codelet from ticking when the window is closed.");
  }
  void initialize() {
    auto frag = fragment();

    window_close_scheduling_term_ =
        frag->make_condition<holoscan::BooleanCondition>("window_close_scheduling_term");
    add_arg(window_close_scheduling_term_.get());

    int argc_ = 1;
    char** argv_;
    // auto argc__ = args();
    //  auto argv__ = argv();
    //   spec.param(argc, "multiplier", "Multiplier", "Multiply the input by this value", 1);
    //    spec.param(argv, "multiplier", "Multiplier", "Multiply the input by this value", "");
    // auto argv__ = &(args().data());
    parse_args(argc_, argv_);

    vc.model = cube_model;
    // vc.gbm_device = NULL;
#if defined(ENABLE_XCB)
    vc.xcb.window = XCB_NONE;
#endif
    vc.width = width;
    vc.height = height;
    vc.protected_ = protected_chain;
    gettimeofday(&vc.start_tv, NULL);
    Operator::initialize();
  }
  void start() {
    init_display(&vc);
    window_close_scheduling_term_->enable_tick();
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    std::cout << std::endl;
    std::cout << "Hello World!" << std::endl;
    std::cout << std::endl;
    auto windowShouldClose = mainloop(&vc);
    // cast Condition to BooleanCondition
    if (windowShouldClose) {
      window_close_scheduling_term_->disable_tick();
      return;
    }
  }

 private:
  struct vkcube vc;
  Parameter<int> argc;
  Parameter<char**> argv;
  Parameter<std::shared_ptr<BooleanCondition>> window_close_scheduling_term_;
};

}  // namespace holoscan::ops

class VkCubeApp : public holoscan::Application {
 public:
  explicit VkCubeApp(int argc, char** argv) : argc(argc), argv(argv) {}
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto hello = make_operator<ops::VkCubeOp>(
        "hello", Arg("argc", argc), Arg("argv", argv), make_condition<CountCondition>(1));

    // Define the one-operator workflow
    add_operator(hello);
  }

 private:
  int argc;
  char** argv;
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<VkCubeApp>(argc, argv);
  app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>(
      "greedy-scheduler", holoscan::Arg("max_duration_ms", 10000L)));
  app->run();
  return 0;
}
