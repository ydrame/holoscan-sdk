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
#include <chrono>

#include <holoscan/holoscan.hpp>
#include <iostream>
#include <string>

extern "C" {
#include "common.h"
}
extern struct model cube_model;

namespace holoscan::ops {

std::string concatenate(int argc, char* argv[]) {
  if (argc < 1) { return ""; }
  std::string result(argv[0]);
  for (int i = 1; i < argc; ++i) {
    result += " ";
    result += argv[i];
  }
  return result;
}

class VkCubeOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VkCubeOp)

  VkCubeOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(window_close_scheduling_term_,
               "window_close_scheduling_term",
               "WindowCloseSchedulingTerm",
               "BooleanSchedulingTerm to stop the codelet from ticking when the window is closed.");
    spec.param(argc_,
               "argv",
               "argv",
               "BooleanSchedulingTerm to stop the codelet from ticking when the window is closed.");
    spec.param(argvchar_,
               "argvchar",
               "argvchar",
               "BooleanSchedulingTerm to stop the codelet from ticking when the window is closed.");
  }
  void initialize() {
    auto frag = fragment();

    window_close_scheduling_term_ =
        frag->make_condition<holoscan::BooleanCondition>("window_close_scheduling_term");
    add_arg(window_close_scheduling_term_.get());

    using namespace std::chrono_literals;

    auto has_argv = std::find_if(
        args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "argvchar"); });
    auto has_argc = std::find_if(
        args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "argc"); });

    if (has_argv != args().end() && has_argc != args().end()) {
      int argcval = std::any_cast<int>(has_argc->value());
      char** value = std::any_cast<char**>((*has_argv).value());
      parse_args(argcval, value);

    } else {
      int argc__ = 1;
      char** argv__;
      parse_args(argc__, argv__);
    }

    vc.model = cube_model;
    // vc.gbm_device = NULL;
#if defined(ENABLE_XCB)
    vc.xcb.window = XCB_NONE;
#endif
#if defined(ENABLE_WAYLAND)
    vc.wl.surface = NULL;
#endif
    vc.width = width;
    vc.height = height;
    vc.protected_ = protected_chain;

    gettimeofday(&vc.start_tv, NULL);
    init_display(&vc);
    Operator::initialize();
  }
  void start() { window_close_scheduling_term_->enable_tick(); }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto windowShouldClose = default_display ? renderloop(&vc) : renderloop_alternate_display(&vc);
    if (windowShouldClose) {
      window_close_scheduling_term_->disable_tick();
      return;
    }
  }

 private:
  struct vkcube vc;
  Parameter<int> argc_;
  Parameter<char**> argvchar_;
  Parameter<std::string> argv_;
  Parameter<std::shared_ptr<BooleanCondition>> window_close_scheduling_term_;
  Parameter<std::shared_ptr<PeriodicCondition>> periodic_scheduling_term_;
};

}  // namespace holoscan::ops

class VkCubeApp : public holoscan::Application {
 public:
  explicit VkCubeApp(int argc, char** argv) : argc(argc), argv(argv) {}
  void compose() override {
    using namespace holoscan;
    // Define the operators
    auto vkCubeOp = make_operator<ops::VkCubeOp>("hello", Arg("argc", argc), Arg("argvchar", argv));

    // Define the one-operator workflow
    add_operator(vkCubeOp);
  }

 private:
  int argc;
  char** argv;
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<VkCubeApp>(argc, argv);
  app->run();
  return 0;
}
