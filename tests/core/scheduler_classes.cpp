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

#include <gtest/gtest.h>
#include <gxf/core/gxf.h>

#include <memory>
#include <string>

#include "common/assert.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"
#include "holoscan/core/resources/gxf/manual_clock.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"
#include "holoscan/core/schedulers/gxf/greedy_scheduler.hpp"
#include "holoscan/core/schedulers/gxf/multithread_scheduler.hpp"
#include "../utils.hpp"

using namespace std::string_literals;

namespace holoscan {

using SchedulerClassesWithGXFContext = TestWithGXFContext;

TEST(SchedulerClasses, TestGreedyScheduler) {
  Fragment F;
  const std::string name{"greedy-scheduler"};
  auto scheduler = F.make_scheduler<GreedyScheduler>(name);
  EXPECT_EQ(scheduler->name(), name);
  EXPECT_EQ(typeid(scheduler), typeid(std::make_shared<GreedyScheduler>()));
  EXPECT_EQ(std::string(scheduler->gxf_typename()), "nvidia::gxf::GreedyScheduler"s);
}

TEST_F(SchedulerClassesWithGXFContext, TestGreedySchedulerWithArgs) {
  const std::string name{"greedy-scheduler"};
  ArgList arglist{
      Arg{"name", name},
      Arg{"stop_on_deadlock", false},
      Arg{"max_duration_ms", 10000LL},
      Arg{"check_recession_period_ms", 0.1},
      Arg{"stop_on_deadlock_timeout", -1LL},
  };
  auto scheduler = F.make_scheduler<GreedyScheduler>(name, arglist);
}

TEST_F(SchedulerClassesWithGXFContext, TestGreedySchedulerWithManualClock) {
  const std::string name{"multithread-scheduler"};
  ArgList arglist{Arg{"clock", F.make_resource<ManualClock>()}};
  auto scheduler = F.make_scheduler<GreedyScheduler>(name, arglist);
}

TEST_F(SchedulerClassesWithGXFContext, TestGreedySchedulerWithRealtimeClock) {
  const std::string name{"multithread-scheduler"};
  ArgList arglist{Arg{"clock", F.make_resource<RealtimeClock>()}};
  auto scheduler = F.make_scheduler<GreedyScheduler>(name, arglist);
}

TEST(SchedulerClasses, TestMultiThreadScheduler) {
  Fragment F;
  const std::string name{"multithread-scheduler"};
  auto scheduler = F.make_scheduler<MultiThreadScheduler>(name);
  EXPECT_EQ(scheduler->name(), name);
  EXPECT_EQ(typeid(scheduler), typeid(std::make_shared<MultiThreadScheduler>()));
  EXPECT_EQ(std::string(scheduler->gxf_typename()), "nvidia::gxf::MultiThreadScheduler"s);
}

TEST_F(SchedulerClassesWithGXFContext, TestMultiThreadSchedulerWithArgs) {
  const std::string name{"multithread-scheduler"};
  ArgList arglist{
      Arg{"name", name},
      Arg{"worker_thread_number", 4L},
      Arg{"stop_on_deadlock", false},
      Arg{"check_recession_period_ms", 5.0},
      Arg{"max_duration_ms", 10000L},
      Arg{"stop_on_deadlock_timeout", 100LL},
  };
  auto scheduler = F.make_scheduler<MultiThreadScheduler>(name, arglist);
}

TEST_F(SchedulerClassesWithGXFContext, TestMultiThreadSchedulerWithManualClock) {
  const std::string name{"multithread-scheduler"};
  ArgList arglist{Arg{"clock", F.make_resource<ManualClock>()}};
  auto scheduler = F.make_scheduler<MultiThreadScheduler>(name, arglist);
}

TEST_F(SchedulerClassesWithGXFContext, TestMultiThreadSchedulerWithRealtimeClock) {
  const std::string name{"multithread-scheduler"};
  ArgList arglist{Arg{"clock", F.make_resource<RealtimeClock>()}};
  auto scheduler = F.make_scheduler<MultiThreadScheduler>(name, arglist);
}

}  // namespace holoscan
