/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/gpu/benchmarks/SyntheticBenchmark.h"

#include "velox/common/memory/Memory.h"

#include <folly/json.h>
#include <gtest/gtest.h>

namespace facebook::velox::gpu::benchmark {
namespace {

class SyntheticBenchmarkTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(SyntheticBenchmarkTest, deterministicData) {
  Options options{
      .rows = 17, .batchRows = 5, .filterPercent = 10, .groupCardinality = 3};
  auto first = makeData(pool_.get(), options);
  auto second = makeData(pool_.get(), options);

  ASSERT_EQ(first.fact.size(), 4);
  ASSERT_TRUE(first.factType->equivalent(*second.factType));
  ASSERT_TRUE(first.ordersType->equivalent(*second.ordersType));
  ASSERT_EQ(first.inputBytes, 17 * 10 * 8 + 5 * 4 * 8);

  for (size_t batch = 0; batch < first.fact.size(); ++batch) {
    ASSERT_TRUE(
        first.fact[batch]->equalValueAt(second.fact[batch].get(), 0, 0));
  }

  auto group = first.fact.front()->childAt(1)->as<SimpleVector<int64_t>>();
  auto quantity = first.fact.front()->childAt(3)->as<SimpleVector<int64_t>>();
  ASSERT_EQ(group->valueAt(4), 1);
  ASSERT_EQ(quantity->valueAt(4), 29);
}

TEST_F(SyntheticBenchmarkTest, workloadAndMetrics) {
  for (auto workload : allWorkloads()) {
    EXPECT_EQ(parseWorkload(workloadName(workload)), workload);
  }
  EXPECT_THROW(parseWorkload("missing"), VeloxUserError);
  EXPECT_EQ(percentile({1, 2, 3, 4, 5}, 0.5), 3);
  EXPECT_EQ(percentile({1, 2, 3, 4, 5}, 0.95), 5);

  Result result{
      .backend = "test",
      .workload = "scan",
      .rows = 100,
      .inputBytes = 800,
      .filterPercent = 10,
      .groupCardinality = 20,
      .warmups = 2,
      .repetitions = 3,
      .checksum = 42,
      .expectedChecksum = 42,
      .coldMs = 10,
      .warmMs = {1, 2, 3}};
  auto json = folly::parseJson(resultJson(result));
  EXPECT_EQ(json["backend"], "test");
  EXPECT_EQ(json["correct"], true);
  EXPECT_EQ(json["filter_percent"], 10);
  EXPECT_EQ(json["group_cardinality"], 20);
  EXPECT_EQ(json["warmups"], 2);
  EXPECT_EQ(json["repetitions"], 3);
  EXPECT_EQ(json["warm_median_ms"], 2);
  EXPECT_NE(resultCsv(result).find("test,scan"), std::string::npos);
}

} // namespace
} // namespace facebook::velox::gpu::benchmark
