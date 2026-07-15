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

#pragma once

#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/ComplexVector.h"

#include <cstdint>
#include <string>
#include <vector>

namespace facebook::velox::gpu::benchmark {

enum class Workload {
  kScan,
  kFilter,
  kProject,
  kAggregate,
  kGroupBy,
  kJoinAggregate,
  kQ1,
  kQ6,
  kQ3,
};

struct Options {
  int64_t rows{1'000'000};
  int32_t batchRows{100'000};
  int32_t filterPercent{10};
  int32_t groupCardinality{1'000};
};

struct DataSet {
  RowTypePtr factType;
  std::vector<RowVectorPtr> fact;
  RowTypePtr ordersType;
  std::vector<RowVectorPtr> orders;
  int64_t inputBytes{0};
};

struct Result {
  std::string backend;
  std::string workload;
  int64_t rows{0};
  int64_t inputBytes{0};
  int32_t filterPercent{0};
  int32_t groupCardinality{0};
  int32_t warmups{0};
  int32_t repetitions{0};
  uint64_t checksum{0};
  uint64_t expectedChecksum{0};
  double coldMs{0};
  std::vector<double> warmMs;
};

std::vector<Workload> allWorkloads();

std::string workloadName(Workload workload);

Workload parseWorkload(const std::string& name);

DataSet makeData(memory::MemoryPool* pool, const Options& options);

DataSet makeDataForWorkload(
    memory::MemoryPool* pool,
    const Options& options,
    Workload workload);

core::PlanNodePtr finishPlan(
    Workload workload,
    exec::test::PlanBuilder& source,
    const std::shared_ptr<core::PlanNodeIdGenerator>& planNodeIdGenerator,
    const DataSet& data,
    const Options& options,
    bool sourceFiltersPushedDown = false);

core::PlanNodePtr makeValuesPlan(
    Workload workload,
    memory::MemoryPool* pool,
    const DataSet& data,
    const Options& options);

core::PlanNodePtr makePostJoinPlan(
    Workload workload,
    memory::MemoryPool* pool,
    const RowVectorPtr& joined);

core::PlanNodePtr finishPostJoinPlan(
    Workload workload,
    exec::test::PlanBuilder& source,
    bool sourceFiltersPushedDown = false);

uint64_t resultChecksum(const RowVectorPtr& result);

double percentile(std::vector<double> samples, double pct);

std::string resultJson(const Result& result);

std::string csvHeader();

std::string resultCsv(const Result& result);

} // namespace facebook::velox::gpu::benchmark
