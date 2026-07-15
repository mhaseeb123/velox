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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/AggregationRegistry.h"
#include "velox/experimental/cudf/exec/PrestoAggregateFunctions.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/gpu/benchmarks/SyntheticBenchmark.h"

#include "velox/common/memory/Memory.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <cudf/utilities/default_stream.hpp>

#include <folly/init/Init.h>

#include <chrono>
#include <iostream>

DEFINE_string(
    synthetic_workload,
    "q6",
    "Workload: scan, filter, project, aggregate, group_by, join_aggregate, q1, q6, or q3");
DEFINE_int64(synthetic_rows, 1'000'000, "Number of fact rows");
DEFINE_int32(synthetic_batch_rows, 100'000, "Rows per input batch");
DEFINE_int32(synthetic_filter_percent, 10, "Filter selectivity percentage");
DEFINE_int32(synthetic_group_cardinality, 1'000, "Group-by cardinality");
DEFINE_int32(synthetic_warmups, 2, "Number of untimed warmup runs");
DEFINE_int32(synthetic_repetitions, 10, "Number of measured warm runs");
DEFINE_string(synthetic_format, "json", "Output format: json or csv");
DEFINE_bool(
    synthetic_cpu_validation,
    true,
    "Validate results against CPU Velox before benchmarking");

namespace facebook::velox::cudf_velox {
namespace {

using gpu::benchmark::DataSet;
using gpu::benchmark::Options;
using gpu::benchmark::Result;
using gpu::benchmark::Workload;

RowVectorPtr runPlan(const core::PlanNodePtr& plan, memory::MemoryPool* pool) {
  return exec::test::AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool);
}

RowVectorPtr executeWorkload(
    Workload workload,
    const core::PlanNodePtr& plan,
    memory::MemoryPool* pool) {
  auto result = runPlan(plan, pool);
  if (workload == Workload::kJoinAggregate || workload == Workload::kQ3) {
    auto postJoin = gpu::benchmark::makePostJoinPlan(workload, pool, result);
    return runPlan(postJoin, pool);
  }
  return result;
}

std::vector<RowVectorPtr> toGpu(
    const std::vector<RowVectorPtr>& input,
    memory::MemoryPool* pool) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<RowVectorPtr> output;
  output.reserve(input.size());
  for (const auto& batch : input) {
    auto table = with_arrow::toCudfTable(batch, pool, stream, mr);
    VELOX_CHECK_NOT_NULL(table);
    output.push_back(
        std::make_shared<CudfVector>(
            pool, batch->type(), batch->size(), std::move(table), stream));
  }
  stream.synchronize();
  return output;
}

std::vector<RowVectorPtr> cloneGpu(
    const std::vector<RowVectorPtr>& input,
    memory::MemoryPool* pool) {
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<RowVectorPtr> output;
  output.reserve(input.size());
  for (const auto& batch : input) {
    auto cudfBatch = std::dynamic_pointer_cast<CudfVector>(batch);
    VELOX_CHECK_NOT_NULL(cudfBatch);
    auto stream = cudfBatch->stream();
    auto table =
        std::make_unique<cudf::table>(cudfBatch->getTableView(), stream, mr);
    output.push_back(
        std::make_shared<CudfVector>(
            pool, batch->type(), batch->size(), std::move(table), stream));
  }
  return output;
}

DataSet cloneGpuData(const DataSet& input, memory::MemoryPool* pool) {
  return DataSet{
      .factType = input.factType,
      .fact = cloneGpu(input.fact, pool),
      .ordersType = input.ordersType,
      .orders = cloneGpu(input.orders, pool),
      .inputBytes = input.inputBytes};
}

double elapsedMs(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

Result runBenchmark(memory::MemoryPool* pool) {
  VELOX_USER_CHECK_GT(FLAGS_synthetic_repetitions, 0);
  VELOX_USER_CHECK_GE(FLAGS_synthetic_warmups, 0);
  const auto workload = gpu::benchmark::parseWorkload(FLAGS_synthetic_workload);
  const Options options{
      .rows = FLAGS_synthetic_rows,
      .batchRows = FLAGS_synthetic_batch_rows,
      .filterPercent = FLAGS_synthetic_filter_percent,
      .groupCardinality = FLAGS_synthetic_group_cardinality};

  uint64_t expectedChecksum = 0;
  if (FLAGS_synthetic_cpu_validation) {
    auto validationData =
        gpu::benchmark::makeDataForWorkload(pool, options, workload);
    auto expectedPlan =
        gpu::benchmark::makeValuesPlan(workload, pool, validationData, options);
    expectedChecksum = gpu::benchmark::resultChecksum(
        executeWorkload(workload, expectedPlan, pool));
  }

  const auto coldStart = std::chrono::steady_clock::now();
  auto data = gpu::benchmark::makeDataForWorkload(pool, options, workload);
  CudfConfig::getInstance().allowCpuFallback = false;
  registerCudf();
  registerPrestoFunctions("");
  registerPrestoAggregateFunctions("");
  data.fact = toGpu(data.fact, pool);
  data.orders = toGpu(data.orders, pool);
  auto plan = gpu::benchmark::makeValuesPlan(workload, pool, data, options);
  auto coldResult = executeWorkload(workload, plan, pool);
  const auto coldMs = elapsedMs(coldStart);
  const auto checksum = gpu::benchmark::resultChecksum(coldResult);
  if (!FLAGS_synthetic_cpu_validation) {
    expectedChecksum = checksum;
  }
  VELOX_CHECK_EQ(checksum, expectedChecksum);

  auto hostData = gpu::benchmark::makeDataForWorkload(pool, options, workload);
  DataSet baseGpuData{
      .factType = hostData.factType,
      .fact = toGpu(hostData.fact, pool),
      .ordersType = hostData.ordersType,
      .orders = toGpu(hostData.orders, pool),
      .inputBytes = hostData.inputBytes};

  auto runWarm = [&]() {
    auto iterationData = cloneGpuData(baseGpuData, pool);
    auto iterationPlan =
        gpu::benchmark::makeValuesPlan(workload, pool, iterationData, options);
    const auto start = std::chrono::steady_clock::now();
    auto result = executeWorkload(workload, iterationPlan, pool);
    return std::pair{elapsedMs(start), std::move(result)};
  };

  for (int32_t i = 0; i < FLAGS_synthetic_warmups; ++i) {
    runWarm();
  }

  std::vector<double> samples;
  samples.reserve(FLAGS_synthetic_repetitions);
  for (int32_t i = 0; i < FLAGS_synthetic_repetitions; ++i) {
    auto [elapsed, result] = runWarm();
    samples.push_back(elapsed);
    VELOX_CHECK_EQ(gpu::benchmark::resultChecksum(result), expectedChecksum);
  }

  return Result{
      .backend = "cudf",
      .workload = gpu::benchmark::workloadName(workload),
      .rows = options.rows,
      .inputBytes = data.inputBytes,
      .filterPercent = options.filterPercent,
      .groupCardinality = options.groupCardinality,
      .warmups = FLAGS_synthetic_warmups,
      .repetitions = FLAGS_synthetic_repetitions,
      .checksum = checksum,
      .expectedChecksum = expectedChecksum,
      .coldMs = coldMs,
      .warmMs = std::move(samples)};
}

} // namespace
} // namespace facebook::velox::cudf_velox

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  facebook::velox::functions::prestosql::registerAllScalarFunctions();
  facebook::velox::aggregate::prestosql::registerAllAggregateFunctions();
  facebook::velox::parse::registerTypeResolver();
  auto rootPool = facebook::velox::memory::memoryManager()->addRootPool(
      "CudfSyntheticBenchmark");
  auto pool = rootPool->addLeafChild("CudfSyntheticBenchmark");
  const auto result = facebook::velox::cudf_velox::runBenchmark(pool.get());
  if (FLAGS_synthetic_format == "csv") {
    std::cout << facebook::velox::gpu::benchmark::csvHeader() << '\n'
              << facebook::velox::gpu::benchmark::resultCsv(result) << '\n';
  } else {
    VELOX_USER_CHECK_EQ(FLAGS_synthetic_format, "json");
    std::cout << facebook::velox::gpu::benchmark::resultJson(result) << '\n';
  }
  facebook::velox::cudf_velox::unregisterFunctions();
  facebook::velox::cudf_velox::unregisterAggregateFunctions();
  facebook::velox::cudf_velox::unregisterCudf();
  return 0;
}
