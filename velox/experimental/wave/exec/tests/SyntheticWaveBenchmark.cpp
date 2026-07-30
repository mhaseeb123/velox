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

#include "velox/benchmarks/QueryBenchmarkBase.h"
#include "velox/exec/ExchangeSource.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/experimental/gpu/benchmarks/SyntheticBenchmark.h"
#include "velox/experimental/wave/exec/ToWave.h"
#include "velox/experimental/wave/exec/WaveHiveDataSource.h"
#include "velox/experimental/wave/exec/tests/utils/FileFormat.h"
#include "velox/experimental/wave/exec/tests/utils/WaveTestSplitReader.h"

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

namespace facebook::velox::wave {
namespace {

using gpu::benchmark::DataSet;
using gpu::benchmark::Options;
using gpu::benchmark::Result;
using gpu::benchmark::Workload;

class BenchmarkEnvironment : public QueryBenchmarkBase {
 public:
  void runMain(std::ostream&, RunStats&) override {}
};

core::PlanNodeId tableScanNodeId(const core::PlanNodePtr& node) {
  if (std::dynamic_pointer_cast<const core::TableScanNode>(node)) {
    return node->id();
  }
  for (const auto& source : node->sources()) {
    auto id = tableScanNodeId(source);
    if (!id.empty()) {
      return id;
    }
  }
  return "";
}

RowVectorPtr runPlan(
    const core::PlanNodePtr& plan,
    const test::SplitVector& splits,
    memory::MemoryPool* pool) {
  auto builder = exec::test::AssertQueryBuilder(plan);
  if (!splits.empty()) {
    const auto scanId = tableScanNodeId(plan);
    VELOX_CHECK(!scanId.empty());
    builder.splits(scanId, splits);
  }
  return builder.maxDrivers(1).copyResults(pool);
}

RowVectorPtr executeWorkload(
    Workload workload,
    const core::PlanNodePtr& plan,
    const test::SplitVector& splits,
    memory::MemoryPool* pool,
    bool useMockPostJoin) {
  auto result = runPlan(plan, splits, pool);
  if (workload == Workload::kJoinAggregate || workload == Workload::kQ3) {
    if (!useMockPostJoin) {
      auto postJoin = gpu::benchmark::makePostJoinPlan(workload, pool, result);
      return runPlan(postJoin, {}, pool);
    }
    test::Table::dropTable("synthetic_post_join");
    auto postSplits =
        test::Table::defineTable("synthetic_post_join", {result})->splits();
    std::vector<std::string> postFilters;
    if (workload == Workload::kQ3) {
      postFilters = {"shipdate > 1000", "orderdate < 1000"};
    }
    auto source = exec::test::PlanBuilder(pool).tableScan(
        asRowType(result->type()), postFilters);
    auto postJoin = gpu::benchmark::finishPostJoinPlan(
        workload, source, !postFilters.empty());
    return runPlan(postJoin, postSplits, pool);
  }
  return result;
}

core::PlanNodePtr makeWaveMockPlan(
    Workload workload,
    memory::MemoryPool* pool,
    const DataSet& data,
    const Options& options) {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  if (workload == Workload::kJoinAggregate || workload == Workload::kQ3) {
    auto source = exec::test::PlanBuilder(ids, pool).values(data.fact);
    return gpu::benchmark::finishPlan(
        workload, source, ids, data, options, false);
  }
  std::vector<std::string> filters;
  switch (workload) {
    case Workload::kFilter:
      filters.push_back(fmt::format("value < {}", options.filterPercent * 10));
      break;
    case Workload::kQ1:
      filters.push_back("shipdate <= 1500");
      break;
    case Workload::kQ6:
      filters = {
          "shipdate between 500 and 1499",
          "discount between 4 and 6",
          "quantity < 24"};
      break;
    default:
      break;
  }
  auto source =
      exec::test::PlanBuilder(ids, pool).tableScan(data.factType, filters);
  return gpu::benchmark::finishPlan(
      workload, source, ids, data, options, !filters.empty());
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
  const bool parquetInput = gpu::benchmark::parquetInputRequested();

  // Stage 1. The CPU Parquet read has to finish before registerWave() installs
  // its global driver adapter, otherwise the scan itself would be routed to
  // Wave, which has no Parquet split reader.
  gpu::benchmark::ScannedInput scanned;
  if (parquetInput) {
    scanned = gpu::benchmark::prepareParquetInput(options, workload, pool);
  }

  uint64_t expectedChecksum = 0;
  if (FLAGS_synthetic_cpu_validation) {
    auto validationData = parquetInput
        ? scanned.data
        : gpu::benchmark::makeDataForWorkload(pool, options, workload);
    auto expectedPlan =
        gpu::benchmark::makeValuesPlan(workload, pool, validationData, options);
    expectedChecksum = gpu::benchmark::resultChecksum(
        executeWorkload(workload, expectedPlan, {}, pool, false));
  }

  // Stage 2. GPU compute over host vectors that are already in memory.
  const auto coldStart = std::chrono::steady_clock::now();
  auto data = parquetInput
      ? scanned.data
      : gpu::benchmark::makeDataForWorkload(pool, options, workload);
  registerWave();
  test::SplitVector splits;
  core::PlanNodePtr plan;
  if (parquetInput) {
    plan = gpu::benchmark::makeValuesPlan(workload, pool, data, options);
  } else {
    WaveHiveDataSource::registerConnector();
    test::WaveTestSplitReader::registerTestSplitReader();
    exec::ExchangeSource::factories().clear();
    exec::ExchangeSource::registerFactory(exec::test::createLocalExchangeSource);
    if (!gpu::benchmark::workloadNeedsOrders(workload)) {
      test::Table::dropTable("synthetic_benchmark");
      splits =
          test::Table::defineTable("synthetic_benchmark", data.fact)->splits();
    }
    plan = makeWaveMockPlan(workload, pool, data, options);
  }
  const bool useMockPostJoin = !parquetInput;
  auto coldResult =
      executeWorkload(workload, plan, splits, pool, useMockPostJoin);
  const auto coldMs = elapsedMs(coldStart);
  const auto checksum = gpu::benchmark::resultChecksum(coldResult);
  if (!FLAGS_synthetic_cpu_validation) {
    expectedChecksum = checksum;
  }
  VELOX_CHECK_EQ(checksum, expectedChecksum);

  for (int32_t i = 0; i < FLAGS_synthetic_warmups; ++i) {
    executeWorkload(workload, plan, splits, pool, useMockPostJoin);
  }

  std::vector<double> samples;
  samples.reserve(FLAGS_synthetic_repetitions);
  for (int32_t i = 0; i < FLAGS_synthetic_repetitions; ++i) {
    const auto start = std::chrono::steady_clock::now();
    auto result = executeWorkload(workload, plan, splits, pool, useMockPostJoin);
    samples.push_back(elapsedMs(start));
    VELOX_CHECK_EQ(gpu::benchmark::resultChecksum(result), expectedChecksum);
  }

  Result result{
      .backend = "wave",
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
  if (parquetInput) {
    gpu::benchmark::applyScannedInput(scanned, data.fact, result);
  } else {
    gpu::benchmark::recordBatchShape(data.fact, result);
  }
  return result;
}

} // namespace
} // namespace facebook::velox::wave

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv, false};
  facebook::velox::wave::BenchmarkEnvironment environment;
  environment.initialize();
  auto rootPool = facebook::velox::memory::memoryManager()->addRootPool(
      "SyntheticWaveBenchmark");
  auto pool = rootPool->addLeafChild("SyntheticWaveBenchmark");
  const auto result = facebook::velox::wave::runBenchmark(pool.get());
  if (FLAGS_synthetic_format == "csv") {
    std::cout << facebook::velox::gpu::benchmark::csvHeader() << '\n'
              << facebook::velox::gpu::benchmark::resultCsv(result) << '\n';
  } else {
    VELOX_USER_CHECK_EQ(FLAGS_synthetic_format, "json");
    std::cout << facebook::velox::gpu::benchmark::resultJson(result) << '\n';
  }
  facebook::velox::wave::test::Table::dropAll();
  environment.shutdown();
  return 0;
}
