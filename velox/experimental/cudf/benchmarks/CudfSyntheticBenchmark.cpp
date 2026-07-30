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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/exec/AggregationRegistry.h"
#include "velox/experimental/cudf/exec/PrestoAggregateFunctions.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/gpu/benchmarks/SyntheticBenchmark.h"

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <cudf/utilities/default_stream.hpp>

#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/init/Init.h>

#include <chrono>
#include <functional>
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
DEFINE_string(
    synthetic_cudf_scan,
    "cpu",
    "Parquet read path when --synthetic_input=parquet: cpu (CPU Velox reader "
    "feeding a Values plan, matched with Wave) or gpu (native "
    "CudfHiveConnector TableScan, all-GPU, total time only)");
DEFINE_int64(
    synthetic_cudf_chunk_read_bytes,
    0,
    "Upper bound in BYTES on the table cuDF returns per read, which is the "
    "GPU-scan analogue of a batch size. 0 means no limit, reading a whole "
    "split as one batch. This is not a row count");
DEFINE_int64(
    synthetic_cudf_pass_read_bytes,
    0,
    "Advisory bound in BYTES on cuDF read and decompression scratch memory. "
    "0 means no limit");
DEFINE_bool(
    synthetic_cudf_buffered_input,
    true,
    "Read through Velox BufferedInput (true) or KvikIO (false)");

namespace facebook::velox::cudf_velox {
namespace {

using gpu::benchmark::DataSet;
using gpu::benchmark::Options;
using gpu::benchmark::Result;
using gpu::benchmark::Workload;

RowVectorPtr runPlan(const core::PlanNodePtr& plan, memory::MemoryPool* pool) {
  return velox::exec::test::AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool);
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

bool gpuScanRequested() {
  if (FLAGS_synthetic_cudf_scan == "cpu") {
    return false;
  }
  VELOX_USER_CHECK_EQ(
      FLAGS_synthetic_cudf_scan,
      "gpu",
      "--synthetic_cudf_scan must be cpu or gpu");
  VELOX_USER_CHECK(
      gpu::benchmark::parquetInputRequested(),
      "--synthetic_cudf_scan=gpu requires --synthetic_input=parquet");
  return true;
}

/// Registers a CudfHiveConnector so a TableScan naming kCudfHiveConnectorId
/// reads Parquet on the GPU, and tears it down in the order the connector
/// requires: executor first so no read is in flight, then the connector.
class GpuScanConnector {
 public:
  GpuScanConnector() {
    // Without a local filesystem the reader logs a warning and silently falls
    // back to KvikIO, which would measure an I/O path we did not configure.
    filesystems::registerLocalFileSystem();
    ioExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(8);

    auto properties = std::unordered_map<std::string, std::string>{
        {connector::hive::CudfHiveConfig::kMaxChunkReadLimit,
         std::to_string(FLAGS_synthetic_cudf_chunk_read_bytes)},
        {connector::hive::CudfHiveConfig::kMaxPassReadLimit,
         std::to_string(FLAGS_synthetic_cudf_pass_read_bytes)},
        {connector::hive::CudfHiveConfig::kAllowMismatchedCudfHiveSchemas,
         "true"},
        {connector::hive::CudfHiveConfig::kUseBufferedInput,
         FLAGS_synthetic_cudf_buffered_input ? "true" : "false"}};

    connector::hive::CudfHiveConnectorFactory factory;
    auto connector = factory.newConnector(
        exec::test::kCudfHiveConnectorId,
        std::make_shared<config::ConfigBase>(std::move(properties)),
        ioExecutor_.get());
    velox::connector::ConnectorRegistry::global().insert(
        connector->connectorId(), connector);
  }

  /// A plain HiveConnector under this id would silently give us the CPU
  /// reader: canRunOnGPU would just return false, and allowCpuFallback=false
  /// does not catch it because the adapter still keeps the operator. Only an
  /// explicit type check rules that out.
  static void assertGpuConnectorRegistered() {
    auto registered = velox::connector::ConnectorRegistry::tryGet(
        exec::test::kCudfHiveConnectorId);
    VELOX_CHECK_NOT_NULL(
        registered,
        "GPU scan mode found no connector under '{}'",
        exec::test::kCudfHiveConnectorId);
    VELOX_CHECK_NOT_NULL(
        std::dynamic_pointer_cast<connector::hive::CudfHiveConnector>(
            registered),
        "GPU scan mode requires a CudfHiveConnector under '{}', but found a {}."
        " A plain HiveConnector would read on the CPU without failing",
        exec::test::kCudfHiveConnectorId,
        typeid(*registered).name());
  }

  ~GpuScanConnector() {
    ioExecutor_.reset();
    velox::connector::ConnectorRegistry::global().erase(
        exec::test::kCudfHiveConnectorId);
  }

  GpuScanConnector(const GpuScanConnector&) = delete;
  GpuScanConnector& operator=(const GpuScanConnector&) = delete;

 private:
  std::unique_ptr<folly::IOThreadPoolExecutor> ioExecutor_;
};

/// Scan-rooted plan for the GPU reader. Carries no subfield or remaining
/// filter: pushing filters into the reader would turn them into a cuDF AST
/// evaluated during decode, which is different work from what the Values plan
/// measures.
core::PlanNodePtr makeGpuScanPlan(
    Workload workload,
    memory::MemoryPool* pool,
    const RowTypePtr& factType,
    const Options& options,
    core::PlanNodeId& scanNodeId) {
  auto tableHandle = exec::test::CudfHiveConnectorTestBase::makeTableHandle(
      "synthetic_fact", factType);
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto source = velox::exec::test::PlanBuilder(ids, pool)
                    .startTableScan()
                    .outputType(factType)
                    .tableHandle(tableHandle)
                    .endTableScan();
  scanNodeId = source.planNode()->id();
  DataSet scanData;
  scanData.factType = factType;
  return gpu::benchmark::finishPlan(workload, source, ids, scanData, options);
}

std::vector<std::shared_ptr<velox::connector::ConnectorSplit>> makeGpuScanSplits(
    const std::string& path) {
  // One split per file: start and length are byte-range hints that cuDF
  // resolves to whole row groups, so an arbitrary split would change the row
  // count rather than divide the work.
  return {velox::connector::hive::HiveConnectorSplitBuilder(path)
              .connectorId(exec::test::kCudfHiveConnectorId)
              .fileFormat(dwio::common::FileFormat::PARQUET)
              .build()};
}

/// Batch shape as the GPU scan actually produced it. The host-side batch
/// vectors do not exist in this mode, so the numbers come from the scan
/// operator instead of being assumed from the byte budget.
void recordScanNodeShape(
    const std::shared_ptr<velox::exec::Task>& task,
    const core::PlanNodeId& scanNodeId,
    Result& result) {
  auto stats = velox::exec::toPlanStats(task->taskStats());
  auto it = stats.find(scanNodeId);
  if (it == stats.end()) {
    result.numBatches = -1;
    result.maxBatchRows = -1;
    result.minBatchRows = -1;
    return;
  }
  const auto& scan = it->second;
  result.numBatches = scan.outputVectors;
  // Only the average is available per node, so report it in both slots rather
  // than inventing a spread the stats cannot support.
  const auto rowsPerBatch = scan.outputVectors > 0
      ? scan.outputRows / static_cast<int64_t>(scan.outputVectors)
      : 0;
  result.maxBatchRows = rowsPerBatch;
  result.minBatchRows = rowsPerBatch;
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
  const bool gpuScan = gpuScanRequested();
  VELOX_USER_CHECK(
      !gpuScan || !gpu::benchmark::workloadNeedsOrders(workload),
      "GPU scan mode does not cover the join workloads: they need a second "
      "scanned source and a post-join materialization round trip");

  // Stage 1. Any CPU Parquet read runs before registerCudf() installs its
  // global driver adapter, so the scan stays on CPU for both backends.
  gpu::benchmark::ScannedInput scanned;
  gpu::benchmark::ParquetFixture fixture;
  if (gpuScan) {
    // The fixture still has to exist, but reading it is the measured plan's
    // job, so only write it here.
    fixture = gpu::benchmark::ensureParquetFixture(options, workload, pool);
  } else if (parquetInput) {
    scanned = gpu::benchmark::prepareParquetInput(options, workload, pool);
  }

  uint64_t expectedChecksum = 0;
  if (FLAGS_synthetic_cpu_validation) {
    // GPU scan mode has no host data, so the oracle pays for its own untimed
    // CPU read of the same fixture.
    DataSet validationData;
    if (gpuScan) {
      double scanMs = 0;
      validationData = gpu::benchmark::readParquetDataSet(
          fixture.dir,
          options,
          fixture.options,
          fixture.withOrders,
          pool,
          scanMs);
    } else if (parquetInput) {
      validationData = scanned.data;
    } else {
      validationData =
          gpu::benchmark::makeDataForWorkload(pool, options, workload);
    }
    auto expectedPlan =
        gpu::benchmark::makeValuesPlan(workload, pool, validationData, options);
    expectedChecksum = gpu::benchmark::resultChecksum(
        executeWorkload(workload, expectedPlan, pool));
  }

  // Stage 2. GPU compute. In Parquet mode the plan reads plain host vectors, so
  // the CudfFromVelox upload falls inside the measured window instead of being
  // hoisted out by a toGpu() pre-pass.
  std::unique_ptr<GpuScanConnector> gpuScanConnector;
  if (gpuScan) {
    gpuScanConnector = std::make_unique<GpuScanConnector>();
  }

  const auto coldStart = std::chrono::steady_clock::now();
  auto data = (parquetInput && !gpuScan)
      ? scanned.data
      : (gpuScan ? DataSet{} : gpu::benchmark::makeDataForWorkload(pool, options, workload));
  if (gpuScan) {
    data.factType = gpu::benchmark::factRowType();
    data.inputBytes =
        options.rows * data.factType->size() * static_cast<int64_t>(sizeof(int64_t));
  }
  const auto hostBatches = data.fact;
  CudfConfig::getInstance().allowCpuFallback = false;
  registerCudf();
  registerPrestoFunctions("");
  registerPrestoAggregateFunctions("");
  if (gpuScan) {
    GpuScanConnector::assertGpuConnectorRegistered();
  }
  if (!parquetInput) {
    data.fact = toGpu(data.fact, pool);
    data.orders = toGpu(data.orders, pool);
  }

  core::PlanNodeId scanNodeId;
  std::shared_ptr<velox::exec::Task> coldTask;
  RowVectorPtr coldResult;
  if (gpuScan) {
    auto plan = makeGpuScanPlan(
        workload, pool, data.factType, options, scanNodeId);
    coldResult = velox::exec::test::AssertQueryBuilder(plan)
                     .maxDrivers(1)
                     .splits(scanNodeId, makeGpuScanSplits(fixture.factPath))
                     .copyResults(pool, coldTask);
  } else {
    auto plan = gpu::benchmark::makeValuesPlan(workload, pool, data, options);
    coldResult = executeWorkload(workload, plan, pool);
  }
  const auto coldMs = elapsedMs(coldStart);
  const auto checksum = gpu::benchmark::resultChecksum(coldResult);
  if (!FLAGS_synthetic_cpu_validation) {
    expectedChecksum = checksum;
  }
  VELOX_CHECK_EQ(checksum, expectedChecksum);

  std::function<std::pair<double, RowVectorPtr>()> runWarm;
  DataSet baseGpuData;
  if (gpuScan) {
    runWarm = [&]() {
      core::PlanNodeId iterationScanId;
      auto iterationPlan = makeGpuScanPlan(
          workload, pool, data.factType, options, iterationScanId);
      auto splits = makeGpuScanSplits(fixture.factPath);
      const auto start = std::chrono::steady_clock::now();
      auto result = velox::exec::test::AssertQueryBuilder(iterationPlan)
                        .maxDrivers(1)
                        .splits(iterationScanId, splits)
                        .copyResults(pool);
      return std::pair{elapsedMs(start), std::move(result)};
    };
  } else if (parquetInput) {
    runWarm = [&]() {
      auto iterationPlan =
          gpu::benchmark::makeValuesPlan(workload, pool, data, options);
      const auto start = std::chrono::steady_clock::now();
      auto result = executeWorkload(workload, iterationPlan, pool);
      return std::pair{elapsedMs(start), std::move(result)};
    };
  } else {
    auto hostData =
        gpu::benchmark::makeDataForWorkload(pool, options, workload);
    baseGpuData = DataSet{
        .factType = hostData.factType,
        .fact = toGpu(hostData.fact, pool),
        .ordersType = hostData.ordersType,
        .orders = toGpu(hostData.orders, pool),
        .inputBytes = hostData.inputBytes};
    runWarm = [&]() {
      auto iterationData = cloneGpuData(baseGpuData, pool);
      auto iterationPlan = gpu::benchmark::makeValuesPlan(
          workload, pool, iterationData, options);
      const auto start = std::chrono::steady_clock::now();
      auto result = executeWorkload(workload, iterationPlan, pool);
      return std::pair{elapsedMs(start), std::move(result)};
    };
  }

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

  Result result{
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
  if (gpuScan) {
    // The scan is fused into the measured plan, so there is no separable scan
    // number. Leaving the scan fields at zero keeps total == compute, and
    // scan_mode is what says those zeros mean "not separable" rather than
    // "free".
    result.inputMode = "parquet";
    result.scanMode = "gpu-decode";
    result.rowGroupRows = fixture.options.rowGroupRows;
    result.scanBatchRows = 0;
    recordScanNodeShape(coldTask, scanNodeId, result);
  } else if (parquetInput) {
    gpu::benchmark::applyScannedInput(scanned, hostBatches, result);
  } else {
    gpu::benchmark::recordBatchShape(hostBatches, result);
  }
  return result;
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
