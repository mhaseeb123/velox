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

#include "velox/common/compression/Compression.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/ComplexVector.h"

#include <cstdint>
#include <memory>
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

/// Shape of an on-disk Parquet fixture plus the batch size the reader is asked
/// for. Only the write-time fields take part in the fixture cache key;
/// 'scanBatchRows' is a read-time knob and the same file serves any value.
struct ParquetOptions {
  int64_t rowGroupRows{100'000};
  int64_t pageBytes{1 << 20};
  bool dictionary{false};
  common::CompressionKind compression{common::CompressionKind_NONE};
  int64_t scanBatchRows{100'000};
};

struct DataSet {
  /// Keeps the scan tasks that own 'fact' and 'orders' alive. Reading Parquet
  /// hands out the reader's own vectors instead of copying them, so the tasks
  /// must outlive the batches. Declared first so it is destroyed last.
  std::vector<std::shared_ptr<void>> owners{};
  RowTypePtr factType;
  std::vector<RowVectorPtr> fact;
  RowTypePtr ordersType;
  std::vector<RowVectorPtr> orders;
  int64_t inputBytes{0};

  /// Releases the batches before the tasks that own them. Plain assignment
  /// would release members in declaration order and drop the tasks first.
  void reset() {
    fact.clear();
    orders.clear();
    owners.clear();
  }
};

struct Result {
  std::string backend;
  std::string workload;
  std::string inputMode{"generated"};
  /// Where and how the Parquet decode happened.
  ///
  /// - "cpu-prepass": decoded on the CPU by dwio::common::RowReader in a
  ///   materialise-everything pass before the measured pipeline. Both Wave and
  ///   cuDF under --synthetic_cudf_scan=cpu do this.
  /// - "cpu-inpipeline": the same CPU decode moved inside the pipeline. Phase
  ///   B's Wave reader; nothing here emits it yet.
  /// - "gpu-decode": velox-cudf's CudfHiveConnector decoding on the device.
  ///   Fused into the measured plan, so the scan fields are zero and only the
  ///   total means anything.
  ///
  /// Wave is never "gpu-decode": it decodes on the CPU in both phases and then
  /// copies to the device, so a Wave total shares no scan floor with a
  /// gpu-decode total.
  std::string scanMode{"cpu-prepass"};
  int64_t rows{0};
  int64_t inputBytes{0};
  int32_t filterPercent{0};
  int32_t groupCardinality{0};
  int32_t warmups{0};
  int32_t repetitions{0};
  uint64_t checksum{0};
  uint64_t expectedChecksum{0};
  /// Rows in the largest and smallest batch the input consists of. In Parquet
  /// mode these show what the reader actually produced.
  int64_t maxBatchRows{0};
  int64_t minBatchRows{0};
  int64_t numBatches{0};
  int64_t rowGroupRows{0};
  int64_t scanBatchRows{0};
  /// CPU Parquet scan, first (cold) read and any repeated (warm) reads. Empty
  /// in generated mode.
  double scanColdMs{0};
  std::vector<double> scanWarmMs{};
  /// GPU stage including the host-to-device upload. This is what the
  /// pre-Parquet results measured.
  double coldMs{0};
  std::vector<double> warmMs;
};

std::vector<Workload> allWorkloads();

/// True for the workloads that join against the 'orders' build side.
bool workloadNeedsOrders(Workload workload);

std::string workloadName(Workload workload);

Workload parseWorkload(const std::string& name);

DataSet makeData(memory::MemoryPool* pool, const Options& options);

DataSet makeDataForWorkload(
    memory::MemoryPool* pool,
    const Options& options,
    Workload workload);

RowTypePtr factRowType();

RowTypePtr ordersRowType();

/// Fact rows [begin, begin + size). Row content depends only on the row index
/// and 'options.groupCardinality', so batching never changes the data.
RowVectorPtr makeFactBatch(
    memory::MemoryPool* pool,
    const Options& options,
    int64_t begin,
    vector_size_t size);

RowVectorPtr makeOrdersBatch(
    memory::MemoryPool* pool,
    int64_t begin,
    vector_size_t size);

/// Number of build-side rows the join workloads use for 'options.rows' facts.
int64_t ordersRowCount(const Options& options);

#ifdef VELOX_ENABLE_PARQUET

/// Directory holding the fixture for this row count and file shape. Read-time
/// options are excluded, so one directory serves every scan batch size.
std::string parquetFixtureDir(
    const std::string& root,
    const Options& options,
    const ParquetOptions& parquetOptions);

bool parquetFixtureExists(const std::string& dir, bool withOrders);

/// Generates and writes the fixture one batch at a time, so writing 1B rows
/// does not need 1B rows of host memory.
void writeParquetFixture(
    const std::string& dir,
    const Options& options,
    const ParquetOptions& parquetOptions,
    bool withOrders,
    memory::MemoryPool* pool);

/// Reads the fixture back with a CPU-only Hive scan. Must be called before any
/// GPU backend is registered, because both install global driver adapters.
/// Sets 'scanMs' to the wall time of the scan.
DataSet readParquetDataSet(
    const std::string& dir,
    const Options& options,
    const ParquetOptions& parquetOptions,
    bool withOrders,
    memory::MemoryPool* pool,
    double& scanMs);

#endif // VELOX_ENABLE_PARQUET

/// An on-disk fixture that is known to exist, and where its files are. The
/// cuDF GPU scan reads these paths itself instead of going through
/// 'readParquetDataSet'.
struct ParquetFixture {
  std::string dir;
  std::string factPath;
  std::string ordersPath;
  ParquetOptions options;
  bool withOrders{false};
};

/// Writes the fixture for these options if it is missing and returns where it
/// lives. Does not read it back, so it touches no connector registry.
ParquetFixture ensureParquetFixture(
    const Options& options,
    Workload workload,
    memory::MemoryPool* pool);

struct ScannedInput {
  DataSet data;
  ParquetOptions parquetOptions;
  double coldMs{0};
  std::vector<double> warmMs;
};

/// True when --synthetic_input=parquet was given.
bool parquetInputRequested();

/// Writes the fixture if it is missing, then reads it back once per
/// --synthetic_scan_repetitions, keeping only the last read. Both backends call
/// this before registering their GPU driver adapters.
ScannedInput prepareParquetInput(
    const Options& options,
    Workload workload,
    memory::MemoryPool* pool);

/// Copies input mode, fixture shape, batch shape, and scan timings from the
/// scan phase into 'result'.
void applyScannedInput(
    const ScannedInput& input,
    const std::vector<RowVectorPtr>& batches,
    Result& result);

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

/// Fills the batch-shape fields of 'result' from the batches that were fed to
/// the GPU stage.
void recordBatchShape(const std::vector<RowVectorPtr>& batches, Result& result);

uint64_t resultChecksum(const RowVectorPtr& result);

double percentile(std::vector<double> samples, double pct);

std::string resultJson(const Result& result);

std::string csvHeader();

std::string resultCsv(const Result& result);

} // namespace facebook::velox::gpu::benchmark
