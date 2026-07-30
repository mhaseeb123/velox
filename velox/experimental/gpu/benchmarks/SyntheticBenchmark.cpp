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

#include "velox/common/base/Exceptions.h"
#include "velox/vector/tests/utils/VectorMaker.h"

#ifdef VELOX_ENABLE_PARQUET
#include "velox/common/config/Config.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <folly/executors/IOThreadPoolExecutor.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#endif

#include <folly/json.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <cmath>
#include <limits>

DEFINE_string(
    synthetic_input,
    "generated",
    "Input source: generated (in-memory vectors) or parquet (CPU Parquet scan)");
DEFINE_string(
    synthetic_parquet_dir,
    "/tmp/velox_synthetic_parquet",
    "Root directory holding cached Parquet fixtures");
DEFINE_int64(
    synthetic_parquet_row_group_rows,
    100'000,
    "Rows per Parquet row group in the fixture");
DEFINE_int64(
    synthetic_parquet_page_bytes,
    1 << 20,
    "Parquet data page size in bytes");
DEFINE_bool(
    synthetic_parquet_dictionary,
    false,
    "Enable Parquet dictionary encoding in the fixture");
DEFINE_string(
    synthetic_parquet_compression,
    "none",
    "Parquet compression: none, snappy, zstd, gzip, lz4, or zlib");
DEFINE_int64(
    synthetic_scan_batch_rows,
    100'000,
    "Rows per batch the Parquet scan is asked to produce");
DEFINE_int32(
    synthetic_scan_repetitions,
    2,
    "Number of Parquet reads. The first is reported cold, the rest warm");
DEFINE_bool(
    synthetic_write_fixture,
    false,
    "Rewrite the Parquet fixture even when a cached one exists");

namespace facebook::velox::gpu::benchmark {
namespace {

common::CompressionKind parseCompression(const std::string& name) {
  if (name == "none" || name == "uncompressed") {
    return common::CompressionKind_NONE;
  }
  return common::stringToCompressionKind(name);
}

constexpr int64_t kDateCardinality = 2'500;
constexpr int64_t kValueCardinality = 1'000;

uint64_t mix(uint64_t hash, uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return hash ^ (value + (hash << 6) + (hash >> 2));
}

const std::vector<std::string>& factNames() {
  static const std::vector<std::string> names{
      "key",
      "group_key",
      "value",
      "quantity",
      "discount",
      "shipdate",
      "extendedprice",
      "returnflag",
      "linestatus",
      "orderkey"};
  return names;
}

const std::vector<std::string>& ordersNames() {
  static const std::vector<std::string> names{
      "o_orderkey", "orderdate", "shippriority", "build_value"};
  return names;
}

RowVectorPtr factBatch(
    test::VectorMaker& maker,
    const Options& options,
    int64_t begin,
    vector_size_t size) {
  auto make = [&](auto fn) {
    return maker.flatVector<int64_t>(
        size, [begin, fn](vector_size_t row) { return fn(begin + row); });
  };
  return maker.rowVector(
      factNames(),
      {make([](auto row) { return row; }),
       make([&](auto row) { return row % options.groupCardinality; }),
       make([](auto row) { return (row * 17 + 13) % kValueCardinality; }),
       make([](auto row) { return 1 + (row * 7) % 50; }),
       make([](auto row) { return (row * 3) % 11; }),
       make([](auto row) { return row % kDateCardinality; }),
       make([](auto row) { return 100 + (row * 19) % 100'000; }),
       make([](auto row) { return row % 3; }),
       make([](auto row) { return row % 2; }),
       make([](auto row) { return row / 4; })});
}

RowVectorPtr ordersBatch(
    test::VectorMaker& maker,
    int64_t begin,
    vector_size_t size) {
  auto make = [&](auto fn) {
    return maker.flatVector<int64_t>(
        size, [begin, fn](vector_size_t row) { return fn(begin + row); });
  };
  return maker.rowVector(
      ordersNames(),
      {make([](auto row) { return row; }),
       make([](auto row) { return row % kDateCardinality; }),
       make([](auto row) { return row % 5; }),
       make([](auto row) { return 1 + (row * 23) % 10'000; })});
}

std::vector<RowVectorPtr> makeFact(
    test::VectorMaker& maker,
    const Options& options) {
  std::vector<RowVectorPtr> batches;
  for (int64_t begin = 0; begin < options.rows; begin += options.batchRows) {
    const auto size = static_cast<vector_size_t>(
        std::min<int64_t>(options.batchRows, options.rows - begin));
    batches.push_back(factBatch(maker, options, begin, size));
  }
  return batches;
}

std::vector<RowVectorPtr> makeOrders(
    test::VectorMaker& maker,
    const Options& options) {
  return {
      ordersBatch(
          maker, 0, static_cast<vector_size_t>(ordersRowCount(options)))};
}

} // namespace

RowTypePtr factRowType() {
  return ROW(
      std::vector<std::string>(factNames()),
      std::vector<TypePtr>(factNames().size(), BIGINT()));
}

RowTypePtr ordersRowType() {
  return ROW(
      std::vector<std::string>(ordersNames()),
      std::vector<TypePtr>(ordersNames().size(), BIGINT()));
}

int64_t ordersRowCount(const Options& options) {
  return std::max<int64_t>(1, (options.rows + 3) / 4);
}

RowVectorPtr makeFactBatch(
    memory::MemoryPool* pool,
    const Options& options,
    int64_t begin,
    vector_size_t size) {
  test::VectorMaker maker(pool);
  return factBatch(maker, options, begin, size);
}

RowVectorPtr makeOrdersBatch(
    memory::MemoryPool* pool,
    int64_t begin,
    vector_size_t size) {
  test::VectorMaker maker(pool);
  return ordersBatch(maker, begin, size);
}

std::vector<Workload> allWorkloads() {
  return {
      Workload::kScan,
      Workload::kFilter,
      Workload::kProject,
      Workload::kAggregate,
      Workload::kGroupBy,
      Workload::kJoinAggregate,
      Workload::kQ1,
      Workload::kQ6,
      Workload::kQ3};
}

std::string workloadName(Workload workload) {
  switch (workload) {
    case Workload::kScan:
      return "scan";
    case Workload::kFilter:
      return "filter";
    case Workload::kProject:
      return "project";
    case Workload::kAggregate:
      return "aggregate";
    case Workload::kGroupBy:
      return "group_by";
    case Workload::kJoinAggregate:
      return "join_aggregate";
    case Workload::kQ1:
      return "q1";
    case Workload::kQ6:
      return "q6";
    case Workload::kQ3:
      return "q3";
  }
  VELOX_UNREACHABLE();
}

bool workloadNeedsOrders(Workload workload) {
  return workload == Workload::kJoinAggregate || workload == Workload::kQ3;
}

Workload parseWorkload(const std::string& name) {
  for (auto workload : allWorkloads()) {
    if (workloadName(workload) == name) {
      return workload;
    }
  }
  VELOX_USER_FAIL("Unknown synthetic benchmark workload: {}", name);
}

DataSet makeData(memory::MemoryPool* pool, const Options& options) {
  VELOX_USER_CHECK_GT(options.rows, 0);
  VELOX_USER_CHECK_GT(options.batchRows, 0);
  VELOX_USER_CHECK_GE(options.filterPercent, 0);
  VELOX_USER_CHECK_LE(options.filterPercent, 100);
  VELOX_USER_CHECK_GT(options.groupCardinality, 0);

  test::VectorMaker maker(pool);
  DataSet data;
  data.fact = makeFact(maker, options);
  data.factType = asRowType(data.fact.front()->type());
  data.orders = makeOrders(maker, options);
  data.ordersType = asRowType(data.orders.front()->type());
  data.inputBytes = options.rows * data.factType->size() * sizeof(int64_t) +
      ((options.rows + 3) / 4) * data.ordersType->size() * sizeof(int64_t);
  return data;
}

DataSet makeDataForWorkload(
    memory::MemoryPool* pool,
    const Options& options,
    Workload /*workload*/) {
  return makeData(pool, options);
}

core::PlanNodePtr finishPlan(
    Workload workload,
    exec::test::PlanBuilder& source,
    const std::shared_ptr<core::PlanNodeIdGenerator>& planNodeIdGenerator,
    const DataSet& data,
    const Options& options,
    bool sourceFiltersPushedDown) {
  switch (workload) {
    case Workload::kScan:
      source.singleAggregation(
          {}, {"sum(key)", "sum(value)", "sum(quantity)", "sum(discount)"});
      break;
    case Workload::kFilter:
      if (!sourceFiltersPushedDown) {
        source.filter(
            fmt::format(
                "value < {}", options.filterPercent * kValueCardinality / 100));
      }
      source.project({"key + 0 as key", "value + 0 as value"})
          .singleAggregation({}, {"sum(key)", "sum(value)"});
      break;
    case Workload::kProject:
      source
          .project(
              {"key",
               "value + quantity as add_value",
               "extendedprice * (100 - discount) as discounted_price"})
          .singleAggregation(
              {}, {"sum(key)", "sum(add_value)", "sum(discounted_price)"});
      break;
    case Workload::kAggregate:
      source.singleAggregation({}, {"sum(value)", "sum(quantity)"});
      break;
    case Workload::kGroupBy:
      source.singleAggregation({"group_key"}, {"sum(value)", "sum(quantity)"});
      break;
    case Workload::kJoinAggregate: {
      auto build =
          exec::test::PlanBuilder(planNodeIdGenerator).values(data.orders);
      source.hashJoin(
          {"orderkey"},
          {"o_orderkey"},
          build.planNode(),
          "",
          {"key",
           "group_key",
           "value",
           "quantity",
           "discount",
           "shipdate",
           "extendedprice",
           "returnflag",
           "linestatus",
           "orderkey",
           "o_orderkey",
           "orderdate",
           "shippriority",
           "build_value"},
          core::JoinType::kInner);
      break;
    }
    case Workload::kQ1:
      if (!sourceFiltersPushedDown) {
        source.filter("shipdate <= 1500");
      }
      source
          .project(
              {"returnflag",
               "linestatus",
               "quantity",
               "extendedprice",
               "extendedprice * (100 - discount) as discounted_price"})
          .singleAggregation(
              {"returnflag", "linestatus"},
              {"sum(quantity)", "sum(extendedprice)", "sum(discounted_price)"});
      break;
    case Workload::kQ6:
      if (!sourceFiltersPushedDown) {
        source.filter("shipdate >= 500")
            .filter("shipdate < 1500")
            .filter("discount >= 4")
            .filter("discount <= 6")
            .filter("quantity < 24");
      }
      source.project({"extendedprice * discount as revenue"})
          .singleAggregation({}, {"sum(revenue)"});
      break;
    case Workload::kQ3: {
      auto build =
          exec::test::PlanBuilder(planNodeIdGenerator).values(data.orders);
      source.hashJoin(
          {"orderkey"},
          {"o_orderkey"},
          build.planNode(),
          "",
          {"key",
           "group_key",
           "value",
           "quantity",
           "discount",
           "shipdate",
           "extendedprice",
           "returnflag",
           "linestatus",
           "orderkey",
           "o_orderkey",
           "orderdate",
           "shippriority",
           "build_value"},
          core::JoinType::kInner);
      break;
    }
  }
  return source.planNode();
}

core::PlanNodePtr makeValuesPlan(
    Workload workload,
    memory::MemoryPool* pool,
    const DataSet& data,
    const Options& options) {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto source = exec::test::PlanBuilder(ids, pool).values(data.fact);
  return finishPlan(workload, source, ids, data, options);
}

core::PlanNodePtr makePostJoinPlan(
    Workload workload,
    memory::MemoryPool* pool,
    const RowVectorPtr& joined) {
  auto source = exec::test::PlanBuilder(pool).values({joined});
  return finishPostJoinPlan(workload, source);
}

core::PlanNodePtr finishPostJoinPlan(
    Workload workload,
    exec::test::PlanBuilder& source,
    bool sourceFiltersPushedDown) {
  if (workload == Workload::kJoinAggregate) {
    return source.singleAggregation({}, {"sum(value)", "sum(build_value)"})
        .planNode();
  }
  VELOX_CHECK(workload == Workload::kQ3);
  if (!sourceFiltersPushedDown) {
    source.filter("shipdate > 1000").filter("orderdate < 1000");
  }
  return source
      .project(
          {"orderdate",
           "shippriority",
           "extendedprice * (100 - discount) as revenue"})
      .singleAggregation({"orderdate", "shippriority"}, {"sum(revenue)"})
      .planNode();
}

void recordBatchShape(
    const std::vector<RowVectorPtr>& batches,
    Result& result) {
  result.numBatches = static_cast<int64_t>(batches.size());
  result.maxBatchRows = 0;
  result.minBatchRows = 0;
  if (batches.empty()) {
    return;
  }
  result.minBatchRows = std::numeric_limits<int64_t>::max();
  for (const auto& batch : batches) {
    result.maxBatchRows = std::max<int64_t>(result.maxBatchRows, batch->size());
    result.minBatchRows = std::min<int64_t>(result.minBatchRows, batch->size());
  }
}

#ifdef VELOX_ENABLE_PARQUET

namespace {

constexpr const char* kFactFile = "fact.parquet";
constexpr const char* kOrdersFile = "orders.parquet";
constexpr const char* kCompleteFile = "_COMPLETE";
// Rows handed to the writer at a time. Keeps fixture generation bounded in
// host memory without making the write call overhead matter.
constexpr int64_t kWriteBatchRows = 100'000;
// LocalWriteFile writes through fwrite, which tops out just below 2 GB per
// call, so the sink must never buffer more than this.
constexpr int64_t kMaxSinkWriteBytes = 1LL << 30;

std::string compressionName(common::CompressionKind kind) {
  return common::compressionKindToString(kind);
}

folly::IOThreadPoolExecutor* scanIoExecutor() {
  static auto executor = std::make_unique<folly::IOThreadPoolExecutor>(8);
  return executor.get();
}

/// Registers everything the CPU Parquet scan needs, once per process. Uses the
/// standard test Hive connector id so PlanBuilder::tableScan finds it, and
/// leaves an already-registered connector (the Wave binary registers one)
/// alone.
void ensureParquetScanRegistered() {
  static const bool registered = [] {
    filesystems::registerLocalFileSystem();
    parquet::registerParquetReaderFactory();
    if (connector::ConnectorRegistry::tryGet(exec::test::kHiveConnectorId) ==
        nullptr) {
      connector::hive::HiveConnectorFactory factory;
      auto connector = factory.newConnector(
          exec::test::kHiveConnectorId,
          std::make_shared<config::ConfigBase>(
              std::unordered_map<std::string, std::string>()),
          scanIoExecutor());
      connector::ConnectorRegistry::global().insert(
          connector->connectorId(), connector);
    }
    return true;
  }();
  VELOX_CHECK(registered);
}

void writeParquetFile(
    const std::string& path,
    const RowTypePtr& schema,
    const ParquetOptions& parquetOptions,
    int64_t totalRows,
    const std::function<RowVectorPtr(int64_t, vector_size_t)>& makeBatch,
    memory::MemoryPool* pool) {
  // LocalWriteFile opens an existing file for append, so rewriting a fixture
  // over the debris of an interrupted write would silently concatenate the two
  // into a file whose page headers stop making sense partway through.
  std::filesystem::remove(path);
  auto sink = std::make_unique<dwio::common::WriteFileSink>(
      std::make_unique<LocalWriteFile>(path, true, false), path);

  // The Parquet writer allocates children of its pool, so it needs an
  // aggregate pool rather than the leaf pool the vectors come from.
  auto writerPool =
      memory::memoryManager()->addRootPool("SyntheticBenchmarkParquetWriter");

  dwio::common::WriterOptions writerOptions;
  writerOptions.memoryPool = writerPool.get();
  writerOptions.compressionKind = parquetOptions.compression;
  // The byte threshold has to sit above a full row group, or it would cut row
  // groups short and silently shrink the reader's batches. It also bounds how
  // much the sink buffers before writing, and a single write must stay under
  // the 2 GB that fwrite accepts.
  const int64_t rowGroupBytes =
      parquetOptions.rowGroupRows * static_cast<int64_t>(schema->size()) * 8;
  VELOX_USER_CHECK_LT(
      rowGroupBytes,
      kMaxSinkWriteBytes,
      "A row group of {} rows is {} bytes, which the writer cannot flush in one "
      "call. Use at most {} rows per row group for this schema.",
      parquetOptions.rowGroupRows,
      rowGroupBytes,
      kMaxSinkWriteBytes / (static_cast<int64_t>(schema->size()) * 8));
  const int64_t bytesInRowGroup = std::clamp<int64_t>(
      rowGroupBytes * 2,
      parquet::DefaultFlushPolicy::kDefaultBytesInRowGroup,
      kMaxSinkWriteBytes);
  writerOptions.flushPolicyFactory = [rows = parquetOptions.rowGroupRows,
                                      bytesInRowGroup]() {
    return std::make_unique<parquet::DefaultFlushPolicy>(
        static_cast<uint64_t>(rows), bytesInRowGroup);
  };
  parquet::ParquetWriterOptions formatOptions;
  formatOptions.encoding = parquet::arrow::Encoding::type::kPlain;
  formatOptions.enableDictionary = parquetOptions.dictionary;
  formatOptions.dataPageSize = parquetOptions.pageBytes;
  formatOptions.dictionaryPageSizeLimit = parquetOptions.pageBytes;
  writerOptions.formatSpecificOptions =
      std::make_shared<parquet::ParquetWriterOptions>(formatOptions);

  parquet::Writer writer{std::move(sink), writerOptions, schema};
  for (int64_t begin = 0; begin < totalRows; begin += kWriteBatchRows) {
    const auto size = static_cast<vector_size_t>(
        std::min<int64_t>(kWriteBatchRows, totalRows - begin));
    writer.write(makeBatch(begin, size));
  }
  writer.close();
}

} // namespace

std::string parquetFixtureDir(
    const std::string& root,
    const Options& options,
    const ParquetOptions& parquetOptions) {
  // The local file system matcher only recognizes absolute paths.
  return fmt::format(
      "{}/rows{}_gc{}_rg{}_pg{}_{}_{}",
      std::filesystem::absolute(root).lexically_normal().string(),
      options.rows,
      options.groupCardinality,
      parquetOptions.rowGroupRows,
      parquetOptions.pageBytes,
      parquetOptions.dictionary ? "dict" : "plain",
      compressionName(parquetOptions.compression));
}

bool parquetFixtureExists(const std::string& dir, bool withOrders) {
  namespace fs = std::filesystem;
  const auto completePath = fmt::format("{}/{}", dir, kCompleteFile);
  if (!fs::exists(completePath)) {
    return false;
  }
  if (!fs::exists(fmt::format("{}/{}", dir, kFactFile))) {
    return false;
  }
  if (withOrders && !fs::exists(fmt::format("{}/{}", dir, kOrdersFile))) {
    return false;
  }

  // The marker alone is not evidence the files are intact: an interrupted
  // write once left a fixture that was three times its expected size and still
  // marked complete, and every reader then died on a malformed page header.
  // The recorded sizes turn that into a cache miss instead.
  std::ifstream complete(completePath);
  int64_t rows = 0;
  int64_t factBytes = -1;
  int64_t ordersBytes = -1;
  complete >> rows >> factBytes >> ordersBytes;
  if (factBytes < 0) {
    // Written before sizes were recorded. Nothing to check against.
    return true;
  }
  const auto matches = [](const std::string& path, int64_t expected) {
    std::error_code error;
    const auto actual = static_cast<int64_t>(
        std::filesystem::file_size(path, error));
    if (error) {
      return false;
    }
    if (actual == expected) {
      return true;
    }
    LOG(WARNING) << "Parquet fixture " << path << " is " << actual
                 << " bytes but was written as " << expected
                 << "; treating it as missing and rewriting";
    return false;
  };
  if (!matches(fmt::format("{}/{}", dir, kFactFile), factBytes)) {
    return false;
  }
  return !withOrders || ordersBytes < 0 ||
      matches(fmt::format("{}/{}", dir, kOrdersFile), ordersBytes);
}

void writeParquetFixture(
    const std::string& dir,
    const Options& options,
    const ParquetOptions& parquetOptions,
    bool withOrders,
    memory::MemoryPool* pool) {
  namespace fs = std::filesystem;
  ensureParquetScanRegistered();
  fs::create_directories(dir);
  fs::remove(fmt::format("{}/{}", dir, kCompleteFile));

  writeParquetFile(
      fmt::format("{}/{}", dir, kFactFile),
      factRowType(),
      parquetOptions,
      options.rows,
      [&](int64_t begin, vector_size_t size) {
        return makeFactBatch(pool, options, begin, size);
      },
      pool);
  if (withOrders) {
    writeParquetFile(
        fmt::format("{}/{}", dir, kOrdersFile),
        ordersRowType(),
        parquetOptions,
        ordersRowCount(options),
        [&](int64_t begin, vector_size_t size) {
          return makeOrdersBatch(pool, begin, size);
        },
        pool);
  }
  // Sizes go in the marker so a later run can tell an intact fixture from a
  // half-written or appended-to one. See parquetFixtureExists.
  std::ofstream complete(fmt::format("{}/{}", dir, kCompleteFile));
  complete << options.rows << '\n'
           << fs::file_size(fmt::format("{}/{}", dir, kFactFile)) << '\n';
  if (withOrders) {
    complete << fs::file_size(fmt::format("{}/{}", dir, kOrdersFile)) << '\n';
  }
}

namespace {

/// Runs a single-driver, serial CPU scan over 'path' and returns the reader's
/// own batches. 'owner' keeps the task alive, which is what makes returning the
/// batches uncopied safe.
std::vector<RowVectorPtr> scanParquetFile(
    const std::string& path,
    const RowTypePtr& rowType,
    int64_t scanBatchRows,
    memory::MemoryPool* pool,
    std::shared_ptr<void>& owner) {
  auto plan = exec::test::PlanBuilder(pool).tableScan(rowType).planNode();
  auto splits = exec::test::HiveConnectorTestBase::makeHiveConnectorSplits(
      path, 1, dwio::common::FileFormat::PARQUET);

  exec::CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.copyResult = false;
  params.queryConfigs = {
      {core::QueryConfig::kMaxOutputBatchRows, std::to_string(scanBatchRows)},
      {core::QueryConfig::kPreferredOutputBatchRows,
       std::to_string(scanBatchRows)},
      // Large enough that the byte budget never becomes the binding limit for
      // the row counts we ask for.
      {core::QueryConfig::kPreferredOutputBatchBytes,
       std::to_string(scanBatchRows * 8 * 64)}};

  std::shared_ptr<exec::TaskCursor> cursor = exec::TaskCursor::create(params);
  cursor->task()->addSplit(plan->id(), exec::Split(splits.front()));
  cursor->task()->noMoreSplits(plan->id());

  std::vector<RowVectorPtr> batches;
  while (cursor->moveNext()) {
    auto batch = cursor->current();
    // The scan hands out lazy children that are only valid until it advances,
    // so materialize them here, inside the measured scan.
    batch->loadedVector();
    batches.push_back(std::move(batch));
  }
  owner = cursor;
  return batches;
}

RowVectorPtr concatenate(
    const std::vector<RowVectorPtr>& batches,
    memory::MemoryPool* pool) {
  int64_t rows = 0;
  for (const auto& batch : batches) {
    rows += batch->size();
  }
  VELOX_CHECK_LE(rows, std::numeric_limits<vector_size_t>::max());
  auto result = BaseVector::create<RowVector>(
      batches.front()->type(), static_cast<vector_size_t>(rows), pool);
  vector_size_t at = 0;
  for (const auto& batch : batches) {
    result->copy(batch.get(), at, 0, batch->size());
    at += batch->size();
  }
  return result;
}

/// Merges consecutive batches until each one holds at least 'targetRows'. A
/// reader batch never crosses a row-group boundary, so this is the only way to
/// hand a backend batches larger than a row group.
std::vector<RowVectorPtr> coalesceBatches(
    const std::vector<RowVectorPtr>& batches,
    int64_t targetRows,
    memory::MemoryPool* pool) {
  std::vector<RowVectorPtr> result;
  std::vector<RowVectorPtr> group;
  int64_t rows = 0;
  for (const auto& batch : batches) {
    group.push_back(batch);
    rows += batch->size();
    if (rows >= targetRows) {
      result.push_back(
          group.size() == 1 ? group.front() : concatenate(group, pool));
      group.clear();
      rows = 0;
    }
  }
  if (!group.empty()) {
    result.push_back(
        group.size() == 1 ? group.front() : concatenate(group, pool));
  }
  return result;
}

} // namespace

DataSet readParquetDataSet(
    const std::string& dir,
    const Options& options,
    const ParquetOptions& parquetOptions,
    bool withOrders,
    memory::MemoryPool* pool,
    double& scanMs) {
  ensureParquetScanRegistered();
  VELOX_USER_CHECK(
      parquetFixtureExists(dir, withOrders),
      "Parquet fixture is missing or incomplete: {}",
      dir);

  DataSet data;
  data.factType = factRowType();
  data.ordersType = ordersRowType();
  data.inputBytes = options.rows * data.factType->size() * sizeof(int64_t) +
      ordersRowCount(options) * data.ordersType->size() * sizeof(int64_t);

  const auto start = std::chrono::steady_clock::now();
  std::shared_ptr<void> factOwner;
  data.fact = scanParquetFile(
      fmt::format("{}/{}", dir, kFactFile),
      data.factType,
      parquetOptions.scanBatchRows,
      pool,
      factOwner);
  data.owners.push_back(std::move(factOwner));
  if (withOrders) {
    std::shared_ptr<void> ordersOwner;
    data.orders = scanParquetFile(
        fmt::format("{}/{}", dir, kOrdersFile),
        data.ordersType,
        std::max<int64_t>(parquetOptions.scanBatchRows, 1),
        pool,
        ordersOwner);
    data.owners.push_back(std::move(ordersOwner));
  }
  // A row group caps how large a reader batch can get, and the writer caps how
  // large a row group can get, so asking for batches above the row-group size
  // means stitching batches together. The copy is charged to the scan phase.
  data.fact = coalesceBatches(data.fact, parquetOptions.scanBatchRows, pool);
  if (withOrders) {
    data.orders =
        coalesceBatches(data.orders, parquetOptions.scanBatchRows, pool);
  }
  scanMs = std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
               .count();

  int64_t scanned = 0;
  for (const auto& batch : data.fact) {
    scanned += batch->size();
  }
  VELOX_CHECK_EQ(
      scanned, options.rows, "Parquet fixture holds the wrong row count");
  return data;
}

#endif // VELOX_ENABLE_PARQUET

bool parquetInputRequested() {
  if (FLAGS_synthetic_input == "generated") {
    return false;
  }
  VELOX_USER_CHECK_EQ(
      FLAGS_synthetic_input,
      "parquet",
      "--synthetic_input must be generated or parquet");
  return true;
}

ParquetFixture ensureParquetFixture(
    const Options& options,
    Workload workload,
    memory::MemoryPool* pool) {
#ifndef VELOX_ENABLE_PARQUET
  VELOX_USER_FAIL("This build has no Parquet support");
#else
  ParquetFixture fixture;
  fixture.withOrders = workloadNeedsOrders(workload);
  fixture.options = ParquetOptions{
      .rowGroupRows = FLAGS_synthetic_parquet_row_group_rows,
      .pageBytes = FLAGS_synthetic_parquet_page_bytes,
      .dictionary = FLAGS_synthetic_parquet_dictionary,
      .compression = parseCompression(FLAGS_synthetic_parquet_compression),
      .scanBatchRows = FLAGS_synthetic_scan_batch_rows};
  fixture.dir =
      parquetFixtureDir(FLAGS_synthetic_parquet_dir, options, fixture.options);
  if (FLAGS_synthetic_write_fixture ||
      !parquetFixtureExists(fixture.dir, fixture.withOrders)) {
    writeParquetFixture(
        fixture.dir, options, fixture.options, fixture.withOrders, pool);
  }
  fixture.factPath = fmt::format("{}/{}", fixture.dir, kFactFile);
  fixture.ordersPath = fixture.withOrders
      ? fmt::format("{}/{}", fixture.dir, kOrdersFile)
      : std::string{};
  return fixture;
#endif
}

ScannedInput prepareParquetInput(
    const Options& options,
    Workload workload,
    memory::MemoryPool* pool) {
#ifndef VELOX_ENABLE_PARQUET
  VELOX_USER_FAIL("This build has no Parquet support");
#else
  VELOX_USER_CHECK_GT(FLAGS_synthetic_scan_repetitions, 0);
  const auto fixture = ensureParquetFixture(options, workload, pool);
  const auto& dir = fixture.dir;
  const bool withOrders = fixture.withOrders;

  ScannedInput input;
  input.parquetOptions = fixture.options;

  for (int32_t i = 0; i < FLAGS_synthetic_scan_repetitions; ++i) {
    // Drop the previous read before the next one so host memory holds a single
    // copy of the input at a time.
    input.data.reset();
    double scanMs = 0;
    input.data = readParquetDataSet(
        dir, options, input.parquetOptions, withOrders, pool, scanMs);
    if (i == 0) {
      input.coldMs = scanMs;
    } else {
      input.warmMs.push_back(scanMs);
    }
  }
  return input;
#endif
}

void applyScannedInput(
    const ScannedInput& input,
    const std::vector<RowVectorPtr>& batches,
    Result& result) {
  result.inputMode = "parquet";
  result.rowGroupRows = input.parquetOptions.rowGroupRows;
  result.scanBatchRows = input.parquetOptions.scanBatchRows;
  result.scanColdMs = input.coldMs;
  result.scanWarmMs = input.warmMs;
  recordBatchShape(batches, result);
}

uint64_t resultChecksum(const RowVectorPtr& result) {
  uint64_t checksum = mix(0, result->size());
  for (vector_size_t row = 0; row < result->size(); ++row) {
    uint64_t rowHash = 0xcbf29ce484222325ULL;
    for (column_index_t column = 0; column < result->childrenSize(); ++column) {
      auto* vector = result->childAt(column)->as<SimpleVector<int64_t>>();
      VELOX_CHECK_NOT_NULL(
          vector, "Synthetic benchmark output must contain only BIGINT");
      rowHash = mix(
          rowHash,
          vector->isNullAt(row) ? 0xd6e8feb86659fd93ULL
                                : static_cast<uint64_t>(vector->valueAt(row)));
    }
    checksum += rowHash;
  }
  return checksum;
}

double percentile(std::vector<double> samples, double pct) {
  VELOX_CHECK(!samples.empty());
  std::sort(samples.begin(), samples.end());
  const auto index = static_cast<size_t>(
      std::ceil((samples.size() - 1) * std::clamp(pct, 0.0, 1.0)));
  return samples[index];
}

namespace {

/// Median of the warm scan samples, or the cold scan when there was only one
/// read. Zero in generated mode, where there is no scan phase.
double scanWarmMedian(const Result& result) {
  if (!result.scanWarmMs.empty()) {
    return percentile(result.scanWarmMs, 0.5);
  }
  return result.scanColdMs;
}

} // namespace

std::string resultJson(const Result& result) {
  folly::dynamic samples = folly::dynamic::array;
  for (auto sample : result.warmMs) {
    samples.push_back(sample);
  }
  folly::dynamic scanSamples = folly::dynamic::array;
  for (auto sample : result.scanWarmMs) {
    scanSamples.push_back(sample);
  }
  const auto median = percentile(result.warmMs, 0.5);
  const auto p95 = percentile(result.warmMs, 0.95);
  const auto scanMedian = scanWarmMedian(result);
  folly::dynamic json = folly::dynamic::object("backend", result.backend)(
      "workload", result.workload)("input_mode", result.inputMode)(
      "scan_mode", result.scanMode)(
      "rows", result.rows)("input_bytes", result.inputBytes)(
      "filter_percent", result.filterPercent)(
      "group_cardinality", result.groupCardinality)("warmups", result.warmups)(
      "repetitions", result.repetitions)(
      "checksum", fmt::format("{:016x}", result.checksum))(
      "expected_checksum", fmt::format("{:016x}", result.expectedChecksum))(
      "correct", result.checksum == result.expectedChecksum)(
      "num_batches", result.numBatches)("max_batch_rows", result.maxBatchRows)(
      "min_batch_rows", result.minBatchRows)(
      "row_group_rows", result.rowGroupRows)(
      "scan_batch_rows", result.scanBatchRows)(
      "scan_cold_ms", result.scanColdMs)("scan_warm_median_ms", scanMedian)(
      "cold_ms", result.coldMs)("warm_median_ms", median)("warm_p95_ms", p95)(
      "total_cold_ms", result.scanColdMs + result.coldMs)(
      "total_warm_median_ms", scanMedian + median)(
      "rows_per_second", result.rows * 1'000.0 / median)(
      "input_gb_per_second", result.inputBytes / median / 1'000'000.0)(
      "warm_samples_ms", std::move(samples))(
      "scan_warm_samples_ms", std::move(scanSamples));
  return folly::toJson(json);
}

std::string csvHeader() {
  return "backend,workload,input_mode,scan_mode,rows,input_bytes,filter_percent,"
         "group_cardinality,warmups,repetitions,checksum,expected_checksum,"
         "correct,num_batches,max_batch_rows,min_batch_rows,row_group_rows,"
         "scan_batch_rows,scan_cold_ms,scan_warm_median_ms,cold_ms,"
         "warm_median_ms,warm_p95_ms,total_cold_ms,total_warm_median_ms,"
         "rows_per_second,input_gb_per_second";
}

std::string resultCsv(const Result& result) {
  const auto median = percentile(result.warmMs, 0.5);
  const auto p95 = percentile(result.warmMs, 0.95);
  const auto scanMedian = scanWarmMedian(result);
  return fmt::format(
      "{},{},{},{},{},{},{},{},{},{},{:016x},{:016x},{},{},{},{},{},{},"
      "{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.3f},{:.6f}",
      result.backend,
      result.workload,
      result.inputMode,
      result.scanMode,
      result.rows,
      result.inputBytes,
      result.filterPercent,
      result.groupCardinality,
      result.warmups,
      result.repetitions,
      result.checksum,
      result.expectedChecksum,
      result.checksum == result.expectedChecksum,
      result.numBatches,
      result.maxBatchRows,
      result.minBatchRows,
      result.rowGroupRows,
      result.scanBatchRows,
      result.scanColdMs,
      scanMedian,
      result.coldMs,
      median,
      p95,
      result.scanColdMs + result.coldMs,
      scanMedian + median,
      result.rows * 1'000.0 / median,
      result.inputBytes / median / 1'000'000.0);
}

} // namespace facebook::velox::gpu::benchmark
