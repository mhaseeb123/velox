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

#ifdef VELOX_ENABLE_PARQUET
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <filesystem>
#endif

#include <folly/json.h>
#include <gtest/gtest.h>

namespace facebook::velox::gpu::benchmark {
namespace {

class SyntheticBenchmarkTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
#ifdef VELOX_ENABLE_PARQUET
    functions::prestosql::registerAllScalarFunctions();
    parse::registerTypeResolver();
#endif
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

TEST_F(SyntheticBenchmarkTest, scanPhaseReporting) {
  Result result{
      .backend = "test",
      .workload = "filter",
      .inputMode = "parquet",
      .rows = 100,
      .inputBytes = 800,
      .warmups = 1,
      .repetitions = 3,
      .checksum = 7,
      .expectedChecksum = 7,
      .rowGroupRows = 50,
      .scanBatchRows = 50,
      .scanColdMs = 40,
      .scanWarmMs = {10, 20, 30},
      .coldMs = 8,
      .warmMs = {1, 2, 3}};
  auto json = folly::parseJson(resultJson(result));
  EXPECT_EQ(json["input_mode"], "parquet");
  EXPECT_EQ(json["scan_cold_ms"], 40);
  EXPECT_EQ(json["scan_warm_median_ms"], 20);
  EXPECT_EQ(json["total_cold_ms"], 48);
  EXPECT_EQ(json["total_warm_median_ms"], 22);
  EXPECT_EQ(json["row_group_rows"], 50);

  // Without a scan phase the totals fall back to the compute numbers.
  Result generated{
      .backend = "test",
      .workload = "filter",
      .rows = 100,
      .repetitions = 1,
      .coldMs = 8,
      .warmMs = {2}};
  auto generatedJson = folly::parseJson(resultJson(generated));
  EXPECT_EQ(generatedJson["input_mode"], "generated");
  EXPECT_EQ(generatedJson["total_warm_median_ms"], 2);

  EXPECT_EQ(
      csvHeader().substr(0, csvHeader().find(",rows")),
      "backend,workload,input_mode,scan_mode");
  // A CPU pre-pass is the default, so a row never claims a GPU decode by
  // omission.
  EXPECT_EQ(generatedJson["scan_mode"], "cpu-prepass");
}

TEST_F(SyntheticBenchmarkTest, batchShape) {
  Options options{
      .rows = 17, .batchRows = 5, .filterPercent = 10, .groupCardinality = 3};
  auto data = makeData(pool_.get(), options);
  Result result;
  recordBatchShape(data.fact, result);
  EXPECT_EQ(result.numBatches, 4);
  EXPECT_EQ(result.maxBatchRows, 5);
  EXPECT_EQ(result.minBatchRows, 2);

  EXPECT_TRUE(workloadNeedsOrders(Workload::kJoinAggregate));
  EXPECT_TRUE(workloadNeedsOrders(Workload::kQ3));
  EXPECT_FALSE(workloadNeedsOrders(Workload::kFilter));

  // The join workloads keep the caller's batch size now that Wave handles
  // multi-batch build sides.
  auto joinData =
      makeDataForWorkload(pool_.get(), options, Workload::kJoinAggregate);
  Result joinShape;
  recordBatchShape(joinData.fact, joinShape);
  EXPECT_EQ(joinShape.numBatches, 4);
}

TEST_F(SyntheticBenchmarkTest, generatorsAreBatchIndependent) {
  Options options{
      .rows = 17, .batchRows = 5, .filterPercent = 10, .groupCardinality = 3};
  auto reference = makeData(pool_.get(), options);
  auto whole = makeFactBatch(pool_.get(), options, 0, 17);

  ASSERT_TRUE(whole->type()->equivalent(*factRowType()));
  vector_size_t at = 0;
  for (const auto& batch : reference.fact) {
    for (vector_size_t row = 0; row < batch->size(); ++row) {
      ASSERT_TRUE(whole->equalValueAt(batch.get(), at + row, row))
          << "row " << at + row;
    }
    at += batch->size();
  }
  EXPECT_EQ(ordersRowCount(options), 5);
  EXPECT_EQ(makeOrdersBatch(pool_.get(), 0, 5)->size(), 5);
}

#ifdef VELOX_ENABLE_PARQUET

TEST_F(SyntheticBenchmarkTest, parquetFixtureRoundTrip) {
  namespace fs = std::filesystem;
  const auto root =
      (fs::temp_directory_path() / "velox_synthetic_parquet_test").string();
  fs::remove_all(root);

  Options options{
      .rows = 1'000,
      .batchRows = 1'000,
      .filterPercent = 10,
      .groupCardinality = 7};
  // Row groups smaller than the requested batch, so the reader has to stop at
  // row-group boundaries.
  ParquetOptions parquetOptions{
      .rowGroupRows = 250,
      .pageBytes = 1 << 16,
      .dictionary = false,
      .compression = common::CompressionKind_NONE,
      .scanBatchRows = 250};

  const auto dir = parquetFixtureDir(root, options, parquetOptions);
  EXPECT_FALSE(parquetFixtureExists(dir, false));
  writeParquetFixture(dir, options, parquetOptions, true, pool_.get());
  EXPECT_TRUE(parquetFixtureExists(dir, true));

  double scanMs = 0;
  auto data = readParquetDataSet(
      dir, options, parquetOptions, true, pool_.get(), scanMs);
  EXPECT_GT(scanMs, 0);
  // A batch never spans row groups, so the row group is the real cap.
  Result shape;
  recordBatchShape(data.fact, shape);
  EXPECT_EQ(shape.numBatches, 4);
  EXPECT_EQ(shape.maxBatchRows, 250);

  auto reference = makeData(pool_.get(), options);
  vector_size_t at = 0;
  for (const auto& batch : data.fact) {
    for (vector_size_t row = 0; row < batch->size(); ++row) {
      ASSERT_TRUE(reference.fact.front()->equalValueAt(
          batch.get(), at + row, row))
          << "row " << at + row;
    }
    at += batch->size();
  }
  EXPECT_EQ(at, options.rows);

  // Asking for batches larger than a row group stitches the reader's batches
  // together, so the requested size is what the backend actually sees.
  ParquetOptions stitched = parquetOptions;
  stitched.scanBatchRows = 1'000;
  auto stitchedData = readParquetDataSet(
      dir, options, stitched, false, pool_.get(), scanMs);
  Result stitchedShape;
  recordBatchShape(stitchedData.fact, stitchedShape);
  EXPECT_EQ(stitchedShape.numBatches, 1);
  EXPECT_EQ(stitchedShape.maxBatchRows, 1'000);

  // Larger row groups let the reader hand back the whole file in one batch.
  ParquetOptions wide = parquetOptions;
  wide.rowGroupRows = 1'000;
  wide.scanBatchRows = 1'000;
  const auto wideDir = parquetFixtureDir(root, options, wide);
  EXPECT_NE(wideDir, dir);
  writeParquetFixture(wideDir, options, wide, false, pool_.get());
  auto wideData = readParquetDataSet(
      wideDir, options, wide, false, pool_.get(), scanMs);
  Result wideShape;
  recordBatchShape(wideData.fact, wideShape);
  EXPECT_EQ(wideShape.numBatches, 1);
  EXPECT_EQ(wideShape.maxBatchRows, 1'000);

  // Small reader batches subdivide a row group.
  ParquetOptions narrow = wide;
  narrow.scanBatchRows = 100;
  auto narrowData = readParquetDataSet(
      wideDir, options, narrow, false, pool_.get(), scanMs);
  Result narrowShape;
  recordBatchShape(narrowData.fact, narrowShape);
  EXPECT_EQ(narrowShape.numBatches, 10);
  EXPECT_EQ(narrowShape.maxBatchRows, 100);

  fs::remove_all(root);
}

#endif // VELOX_ENABLE_PARQUET

} // namespace
} // namespace facebook::velox::gpu::benchmark
