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
#include <cuda_runtime.h> // @manual

#include <filesystem>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/experimental/wave/dwio/parquet/WaveParquetSplitReader.h"
#include "velox/experimental/wave/exec/ToWave.h"
#include "velox/experimental/wave/exec/WaveHiveDataSource.h"

DECLARE_int32(wave_max_reader_batch_rows);
DECLARE_int32(wave_reader_rows_per_tb);
DECLARE_int32(max_streams_per_driver);

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

struct WaveParquetScanTestParam {
  // Wave reader batch size. Small values make many batches per row group,
  // large values make a batch span row groups.
  int32_t batchRows{20'000};
  int32_t numStreams{1};
  int32_t numDrivers{1};
};

std::vector<WaveParquetScanTestParam> params() {
  return {
      WaveParquetScanTestParam{},
      WaveParquetScanTestParam{.batchRows = 1'024, .numStreams = 4},
      WaveParquetScanTestParam{
          .batchRows = 1'111, .numStreams = 2, .numDrivers = 2}};
}

class WaveParquetTableScanTest
    : public virtual HiveConnectorTestBase,
      public testing::WithParamInterface<WaveParquetScanTestParam> {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    if (int device; cudaGetDevice(&device) != cudaSuccess) {
      GTEST_SKIP() << "No CUDA detected, skipping all tests";
    }
    parquet::registerParquetReaderFactory();
    wave::registerWave();
    // Wave over a normally registered Hive connector, no "wavemock" connector
    // and no mock file format.
    wave::WaveHiveDataSource::registerWaveDelegate();
    wave::WaveParquetSplitReader::registerSplitReader();

    const auto param = GetParam();
    FLAGS_wave_max_reader_batch_rows = param.batchRows;
    FLAGS_max_streams_per_driver = param.numStreams;
    numDrivers_ = param.numDrivers;
    tempDirectory_ = TempDirectoryPath::create();
  }

  void TearDown() override {
    parquet::unregisterParquetReaderFactory();
    HiveConnectorTestBase::TearDown();
  }

  // Writes 'vectors' as one Parquet file with row groups of 'rowsInRowGroup'
  // rows and returns a split for it.
  std::shared_ptr<connector::hive::HiveConnectorSplit> writeFile(
      const std::vector<RowVectorPtr>& vectors,
      int64_t rowsInRowGroup,
      const std::string& name,
      bool enableDictionary = false) {
    const auto path = fmt::format("{}/{}", tempDirectory_->getPath(), name);
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::make_unique<LocalWriteFile>(
            path, /*shouldThrowOnFileAlreadyExists=*/
            true,
            /*bufferIo=*/false),
        path);
    dwio::common::WriterOptions options;
    auto writerPool = rootPool_->addAggregateChild("waveParquetWriter");
    options.memoryPool = writerPool.get();
    options.compressionKind = common::CompressionKind_NONE;
    options.flushPolicyFactory = [rowsInRowGroup]() {
      return std::make_unique<parquet::DefaultFlushPolicy>(
          rowsInRowGroup, std::numeric_limits<int64_t>::max());
    };
    parquet::ParquetWriterOptions parquetOptions;
    parquetOptions.enableDictionary = enableDictionary;
    options.formatSpecificOptions =
        std::make_shared<parquet::ParquetWriterOptions>(parquetOptions);
    auto writer = std::make_unique<parquet::Writer>(
        std::move(sink), options, asRowType(vectors[0]->type()));
    for (const auto& vector : vectors) {
      writer->write(vector);
    }
    writer->flush();
    writer->close();
    return HiveConnectorSplitBuilder(path)
        .fileFormat(dwio::common::FileFormat::PARQUET)
        .build();
  }

  // Writes one file per element of 'vectors' and registers all the data as the
  // DuckDB reference table 'tmp'.
  std::vector<std::shared_ptr<connector::hive::HiveConnectorSplit>> writeFiles(
      const std::vector<RowVectorPtr>& vectors,
      int64_t rowsInRowGroup,
      bool enableDictionary = false) {
    std::vector<std::shared_ptr<connector::hive::HiveConnectorSplit>> splits;
    for (auto i = 0; i < vectors.size(); ++i) {
      splits.push_back(writeFile(
          {vectors[i]},
          rowsInRowGroup,
          fmt::format("f{}", i),
          enableDictionary));
    }
    createDuckDbTable(vectors);
    return splits;
  }

  // 'numColumns' BIGINT columns named c0..c<n>. Column i of row r is
  // r * 10 + i, except that every 'nullEvery'th row of column i > 0 is null
  // when 'nullEvery' is non-zero.
  RowVectorPtr makeBatch(
      int32_t numColumns,
      int32_t numRows,
      int64_t firstRow,
      int32_t nullEvery = 0) {
    std::vector<VectorPtr> children;
    std::vector<std::string> names;
    for (auto column = 0; column < numColumns; ++column) {
      names.push_back(fmt::format("c{}", column));
      if (nullEvery > 0 && column > 0) {
        children.push_back(
            makeFlatVector<int64_t>(
                numRows,
                [&](auto row) { return (firstRow + row) * 10 + column; },
                [&](auto row) { return (row + column) % nullEvery == 0; }));
      } else {
        children.push_back(makeFlatVector<int64_t>(numRows, [&](auto row) {
          return (firstRow + row) * 10 + column;
        }));
      }
    }
    return makeRowVector(names, children);
  }

  std::vector<RowVectorPtr> makeBatches(
      int32_t numColumns,
      int32_t numBatches,
      int32_t rowsPerBatch,
      int32_t nullEvery = 0) {
    std::vector<RowVectorPtr> batches;
    for (auto i = 0; i < numBatches; ++i) {
      batches.push_back(
          makeBatch(numColumns, rowsPerBatch, i * rowsPerBatch, nullEvery));
    }
    return batches;
  }

  std::shared_ptr<Task> assertQuery(
      const core::PlanNodePtr& plan,
      const std::vector<std::shared_ptr<connector::hive::HiveConnectorSplit>>&
          splits,
      const std::string& duckDbSql) {
    std::vector<std::shared_ptr<connector::ConnectorSplit>> connectorSplits(
        splits.begin(), splits.end());
    return AssertQueryBuilder(plan, duckDbQueryRunner_)
        .splits(connectorSplits)
        .maxDrivers(numDrivers_)
        .assertResults(duckDbSql);
  }

  static RowTypePtr rowType(int32_t numColumns) {
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    for (auto i = 0; i < numColumns; ++i) {
      names.push_back(fmt::format("c{}", i));
      types.push_back(BIGINT());
    }
    return ROW(std::move(names), std::move(types));
  }

  std::shared_ptr<TempDirectoryPath> tempDirectory_;
  int32_t numDrivers_{1};
};

TEST_P(WaveParquetTableScanTest, basic) {
  auto type = rowType(2);
  auto splits = writeFiles(makeBatches(2, 1, 20'000), 20'000);
  auto task = assertQuery(
      PlanBuilder(pool_.get()).tableScan(type).planNode(),
      splits,
      "SELECT * FROM tmp");
  auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(20'000, stats.at("0").rawInputRows);
}

TEST_P(WaveParquetTableScanTest, prune) {
  // The file has four columns, the scan reads two of them in a different
  // order than the file.
  auto splits = writeFiles(makeBatches(4, 2, 10'000), 10'000);
  auto scanType = ROW({"c2", "c0"}, {BIGINT(), BIGINT()});
  assertQuery(
      PlanBuilder(pool_.get()).tableScan(scanType).planNode(),
      splits,
      "SELECT c2, c0 FROM tmp");
}

TEST_P(WaveParquetTableScanTest, nulls) {
  auto type = rowType(3);
  auto splits = writeFiles(makeBatches(3, 2, 10'000, /*nullEvery=*/7), 10'000);
  assertQuery(
      PlanBuilder(pool_.get()).tableScan(type).planNode(),
      splits,
      "SELECT * FROM tmp");
}

TEST_P(WaveParquetTableScanTest, filterAboveScan) {
  auto type = rowType(3);
  auto splits = writeFiles(makeBatches(3, 2, 10'000), 4'000);
  // The filter is a FilterProject above the scan, so it runs on the GPU
  // instead of in the CPU Parquet reader.
  auto plan = PlanBuilder(pool_.get())
                  .tableScan(type)
                  .filter("c0 < 100000")
                  .project({"c0", "c1 + 1 as c1", "c2"})
                  .planNode();
  assertQuery(plan, splits, "SELECT c0, c1 + 1, c2 FROM tmp WHERE c0 < 100000");
}

TEST_P(WaveParquetTableScanTest, filterAboveScanWithNulls) {
  auto type = rowType(3);
  auto splits = writeFiles(makeBatches(3, 2, 10'000, /*nullEvery=*/5), 3'000);
  auto plan = PlanBuilder(pool_.get())
                  .tableScan(type)
                  .filter("c1 < 100000")
                  .project({"c0", "c1", "c2 + 1 as c2"})
                  .planNode();
  assertQuery(plan, splits, "SELECT c0, c1, c2 + 1 FROM tmp WHERE c1 < 100000");
}

TEST_P(WaveParquetTableScanTest, manyRowGroups) {
  auto type = rowType(2);
  // 25 row groups of 1000 rows.
  auto splits = writeFiles(makeBatches(2, 1, 25'000), 1'000);
  assertQuery(
      PlanBuilder(pool_.get()).tableScan(type).planNode(),
      splits,
      "SELECT * FROM tmp");
}

TEST_P(WaveParquetTableScanTest, multipleSplits) {
  auto type = rowType(2);
  auto splits = writeFiles(makeBatches(2, 4, 7'777), 2'000);
  assertQuery(
      PlanBuilder(pool_.get()).tableScan(type).planNode(),
      splits,
      "SELECT * FROM tmp");
}

TEST_P(WaveParquetTableScanTest, benchmarkScaleSplit) {
  // 8 BIGINT columns of 200'000 rows, so 12.8 MB of values in one split, which
  // is the order of magnitude a benchmark reads and two orders above every
  // other fixture in this file. Covers a split that spans many batches and many
  // row groups.
  //
  // This is deliberately not a test of the sharded staging copy in
  // SplitStaging::transfer() (wave/dwio/FormatData.cpp). That copy stages
  // encoded bytes for GPU decode, and this reader bypasses the whole staging
  // and decode layer by uploading values the CPU already decoded, so no split
  // size can reach it from here. The sharded copy is covered by
  // TableScanTest.shardedStaging over the mock format.
  constexpr int32_t kNumColumns = 8;
  constexpr int32_t kNumRows = 200'000;
  auto splits = writeFiles(makeBatches(kNumColumns, 1, kNumRows), 50'000);
  ASSERT_EQ(1, splits.size());
  // Keeps the fixture at benchmark scale if someone retunes the row count.
  EXPECT_GT(std::filesystem::file_size(splits[0]->filePath), 8'000'000);
  assertQuery(
      PlanBuilder(pool_.get()).tableScan(rowType(kNumColumns)).planNode(),
      splits,
      "SELECT * FROM tmp");
}

TEST_P(WaveParquetTableScanTest, dictionaryEncoded) {
  auto type = rowType(2);
  // Few distinct values, so the writer dictionary encodes the columns and the
  // CPU reader hands out dictionary vectors, which the reader must flatten
  // before the transfer.
  std::vector<RowVectorPtr> batches = {makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>(20'000, [](auto row) { return row % 5; }),
       makeFlatVector<int64_t>(20'000, [](auto row) { return row % 7; })})};
  auto splits = writeFiles(batches, 5'000, /*enableDictionary=*/true);
  assertQuery(
      PlanBuilder(pool_.get()).tableScan(type).planNode(),
      splits,
      "SELECT * FROM tmp");
}

TEST_P(WaveParquetTableScanTest, filterPushedIntoScan) {
  auto type = rowType(2);
  auto splits = writeFiles(makeBatches(2, 2, 10'000), 2'500);
  // The filter is in the table handle, so the CPU Parquet reader evaluates it
  // and the reader sees batches with fewer rows than requested, some of them
  // empty. Correct but it measures CPU filtering, see the class comment of
  // WaveParquetSplitReader.
  auto plan =
      PlanBuilder(pool_.get()).tableScan(type, {"c0 < 50000"}).planNode();
  assertQuery(plan, splits, "SELECT * FROM tmp WHERE c0 < 50000");
}

TEST_P(WaveParquetTableScanTest, aggregation) {
  auto type = rowType(2);
  auto splits = writeFiles(makeBatches(2, 3, 10'000), 5'000);
  auto plan = PlanBuilder(pool_.get())
                  .tableScan(type)
                  .singleAggregation({}, {"sum(c0)", "sum(c1)"})
                  .planNode();
  assertQuery(plan, splits, "SELECT sum(c0), sum(c1) FROM tmp");
}

TEST_P(WaveParquetTableScanTest, unsupportedType) {
  // Wave code generation is BIGINT only. An INTEGER column must fail with an
  // explicit error rather than produce wrong results.
  auto vector = makeRowVector(
      {"c0"}, {makeFlatVector<int32_t>(1'000, [](auto row) { return row; })});
  auto splits = writeFiles({vector}, 1'000);
  auto plan =
      PlanBuilder(pool_.get()).tableScan(ROW({"c0"}, {INTEGER()})).planNode();
  VELOX_ASSERT_THROW(assertQuery(plan, splits, "SELECT * FROM tmp"), "");
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    WaveParquetTableScanTests,
    WaveParquetTableScanTest,
    testing::ValuesIn(params()));

} // namespace
