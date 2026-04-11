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

/// End-to-end tests for the cudf Iceberg connector's handling of equality
/// delete files. Ported from the upstream EqualityDeleteFileReaderTest.
///
/// Data files are written as Parquet (via cudf writer) while equality delete
/// files are written as DWRF (via the upstream velox::dwrf::Writer) since
/// they are read by the upstream Velox EqualityDeleteFileReader, not cudf.

#include "velox/experimental/cudf/connectors/hive/iceberg/tests/CudfIcebergTestBase.h"

#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReader.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <folly/Singleton.h>

#include <fstream>
#include <map>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector::hive::iceberg;
using facebook::velox::common::testutil::TempFilePath;
using facebook::velox::cudf_velox::connector::hive::iceberg::
    CudfDeletionVectorReader;

namespace {

/// Serializes a roaring bitmap in the portable format (no-run variant,
/// cookie = 12346). Supports only array containers (cardinality <= 4096).
std::string serializeDvBitmap(const std::vector<int64_t>& positions) {
  if (positions.empty()) {
    std::string data(8, '\0');
    uint32_t cookie = 12346;
    uint32_t numContainers = 0;
    std::memcpy(data.data(), &cookie, 4);
    std::memcpy(data.data() + 4, &numContainers, 4);
    return data;
  }

  std::map<uint16_t, std::vector<uint16_t>> containers;
  for (auto pos : positions) {
    auto key = static_cast<uint16_t>(pos >> 16);
    auto low = static_cast<uint16_t>(pos & 0xFFFF);
    containers[key].push_back(low);
  }
  for (auto& [key, vals] : containers) {
    std::sort(vals.begin(), vals.end());
  }

  uint32_t numContainers = static_cast<uint32_t>(containers.size());
  std::string data;
  uint32_t cookie = 12346;
  data.append(reinterpret_cast<const char*>(&cookie), 4);
  data.append(reinterpret_cast<const char*>(&numContainers), 4);

  for (auto& [key, vals] : containers) {
    uint16_t cardMinus1 = static_cast<uint16_t>(vals.size() - 1);
    data.append(reinterpret_cast<const char*>(&key), 2);
    data.append(reinterpret_cast<const char*>(&cardMinus1), 2);
  }

  if (numContainers >= 4) {
    uint32_t offset = 4 + 4 + numContainers * 4 + numContainers * 4;
    for (auto& [key, vals] : containers) {
      data.append(reinterpret_cast<const char*>(&offset), 4);
      offset += static_cast<uint32_t>(vals.size()) * 2;
    }
  }

  for (auto& [key, vals] : containers) {
    for (auto v : vals) {
      data.append(reinterpret_cast<const char*>(&v), 2);
    }
  }

  return data;
}

/// Writes a raw roaring bitmap to a temp file and returns the path.
std::shared_ptr<TempFilePath> writeDvToFile(const std::string& bitmapData) {
  auto tempFile = TempFilePath::create();
  std::ofstream out(tempFile->getPath(), std::ios::binary | std::ios::trunc);
  out.write(bitmapData.data(), static_cast<std::streamsize>(bitmapData.size()));
  out.close();
  return tempFile;
}

/// Creates an IcebergDeleteFile for a deletion vector.
IcebergDeleteFile makeDvDeleteFile(
    const std::string& filePath,
    uint64_t fileSize,
    int64_t recordCount,
    int64_t dataSequenceNumber = 0) {
  std::unordered_map<int32_t, std::string> lowerBounds;
  std::unordered_map<int32_t, std::string> upperBounds;
  lowerBounds[CudfDeletionVectorReader::kDvOffsetFieldId] = "0";
  upperBounds[CudfDeletionVectorReader::kDvLengthFieldId] =
      std::to_string(fileSize);
  return IcebergDeleteFile(
      FileContent::kDeletionVector,
      filePath,
      facebook::velox::dwio::common::FileFormat::UNKNOWN,
      recordCount,
      fileSize,
      {},
      std::move(lowerBounds),
      std::move(upperBounds),
      dataSequenceNumber);
}

} // namespace

namespace facebook::velox::cudf_velox::exec::test {

class CudfEqualityDeleteFileReaderTest
    : public CudfIcebergTestBase,
      public ::testing::WithParamInterface<DeleteFileFormat> {};

/// Basic single-column equality delete.
/// (Ported from upstream EqualityDeleteFileReaderTest::basicSingleColumnDelete)
TEST_P(CudfEqualityDeleteFileReaderTest, basicSingleColumnDelete) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      makeFlatVector<int64_t>({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({3, 7}),
  });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2, 4, 5, 6, 8, 9}),
      makeFlatVector<int64_t>({10, 11, 12, 14, 15, 16, 18, 19}),
  });

  assertEqualResults({expected}, {result});
}

/// Multi-column equality deletes (both columns must match).
/// (Ported from upstream EqualityDeleteFileReaderTest::multiColumnDelete)
TEST_P(CudfEqualityDeleteFileReaderTest, multiColumnDelete) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int32_t>({10, 20, 30, 10, 20}),
      makeFlatVector<int64_t>({100, 200, 300, 400, 500}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int32_t>({2, 5, 1}),
      makeFlatVector<int32_t>({20, 20, 20}),
  });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 3, 4}),
      makeFlatVector<int32_t>({10, 30, 10}),
      makeFlatVector<int64_t>({100, 300, 400}),
  });

  assertEqualResults({expected}, {result});
}

/// When no rows match, all rows survive.
/// (Ported from upstream EqualityDeleteFileReaderTest::noMatchingDeletes)
TEST_P(CudfEqualityDeleteFileReaderTest, noMatchingDeletes) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({100, 200}),
  });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });

  assertEqualResults({expected}, {result});
}

/// All rows deleted.
/// (Ported from upstream EqualityDeleteFileReaderTest::allRowsDeleted)
TEST_P(CudfEqualityDeleteFileReaderTest, allRowsDeleted) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  EXPECT_EQ(result->size(), 0);
}

/// Equality deletes with higher sequence number should apply.
/// (Ported from upstream
/// EqualityDeleteFileReaderTest::sequenceNumberDeleteApplies)
TEST_P(CudfEqualityDeleteFileReaderTest, sequenceNumberDeleteApplies) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
  });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
      /*dataSequenceNumber=*/3);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 5}),
      makeFlatVector<int64_t>({10, 30, 50}),
  });

  assertEqualResults({expected}, {result});
}

/// Equality deletes with lower sequence number should be skipped.
/// (Ported from upstream
/// EqualityDeleteFileReaderTest::sequenceNumberDeleteSkipped)
TEST_P(CudfEqualityDeleteFileReaderTest, sequenceNumberDeleteSkipped) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
      /*dataSequenceNumber=*/5);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });

  assertEqualResults({expected}, {result});
}

/// Equal sequence numbers should also skip.
/// (Ported from upstream
/// EqualityDeleteFileReaderTest::sequenceNumberEqualSkipped)
TEST_P(CudfEqualityDeleteFileReaderTest, sequenceNumberEqualSkipped) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
      /*dataSequenceNumber=*/5);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });

  assertEqualResults({expected}, {result});
}

/// Sequence number 0 means legacy/unassigned — always apply.
/// (Ported from upstream
/// EqualityDeleteFileReaderTest::sequenceNumberZeroAlwaysApplies)
TEST_P(CudfEqualityDeleteFileReaderTest, sequenceNumberZeroAlwaysApplies) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({2}),
  });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/0);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
      /*dataSequenceNumber=*/10);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3}),
  });

  assertEqualResults({expected}, {result});
}

/// Mixed sequence numbers: only delete files with higher seqNum apply.
/// (Ported from upstream EqualityDeleteFileReaderTest::mixedSequenceNumbers)
TEST_P(CudfEqualityDeleteFileReaderTest, mixedSequenceNumbers) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  const auto eqDeleteFileFormat = GetParam();

  auto deleteData1 = makeRowVector({
      makeFlatVector<int64_t>({2}),
  });
  auto eqDeleteFile1 = TempFilePath::create();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile1->getPath(), {deleteData1});
  IcebergDeleteFile icebergDeleteFile1(
      FileContent::kEqualityDeletes,
      eqDeleteFile1->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      1,
      getFileSize(eqDeleteFile1->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/10);

  auto deleteData2 = makeRowVector({
      makeFlatVector<int64_t>({4}),
  });
  auto eqDeleteFile2 = TempFilePath::create();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile2->getPath(), {deleteData2});
  IcebergDeleteFile icebergDeleteFile2(
      FileContent::kEqualityDeletes,
      eqDeleteFile2->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      1,
      getFileSize(eqDeleteFile2->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/3);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile1, icebergDeleteFile2},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
      /*dataSequenceNumber=*/5);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 30, 40, 50}),
  });

  assertEqualResults({expected}, {result});
}

/// Mixed-format test (not parameterized)

INSTANTIATE_TEST_SUITE_P(
    DeleteFormats,
    CudfEqualityDeleteFileReaderTest,
    ::testing::Values(DeleteFileFormat::DWRF, DeleteFileFormat::PARQUET),
    [](const auto& info) {
      return info.param == DeleteFileFormat::PARQUET ? "Parquet" : "Dwrf";
    });

class CudfMixedFormatEqualityDeleteTest : public CudfIcebergTestBase {};

/// Mixed delete file formats: one DWRF and one Parquet equality delete file.
TEST_F(CudfMixedFormatEqualityDeleteTest, mixedFormatDeleteFiles) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData1 = makeRowVector({
      makeFlatVector<int64_t>({2}),
  });
  auto eqDeleteFile1 = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::DWRF, eqDeleteFile1->getPath(), {deleteData1});
  IcebergDeleteFile icebergDeleteFile1(
      FileContent::kEqualityDeletes,
      eqDeleteFile1->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile1->getPath()),
      /*equalityFieldIds=*/{1});

  auto deleteData2 = makeRowVector({
      makeFlatVector<int64_t>({4}),
  });
  auto eqDeleteFile2 = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::PARQUET, eqDeleteFile2->getPath(), {deleteData2});
  IcebergDeleteFile icebergDeleteFile2(
      FileContent::kEqualityDeletes,
      eqDeleteFile2->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDeleteFile2->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(
      dataFile->getPath(), {icebergDeleteFile1, icebergDeleteFile2});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 5}),
      makeFlatVector<int64_t>({10, 30, 50}),
  });

  assertEqualResults({expected}, {result});
}

/// =========================================================================
/// Gap-exposing tests from Sirius project
/// =========================================================================

class CudfIcebergGapTests : public CudfIcebergTestBase {};

/// Combined positional + equality deletes on the same data file.
TEST_F(CudfIcebergGapTests, combinedPositionalAndEqualityDeletes) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      makeFlatVector<int64_t>({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Positional delete: remove rows at positions 1 and 3
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto posDeleteFile = TempFilePath::create();
  auto filePathVec = makeFlatVector<std::string>(
      2, [&](vector_size_t) { return dataFile->getPath(); });
  auto posVec = makeFlatVector<int64_t>({1, 3});
  auto posDeleteVector =
      makeRowVector({pathColumn->name, posColumn->name}, {filePathVec, posVec});
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      posDeleteFile->getPath(),
      std::vector<RowVectorPtr>{posDeleteVector});

  IcebergDeleteFile posIcebergDelete(
      FileContent::kPositionalDeletes,
      posDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(posDeleteFile->getPath()));

  // Equality delete: remove c0={5, 7}
  auto eqDeleteData = makeRowVector({
      makeFlatVector<int64_t>({5, 7}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::PARQUET, eqDeleteFile->getPath(), {eqDeleteData});

  IcebergDeleteFile eqIcebergDelete(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(
      dataFile->getPath(), {posIcebergDelete, eqIcebergDelete});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Positional removes pos 1,3 (c0=1,3). Equality removes c0=5,7.
  // Surviving: 0, 2, 4, 6, 8, 9
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({0, 2, 4, 6, 8, 9}),
      makeFlatVector<int64_t>({10, 12, 14, 16, 18, 19}),
  });

  assertEqualResults({expected}, {result});
}

/// Non-projected equality delete key column.
/// SELECT c1 but equality delete key is c0.
TEST_F(CudfIcebergGapTests, nonProjectedDeleteKeyColumn) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto fullType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto outputType = ROW({"c1"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto eqDeleteData = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::PARQUET, eqDeleteFile->getPath(), {eqDeleteData});

  IcebergDeleteFile eqIcebergDelete(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {eqIcebergDelete});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(fullType)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // c0=2,4 deleted -> c1=20,40 removed
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({10, 30, 50}),
  });

  assertEqualResults({expected}, {result});
}

/// Insert-delete-insert interleaving with sequence numbers.
TEST_F(CudfIcebergGapTests, insertDeleteInsertInterleaving) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
      makeFlatVector<int64_t>({200, 400}),
  });
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  auto eqDeleteData = makeRowVector({
      makeFlatVector<int64_t>({2}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::PARQUET, eqDeleteFile->getPath(), {eqDeleteData});

  IcebergDeleteFile eqIcebergDelete(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  // File 1: data seq=1, delete seq=2 -> delete APPLIES
  auto splits1 = makeIcebergSplits(
      dataFile1->getPath(), {eqIcebergDelete}, {}, 1, /*dataSeq=*/1);
  // File 2: data seq=3, delete seq=2 -> delete SKIPPED
  auto splits2 = makeIcebergSplits(
      dataFile2->getPath(), {eqIcebergDelete}, {}, 1, /*dataSeq=*/3);

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      allSplits;
  allSplits.insert(allSplits.end(), splits1.begin(), splits1.end());
  allSplits.insert(allSplits.end(), splits2.begin(), splits2.end());

  auto plan = makeTableScanPlan(rowType);
  auto result =
      AssertQueryBuilder(plan).splits(allSplits).copyResults(pool());

  // File1 loses c0=2, file2 keeps c0=2
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 2, 4}),
      makeFlatVector<int64_t>({10, 30, 200, 400}),
  });

  assertEqualResults({expected}, {result});
}

/// Multiple equality delete files at different sequence numbers targeting
/// overlapping values. Simulates: write(seq=1), delete c0=2(seq=2),
/// insert c0=2 back(seq=3), delete c0=2 again(seq=4).
/// Only the seq=4 delete should affect the seq=3 data.
TEST_F(CudfIcebergGapTests, multipleDeletesAtDifferentSequenceNumbers) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  // File 1: original data (seq=1) — has c0=2
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  // File 2: re-inserted data (seq=3) — c0=2 re-appears with new c1
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({2, 5}),
      makeFlatVector<int64_t>({200, 500}),
  });
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  // File 3: latest data (seq=5)
  auto data3 = makeRowVector({
      makeFlatVector<int64_t>({2, 6}),
      makeFlatVector<int64_t>({2000, 6000}),
  });
  auto dataFile3 = TempFilePath::create();
  writeToFile(dataFile3->getPath(), data3);

  // Delete 1 at seq=2: delete c0=2
  auto del1 = makeRowVector({makeFlatVector<int64_t>({2})});
  auto delFile1 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, delFile1->getPath(), {del1});
  IcebergDeleteFile icebergDel1(
      FileContent::kEqualityDeletes,
      delFile1->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(delFile1->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  // Delete 2 at seq=4: delete c0=2 again
  auto del2 = makeRowVector({makeFlatVector<int64_t>({2})});
  auto delFile2 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, delFile2->getPath(), {del2});
  IcebergDeleteFile icebergDel2(
      FileContent::kEqualityDeletes,
      delFile2->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(delFile2->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/4);

  // File 1 (seq=1): both del1(seq=2) and del2(seq=4) apply -> c0=2 deleted
  auto splits1 = makeIcebergSplits(
      dataFile1->getPath(), {icebergDel1, icebergDel2}, {}, 1, 1);
  // File 2 (seq=3): del1(seq=2) skipped (2<3), del2(seq=4) applies -> c0=2 deleted
  auto splits2 = makeIcebergSplits(
      dataFile2->getPath(), {icebergDel1, icebergDel2}, {}, 1, 3);
  // File 3 (seq=5): del1(seq=2) skipped, del2(seq=4) skipped (4<5) -> c0=2 survives
  auto splits3 = makeIcebergSplits(
      dataFile3->getPath(), {icebergDel1, icebergDel2}, {}, 1, 5);

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());
  all.insert(all.end(), splits3.begin(), splits3.end());

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  // File1: c0=2 deleted (both deletes apply) -> (1,10), (3,30)
  // File2: c0=2 deleted (del2 applies) -> (5,500)
  // File3: c0=2 survives (no delete applies) -> (2,2000), (6,6000)
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 5, 2, 6}),
      makeFlatVector<int64_t>({10, 30, 500, 2000, 6000}),
  });

  assertEqualResults({expected}, {result});
}

/// Mixed positional + equality + deletion vector interleaving.
/// Tests the full DV -> positional -> equality pipeline with sequence numbers.
TEST_F(CudfIcebergGapTests, positionalAndEqualityWithSequenceNumbers) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  // Data file at seq=1
  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60, 70, 80}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Positional delete at seq=2: remove positions 0 and 7 (c0=10, c0=80)
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto posDeleteFile = TempFilePath::create();
  auto filePathVec = makeFlatVector<std::string>(
      2, [&](vector_size_t) { return dataFile->getPath(); });
  auto posVec = makeFlatVector<int64_t>({0, 7});
  auto posDeleteVector =
      makeRowVector({pathColumn->name, posColumn->name}, {filePathVec, posVec});
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      posDeleteFile->getPath(),
      std::vector<RowVectorPtr>{posDeleteVector});

  IcebergDeleteFile posDelete(
      FileContent::kPositionalDeletes,
      posDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(posDeleteFile->getPath()),
      /*equalityFieldIds=*/{},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  // Equality delete at seq=3: remove c0=30 and c0=60
  auto eqDel = makeRowVector({makeFlatVector<int64_t>({30, 60})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/3);

  // Data seq=1: both deletes apply (seq=2>=1, seq=3>1)
  auto splits = makeIcebergSplits(
      dataFile->getPath(), {posDelete, eqDelete}, {}, 1, 1);

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Positional: remove pos 0,7 (c0=10,80)
  // Equality: remove c0=30,60
  // Surviving: (20,2), (40,4), (50,5), (70,7)
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({20, 40, 50, 70}),
      makeFlatVector<int64_t>({2, 4, 5, 7}),
  });

  assertEqualResults({expected}, {result});
}

/// Multi-column equality delete with overlapping but non-matching values.
/// Delete by (c0, c1) where c0 matches but c1 doesn't — should NOT delete.
TEST_F(CudfIcebergGapTests, multiColumnPartialMatchDoesNotDelete) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 2, 1}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeFlatVector<int64_t>({100, 200, 300, 400, 500}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Delete where (c0=2, c1=20) — should only delete row (2,20,200),
  // NOT row (2,40,400) even though c0 matches
  auto eqDel = makeRowVector({
      makeFlatVector<int64_t>({2}),
      makeFlatVector<int64_t>({20}),
  });
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1, 2});

  auto splits = makeIcebergSplits(dataFile->getPath(), {eqDelete});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Only (2,20,200) deleted. (2,40,400) survives because c1 doesn't match.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 2, 1}),
      makeFlatVector<int64_t>({10, 30, 40, 50}),
      makeFlatVector<int64_t>({100, 300, 400, 500}),
  });

  assertEqualResults({expected}, {result});
}

/// Equality delete where the delete value doesn't exist in any data file.
/// Should return all rows unchanged.
TEST_F(CudfIcebergGapTests, equalityDeleteNoMatchAcrossFiles) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});

  auto data1 = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  auto data2 = makeRowVector({makeFlatVector<int64_t>({4, 5, 6})});
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  // Delete c0=999 — doesn't exist anywhere
  auto eqDel = makeRowVector({makeFlatVector<int64_t>({999})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits1 = makeIcebergSplits(dataFile1->getPath(), {eqDelete});
  auto splits2 = makeIcebergSplits(dataFile2->getPath(), {eqDelete});
  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6}),
  });

  assertEqualResults({expected}, {result});
}

/// Hive partitioned Iceberg table: data file in a partition directory,
/// query selects both data columns and the partition column.
TEST_F(CudfIcebergGapTests, hivePartitionedTable) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Table schema: c0 (data), c1 (data), country (partition)
  auto fullType =
      ROW({"c0", "c1", "country"}, {BIGINT(), BIGINT(), VARCHAR()});

  // Data file only contains c0, c1 (partition column is NOT in parquet)
  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // The partition value comes from the split, not from the parquet file
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"country", "US"},
  };

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      /*deleteFiles=*/{},
      partitionKeys);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(ROW({"c0", "c1"}, {BIGINT(), BIGINT()}))
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Expected: data columns from parquet + partition column "US" filled in
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<std::string>({"US", "US", "US"}),
  });

  assertEqualResults({expected}, {result});
}

/// Hive partitioned table with equality deletes — delete key is a data column,
/// partition column should still be correctly synthesized.
TEST_F(CudfIcebergGapTests, hivePartitionWithEqualityDelete) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto fullType =
      ROW({"c0", "c1", "region"}, {BIGINT(), BIGINT(), VARCHAR()});
  auto dataColumns = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Equality delete on c0=2
  auto eqDel = makeRowVector({makeFlatVector<int64_t>({2})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"region", "APAC"},
  };

  auto splits = makeIcebergSplits(
      dataFile->getPath(), {eqDelete}, partitionKeys);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(dataColumns)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // c0=2 deleted, partition column "APAC" filled in
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 30, 40, 50}),
      makeFlatVector<std::string>({"APAC", "APAC", "APAC", "APAC"}),
  });

  assertEqualResults({expected}, {result});
}

/// Schema evolution: file 1 has [c0, c1], file 2 has [c0, c1, c2].
/// Query reads all three columns. File 1 should return NULL for c2.
TEST_F(CudfIcebergGapTests, schemaEvolutionAddedColumn) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto fullType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});

  // File 1: old schema, only c0 and c1
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeFlatVector<int64_t>({10, 20}),
  });
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  // File 2: new schema, has c0, c1, c2
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({3, 4}),
      makeFlatVector<int64_t>({30, 40}),
      makeFlatVector<int64_t>({300, 400}),
  });
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  auto splits1 = makeIcebergSplits(dataFile1->getPath());
  auto splits2 = makeIcebergSplits(dataFile2->getPath());

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(fullType)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  // File 1 rows: c2 should be NULL since it doesn't exist in that file
  // File 2 rows: all columns present
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 300, 400}),
  });

  assertEqualResults({expected}, {result});
}

/// Schema evolution with equality delete: delete key column exists in both
/// files but an extra column was added in the newer file.
TEST_F(CudfIcebergGapTests, schemaEvolutionWithEqualityDelete) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto fullType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});

  // File 1: old schema [c0, c1]
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  // File 2: new schema [c0, c1, c2]
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
      makeFlatVector<int64_t>({200, 400}),
      makeFlatVector<int64_t>({2000, 4000}),
  });
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  // Equality delete on c0=2 (exists in both files)
  auto eqDel = makeRowVector({makeFlatVector<int64_t>({2})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits1 = makeIcebergSplits(dataFile1->getPath(), {eqDelete});
  auto splits2 = makeIcebergSplits(dataFile2->getPath(), {eqDelete});

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(fullType)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  // c0=2 deleted from both files
  // File 1: (1,10,NULL), (3,30,NULL) survive
  // File 2: (4,400,4000) survives
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 4}),
      makeFlatVector<int64_t>({10, 30, 400}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 4000}),
  });

  assertEqualResults({expected}, {result});
}

/// =========================================================================
/// Spec edge-case tests — high and medium priority gaps
/// =========================================================================

/// NULL equality matching: per Iceberg spec, NULL == NULL is TRUE for
/// equality deletes (unlike standard SQL). Verifies that rows with NULL
/// values are correctly deleted when the delete file also contains NULL.
TEST_F(CudfIcebergGapTests, equalityDeleteNullMatchesNull) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  // Data with NULLs in c0
  auto baseData = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 3, std::nullopt, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Equality delete: c0 = NULL (should match rows where c0 IS NULL)
  auto eqDel = makeRowVector({
      makeNullableFlatVector<int64_t>({std::nullopt}),
  });
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {eqDelete});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Both NULL rows (c1=20, c1=40) should be deleted
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 5}),
      makeFlatVector<int64_t>({10, 30, 50}),
  });

  assertEqualResults({expected}, {result});
}

/// Multiple equality delete files with DIFFERENT key columns targeting
/// the same data file. Each delete file is applied independently per spec.
/// A row is deleted if it matches ANY of the applicable delete files.
TEST_F(CudfIcebergGapTests, multipleEqualityDeletesDifferentKeyColumns) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeFlatVector<int64_t>({100, 200, 300, 400, 500}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Delete file 1: delete where c0=2 (equalityFieldIds={1})
  auto eqDel1 = makeRowVector({makeFlatVector<int64_t>({2})});
  auto eqDelFile1 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile1->getPath(), {eqDel1});

  IcebergDeleteFile eqDelete1(
      FileContent::kEqualityDeletes,
      eqDelFile1->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile1->getPath()),
      /*equalityFieldIds=*/{1});

  // Delete file 2: delete where c1=40 (equalityFieldIds={2})
  // Column name must match the table schema column name for cudf read.
  auto eqDel2 = makeRowVector({"c1"}, {makeFlatVector<int64_t>({40})});
  auto eqDelFile2 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile2->getPath(), {eqDel2});

  IcebergDeleteFile eqDelete2(
      FileContent::kEqualityDeletes,
      eqDelFile2->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile2->getPath()),
      /*equalityFieldIds=*/{2});

  auto splits =
      makeIcebergSplits(dataFile->getPath(), {eqDelete1, eqDelete2});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // c0=2 (row 2) deleted by file 1, c1=40 (row 4) deleted by file 2
  // Surviving: rows 1, 3, 5
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 5}),
      makeFlatVector<int64_t>({10, 30, 50}),
      makeFlatVector<int64_t>({100, 300, 500}),
  });

  assertEqualResults({expected}, {result});
}

/// All rows deleted by equality deletes — verify that reading continues
/// past an empty chunk (DuckDB issue #624: empty chunk treated as EOS).
TEST_F(CudfIcebergGapTests, allRowsDeletedContinuesReading) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});

  // File 1: all values will be deleted
  auto data1 = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  // File 2: some values survive
  auto data2 = makeRowVector({makeFlatVector<int64_t>({4, 5, 6})});
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  // Delete c0 = {1, 2, 3, 4} — kills all of file 1 and one row of file 2
  auto eqDel =
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits1 = makeIcebergSplits(dataFile1->getPath(), {eqDelete});
  auto splits2 = makeIcebergSplits(dataFile2->getPath(), {eqDelete});
  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  // File 1: 0 rows survive. File 2: c0=5,6 survive.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({5, 6}),
  });

  assertEqualResults({expected}, {result});
}

/// Equality delete file with extra non-key columns (for CDC).
/// Only equalityFieldIds columns should be used for matching.
TEST_F(CudfIcebergGapTests, equalityDeleteWithExtraNonKeyColumns) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Delete file has c0 AND c1, but only c0 is in equalityFieldIds.
  // The c1 column is extra metadata (for CDC reconstruction).
  // Only c0 should be used for matching.
  auto eqDel = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
      makeFlatVector<int64_t>({999, 888}), // extra column, not used
  });
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1}); // Only c0 is the key

  auto splits = makeIcebergSplits(dataFile->getPath(), {eqDelete});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // c0=2 and c0=4 deleted regardless of extra c1 values in delete file
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 5}),
      makeFlatVector<int64_t>({10, 30, 50}),
  });

  assertEqualResults({expected}, {result});
}

/// Schema evolution: column added in the MIDDLE of the schema, not at the
/// end. File 1 has [c0, c2], file 2 has [c0, c1, c2]. Tests that the
/// schema reconciliation handles non-trailing missing columns.
TEST_F(CudfIcebergGapTests, schemaEvolutionColumnAddedInMiddle) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto fullType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});
  // File 1 was written before c1 was added — it only has c0, c2.
  // Use writeDeleteFile(PARQUET) to preserve the actual column names
  // (writeToFile renames columns to c0,c1,... by index).
  auto data1 = makeRowVector(
      {"c0", "c2"},
      {
          makeFlatVector<int64_t>({1, 2}),
          makeFlatVector<int64_t>({100, 200}),
      });
  auto dataFile1 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, dataFile1->getPath(), {data1});

  // File 2: full schema
  auto data2 = makeRowVector(
      {"c0", "c1", "c2"},
      {
          makeFlatVector<int64_t>({3, 4}),
          makeFlatVector<int64_t>({30, 40}),
          makeFlatVector<int64_t>({300, 400}),
      });
  auto dataFile2 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, dataFile2->getPath(), {data2});

  auto splits1 = makeIcebergSplits(dataFile1->getPath());
  auto splits2 = makeIcebergSplits(dataFile2->getPath());

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(fullType)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  // File 1: c1 should be NULL since the column doesn't exist in file
  // File 2: all columns present
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 30, 40}),
      makeFlatVector<int64_t>({100, 200, 300, 400}),
  });

  assertEqualResults({expected}, {result});
}

/// Empty data file (0 rows) with delete files attached. Should not error.
TEST_F(CudfIcebergGapTests, emptyDataFileWithDeletes) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  // Empty data file
  auto emptyData = makeRowVector({
      makeFlatVector<int64_t>({}),
      makeFlatVector<int64_t>({}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), emptyData);

  // Non-empty data file
  auto realData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), realData);

  // Equality delete on c0=2
  auto eqDel = makeRowVector({makeFlatVector<int64_t>({2})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  // Attach delete file to both the empty and non-empty data files
  auto splits1 = makeIcebergSplits(dataFile->getPath(), {eqDelete});
  auto splits2 = makeIcebergSplits(dataFile2->getPath(), {eqDelete});
  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  // Empty file contributes nothing, real file loses c0=2
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3}),
      makeFlatVector<int64_t>({10, 30}),
  });

  assertEqualResults({expected}, {result});
}

/// Partition column with INT32 type (not just string).
/// Tests that injectMissingColumns handles typed partition values.
TEST_F(CudfIcebergGapTests, partitionColumnInt32Type) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto fullType = ROW({"c0", "year"}, {BIGINT(), INTEGER()});
  auto dataColumns = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"year", "2025"},
  };

  auto splits = makeIcebergSplits(dataFile->getPath(), {}, partitionKeys);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(dataColumns)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int32_t>({2025, 2025, 2025}),
  });

  assertEqualResults({expected}, {result});
}

/// Partition column with INT64 type.
TEST_F(CudfIcebergGapTests, partitionColumnInt64Type) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto fullType = ROW({"c0", "timestamp_ms"}, {BIGINT(), BIGINT()});
  auto dataColumns = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"timestamp_ms", "1700000000000"},
  };

  auto splits = makeIcebergSplits(dataFile->getPath(), {}, partitionKeys);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(dataColumns)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<int64_t>(
          {1700000000000LL, 1700000000000LL, 1700000000000LL}),
  });

  assertEqualResults({expected}, {result});
}

/// Deletion vector combined with equality deletes on the same data file.
/// DV removes by position, equality delete removes by value. Both must apply.
TEST_F(CudfIcebergGapTests, deletionVectorPlusEqualityDelete) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60, 70, 80}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // DV: delete positions 0 and 7 (c0=10, c0=80)
  auto bitmapData = serializeDvBitmap({0, 7});
  auto dvFile = writeDvToFile(bitmapData);
  auto dvDelete = makeDvDeleteFile(
      dvFile->getPath(), bitmapData.size(), 2, /*dataSequenceNumber=*/2);

  // Equality delete: delete c0=30, c0=60
  auto eqDel = makeRowVector({makeFlatVector<int64_t>({30, 60})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/3);

  auto splits = makeIcebergSplits(
      dataFile->getPath(), {dvDelete, eqDelete}, {}, 1, /*dataSeq=*/1);

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // DV removes pos 0,7 (c0=10,80). Equality removes c0=30,60.
  // Surviving: (20,2), (40,4), (50,5), (70,7)
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({20, 40, 50, 70}),
      makeFlatVector<int64_t>({2, 4, 5, 7}),
  });

  assertEqualResults({expected}, {result});
}

/// Deletion vector combined with positional deletes (V2 + V3 coexistence).
/// Both should apply — DV removes some positions, positional delete removes
/// others. Requires the unified row mask pipeline.
TEST_F(CudfIcebergGapTests, deletionVectorPlusPositionalDelete) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // DV: delete positions 0 and 5 (c0=10, c0=60)
  auto bitmapData = serializeDvBitmap({0, 5});
  auto dvFile = writeDvToFile(bitmapData);
  auto dvDelete = makeDvDeleteFile(
      dvFile->getPath(), bitmapData.size(), 2, /*dataSequenceNumber=*/2);

  // Positional delete: delete position 2 (c0=30)
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto posDeleteFile = TempFilePath::create();
  auto filePathVec = makeFlatVector<std::string>(
      1, [&](vector_size_t) { return dataFile->getPath(); });
  auto posVec = makeFlatVector<int64_t>({2});
  auto posDeleteVector =
      makeRowVector({pathColumn->name, posColumn->name}, {filePathVec, posVec});
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      posDeleteFile->getPath(),
      std::vector<RowVectorPtr>{posDeleteVector});

  IcebergDeleteFile posDelete(
      FileContent::kPositionalDeletes,
      posDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(posDeleteFile->getPath()),
      /*equalityFieldIds=*/{},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  auto splits = makeIcebergSplits(
      dataFile->getPath(), {dvDelete, posDelete}, {}, 1, /*dataSeq=*/1);

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // DV removes pos 0,5 (c0=10,60). Positional removes pos 2 (c0=30).
  // Surviving: (20,2), (40,4), (50,5)
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({20, 40, 50}),
      makeFlatVector<int64_t>({2, 4, 5}),
  });

  assertEqualResults({expected}, {result});
}

} // namespace facebook::velox::cudf_velox::exec::test
