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

#include "velox/experimental/wave/dwio/parquet/WaveParquetSplitReader.h"

#include <algorithm>

#include "velox/common/time/Timer.h"
#include "velox/experimental/wave/exec/Vectors.h"

DECLARE_int32(wave_max_reader_batch_rows);

namespace facebook::velox::wave {
namespace {

// The Wave TableScan is always the first operator of the WaveDriver, so its
// LaunchControl key is 0. ReadStream::makeControl() relies on the same.
constexpr int32_t kScanOperatorId = 0;

// The CPU reader wants a map of info columns ($path, $bucket, ...).
// SplitReaderParams does not carry them, so the Wave path has none.
const std::unordered_map<std::string, connector::hive::FileColumnHandlePtr>&
emptyInfoColumns() {
  static const std::
      unordered_map<std::string, connector::hive::FileColumnHandlePtr>
          empty;
  return empty;
}

// Extracts the top level column name to operand mapping from 'defines'. The
// DefinesMap of a TableScan is keyed on the Subfield of each scan output
// column (CompileState::rowTypeToOperands), so a single element path is a top
// level column. pathToOperand() in Wave.cpp does the same lookup but depends
// on the thread local subfield map being set, which is only true while the
// split is being added.
folly::F14FastMap<std::string, const AbstractOperand*> topLevelDefines(
    const DefinesMap* defines) {
  VELOX_CHECK_NOT_NULL(
      defines,
      "Wave TableScan must set the output operands before adding a split");
  folly::F14FastMap<std::string, const AbstractOperand*> result;
  for (const auto& [value, operand] : *defines) {
    if (value.subfield == nullptr || operand == nullptr) {
      continue;
    }
    const auto& path = value.subfield->path();
    if (path.size() != 1) {
      continue;
    }
    if (auto* field =
            dynamic_cast<const common::Subfield::NestedField*>(path[0].get())) {
      result[field->name()] = operand;
    }
  }
  return result;
}

std::string definedColumnNames(
    const folly::F14FastMap<std::string, const AbstractOperand*>& map) {
  std::vector<std::string> names;
  names.reserve(map.size());
  for (const auto& [name, _] : map) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return folly::join(", ", names);
}

} // namespace

WaveParquetSplitReader::WaveParquetSplitReader(
    const std::shared_ptr<connector::ConnectorSplit>& split,
    const SplitReaderParams& params,
    const DefinesMap* defines)
    : params_(params),
      hiveSplit_(
          std::dynamic_pointer_cast<const connector::hive::HiveConnectorSplit>(
              split)),
      nameToOperand_(topLevelDefines(defines)),
      metadataIoStats_(std::make_shared<io::IoStatistics>()),
      ioStats_(std::make_shared<IoStats>()) {
  VELOX_CHECK_NOT_NULL(hiveSplit_, "Wave Parquet reader needs a Hive split");
  VELOX_CHECK_EQ(
      static_cast<int32_t>(hiveSplit_->fileFormat),
      static_cast<int32_t>(dwio::common::FileFormat::PARQUET));
  // Reverses the cast that WaveHiveDataSource::registerWaveDelegate() does on
  // the way in. HiveColumnHandle extends FileColumnHandle.
  auto* partitionKeys = reinterpret_cast<const std::unordered_map<
      std::string,
      connector::hive::FileColumnHandlePtr>*>(params_.partitionKeys);
  // 'params_.scanSpec' carries the filters and the projected columns the
  // planner pushed into the table handle. Giving it to the CPU reader means
  // the filters are evaluated on the CPU. GPU filter pushdown would instead
  // keep the filters out of this ScanSpec and turn them into Wave filter
  // instructions on the uploaded columns in schedule().
  cpuReader_ = std::make_unique<connector::hive::HiveSplitReader>(
      hiveSplit_,
      params_.hiveTableHandle,
      partitionKeys,
      params_.connectorQueryCtx,
      params_.hiveConfig,
      params_.readerOutputType,
      params_.ioStatistics,
      metadataIoStats_,
      ioStats_,
      params_.fileHandleFactory,
      params_.executor,
      params_.scanSpec,
      &emptyInfoColumns());
}

void WaveParquetSplitReader::configureReaderOptions() {
  cpuReader_->configureReaderOptions(/*randomSkip=*/nullptr);
}

void WaveParquetSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats) {
  cpuReader_->prepareSplit(std::move(metadataFilter), runtimeStats);
}

bool WaveParquetSplitReader::emptySplit() {
  return cpuReader_->emptySplit();
}

void WaveParquetSplitReader::readNextBatch() {
  if (finished_) {
    return;
  }
  if (cpuReader_->emptySplit()) {
    finished_ = true;
    return;
  }
  if (!output_) {
    output_ = BaseVector::create(cpuReader_->readerOutputType(), 0, pool());
  }
  const auto batchRows = std::max<int32_t>(1, FLAGS_wave_max_reader_batch_rows);
  uint64_t nanos = 0;
  {
    NanosecondTimer timer(&nanos);
    for (;;) {
      const auto scanned = cpuReader_->next(batchRows, output_);
      scannedRows_ += scanned;
      if (scanned == 0) {
        finished_ = true;
        pendingRows_ = 0;
        break;
      }
      const auto numRows = output_->asUnchecked<RowVector>()->size();
      if (numRows > 0) {
        pendingRows_ = numRows;
        ++numBatches_;
        break;
      }
      // Every row of the batch was dropped by a ScanSpec filter. Read on.
    }
  }
  decodeNanos_ += nanos;
}

int32_t WaveParquetSplitReader::canAdvance(WaveStream& /*stream*/) {
  if (pendingRows_ == 0) {
    readNextBatch();
  }
  return std::min<int32_t>(FLAGS_wave_max_reader_batch_rows, pendingRows_);
}

void WaveParquetSplitReader::ensureColumns() {
  if (!columns_.empty()) {
    return;
  }
  const auto& rowType = output_->type()->asRow();
  for (auto i = 0; i < rowType.size(); ++i) {
    const auto& name = rowType.nameOf(i);
    auto it = nameToOperand_.find(name);
    VELOX_CHECK(
        it != nameToOperand_.end(),
        "Wave Parquet split reader found no operand for reader output column "
        "'{}'. Columns are matched to Wave operands by name, so the table scan "
        "output names must be the Hive column names and the scan must not read "
        "extra columns for a remaining filter. Wave defines: {}",
        name,
        definedColumnNames(nameToOperand_));
    const auto& type = rowType.childAt(i);
    VELOX_USER_CHECK_EQ(
        type->kind(),
        TypeKind::BIGINT,
        "Wave Parquet split reader supports only BIGINT columns, column '{}' "
        "is {}. Wave code generation is BIGINT only, see cudaTypeName() in "
        "WaveGen.cpp",
        name,
        type->toString());
    columns_.push_back(Column{it->second->id, i});
  }
  std::sort(columns_.begin(), columns_.end(), [](auto& left, auto& right) {
    return left.operandId < right.operandId;
  });
}

void WaveParquetSplitReader::schedule(WaveStream& stream, int32_t maxRows) {
  if (pendingRows_ == 0) {
    readNextBatch();
  }
  VELOX_CHECK_GT(
      pendingRows_, 0, "Wave Parquet split reader scheduled with no rows");
  ensureColumns();
  const auto numRows = pendingRows_;
  if (maxRows > 0) {
    VELOX_CHECK_LE(numRows, maxRows);
  }
  auto* row = output_->asUnchecked<RowVector>();
  nullFlags_.clear();
  nullFlags_.resize(columns_.size());
  OperandSet ids;
  std::vector<WaveVectorPtr> waveVectors;
  std::vector<Transfer> transfers;
  waveVectors.reserve(columns_.size());
  transfers.reserve(2 * columns_.size());
  auto& arena = stream.arena();
  for (auto i = 0; i < columns_.size(); ++i) {
    const auto& column = columns_[i];
    auto& child = row->childAt(column.channel);
    if (child->encoding() != VectorEncoding::Simple::FLAT) {
      // A dictionary or constant column, e.g. a dictionary encoded Parquet
      // column or a partition key. Wave transfers flat values only.
      BaseVector::flattenVector(child);
      ++numFlattened_;
    }
    VELOX_CHECK(
        child->encoding() == VectorEncoding::Simple::FLAT,
        "Wave Parquet split reader could not flatten column {}",
        column.channel);
    VELOX_CHECK_GE(child->size(), numRows);
    const auto* rawNulls = child->rawNulls();
    const bool nullable = rawNulls != nullptr;
    stream.setNullable(*stream.operandAt(column.operandId), nullable);
    ids.add(column.operandId);
    auto waveVector = WaveVector::create(child->type(), arena);
    waveVector->resize(numRows, nullable);
    const auto valueBytes = numRows * child->type()->cppSizeInBytes();
    transfers.emplace_back(
        child->values()->as<char>(), waveVector->values<char>(), valueBytes);
    uploadedBytes_ += valueBytes;
    if (nullable) {
      // Wave nulls are one byte per row, Velox nulls are one bit per row.
      auto flagBuffer = AlignedBuffer::allocate<uint8_t>(numRows, pool());
      auto* flags = flagBuffer->asMutable<uint8_t>();
      for (auto rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        flags[rowIdx] = bits::isBitSet(rawNulls, rowIdx) ? kNotNull : kNull;
      }
      nullFlags_[i] = std::move(flagBuffer);
      transfers.emplace_back(flags, waveVector->nulls(), numRows);
      uploadedBytes_ += numRows;
    }
    waveVectors.push_back(std::move(waveVector));
  }

  // Mirrors Values::schedule(): size the stream, make the BlockStatus for the
  // scan and then start the host to device transfer, which registers the
  // operands with the stream. All output operands get a vector, so the
  // Executable::ensureLazyArrived() contract for scan executables does not
  // apply: WaveStream::getOutput() only calls it when an output vector is
  // missing.
  folly::Range<Executable**> noExes(nullptr, nullptr);
  const auto numBlocks = bits::roundUp(numRows, kBlockSize) / kBlockSize;
  stream.setNumRows(numRows);
  stream.prepareProgramLaunch(
      kScanOperatorId, 0, numRows, noExes, numBlocks, nullptr, nullptr);
  Executable::startTransfer(
      std::move(ids), std::move(waveVectors), std::move(transfers), stream);
  uploadedRows_ += numRows;
  pendingRows_ = 0;
}

bool WaveParquetSplitReader::isFinished() const {
  return finished_ && pendingRows_ == 0;
}

uint64_t WaveParquetSplitReader::getCompletedBytes() {
  return params_.ioStatistics ? params_.ioStatistics->rawBytesRead() : 0;
}

uint64_t WaveParquetSplitReader::getCompletedRows() {
  return scannedRows_;
}

std::unordered_map<std::string, RuntimeCounter>
WaveParquetSplitReader::runtimeStats() {
  // All counters are for this split only: WaveHiveDataSource makes a new split
  // reader for every split and TableScan::updateStats() adds the values.
  dwio::common::RuntimeStatistics readerStats;
  cpuReader_->updateRuntimeStats(readerStats);
  std::unordered_map<std::string, RuntimeCounter> result;
  result.emplace(
      "waveParquetCpuDecodeNanos",
      RuntimeCounter(decodeNanos_, RuntimeCounter::Unit::kNanos));
  result.emplace(
      "waveParquetUploadedBytes",
      RuntimeCounter(uploadedBytes_, RuntimeCounter::Unit::kBytes));
  result.emplace("waveParquetUploadedRows", RuntimeCounter(uploadedRows_));
  result.emplace("waveParquetBatches", RuntimeCounter(numBatches_));
  if (numFlattened_ > 0) {
    result.emplace(
        "waveParquetFlattenedVectors", RuntimeCounter(numFlattened_));
  }
  if (readerStats.processedStrides > 0) {
    result.emplace(
        "processedStrides", RuntimeCounter(readerStats.processedStrides));
  }
  if (readerStats.skippedStrides > 0) {
    result.emplace(
        "skippedStrides", RuntimeCounter(readerStats.skippedStrides));
  }
  return result;
}

namespace {
class WaveParquetSplitReaderFactory : public WaveSplitReaderFactory {
 public:
  std::shared_ptr<WaveSplitReader> create(
      const std::shared_ptr<connector::ConnectorSplit>& split,
      const SplitReaderParams& params,
      const DefinesMap* defines) override {
    auto* hiveSplit =
        dynamic_cast<connector::hive::HiveConnectorSplit*>(split.get());
    if (hiveSplit == nullptr ||
        hiveSplit->fileFormat != dwio::common::FileFormat::PARQUET) {
      return nullptr;
    }
    return std::make_shared<WaveParquetSplitReader>(split, params, defines);
  }
};
} // namespace

// static
void WaveParquetSplitReader::registerSplitReader() {
  static bool registered = false;
  if (registered) {
    return;
  }
  registered = true;
  WaveSplitReader::registerFactory(
      std::make_unique<WaveParquetSplitReaderFactory>());
}

} // namespace facebook::velox::wave
