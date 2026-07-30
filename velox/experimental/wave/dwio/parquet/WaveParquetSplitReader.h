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

#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveSplitReader.h"
#include "velox/experimental/wave/exec/WaveSplitReader.h"

namespace facebook::velox::wave {

/// A WaveSplitReader that lets a Wave TableScan consume Parquet splits.
///
/// The split is decoded on the CPU by the normal Velox Parquet reader and the
/// decoded batch is copied to the device in schedule(), which is the same host
/// ingest path Values uses for host input vectors (see Values::schedule and
/// vectorsToDevice). The Wave staging/decode layer (FormatData, ColumnReader,
/// ReadStream, GpuDecode) is deliberately bypassed: it expects encoded bytes
/// and emits decode kernels, whereas here the values are already decoded.
///
/// Only flat, top level BIGINT columns are supported because Wave code
/// generation is BIGINT only (cudaTypeName in WaveGen.cpp) and only fixed
/// width values plus null flags are transferred. Anything else fails with an
/// explicit error instead of producing wrong results.
///
/// Filters: filters that the planner pushes into the table handle end up in
/// 'params.scanSpec' and are therefore evaluated by the CPU Parquet reader,
/// which returns only the passing rows. That is correct but it measures CPU
/// filtering, so benchmarks that want to measure Wave should keep filters in a
/// FilterProject above the scan. GPU filter pushdown would belong in
/// schedule(), by turning the ScanSpec filters into Wave filter instructions
/// on the uploaded columns rather than passing them to the CPU reader.
class WaveParquetSplitReader : public WaveSplitReader {
 public:
  WaveParquetSplitReader(
      const std::shared_ptr<connector::ConnectorSplit>& split,
      const SplitReaderParams& params,
      const DefinesMap* defines);

  bool emptySplit() override;

  int32_t canAdvance(WaveStream& stream) override;

  void schedule(WaveStream& stream, int32_t maxRows = 0) override;

  bool isFinished() const override;

  uint64_t getCompletedBytes() override;

  uint64_t getCompletedRows() override;

  std::unordered_map<std::string, RuntimeCounter> runtimeStats() override;

  void configureReaderOptions() override;

  void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStatistics& runtimeStats) override;

  /// Registers a factory that claims Hive splits with PARQUET file format.
  /// Idempotent.
  static void registerSplitReader();

 private:
  // One column of the CPU reader output that is uploaded to the device.
  struct Column {
    // Wave operand receiving the column.
    OperandId operandId;
    // Index of the column in the CPU reader output row.
    int32_t channel;
  };

  // Resolves the reader output columns to Wave operands. Called after the
  // first batch, since the reader output type is final only after
  // prepareSplit().
  void ensureColumns();

  // Reads the next batch from the CPU reader into 'output_' and sets
  // 'pendingRows_'. Sets 'finished_' at end of split.
  void readNextBatch();

  memory::MemoryPool* pool() const {
    return params_.connectorQueryCtx->memoryPool();
  }

  const SplitReaderParams params_;
  const std::shared_ptr<const connector::hive::HiveConnectorSplit> hiveSplit_;

  // Top level column name to the operand defined for it by the Wave
  // TableScan. Copied from the DefinesMap at construction.
  folly::F14FastMap<std::string, const AbstractOperand*> nameToOperand_;

  // Statistics that the CPU reader requires but that WaveSplitReader does not
  // provide.
  const std::shared_ptr<io::IoStatistics> metadataIoStats_;
  const std::shared_ptr<IoStats> ioStats_;

  std::unique_ptr<connector::hive::FileSplitReader> cpuReader_;

  // Columns to upload, ordered by ascending operand id. The order matters:
  // Executable::operandVector() finds the vector for an operand by its ordinal
  // in the OperandSet, and an OperandSet iterates in ascending id order.
  std::vector<Column> columns_;

  // Last batch from the CPU reader. Reused across batches.
  VectorPtr output_;

  // One byte per row null flags for the columns of the current batch. Wave
  // wants a byte per row (kNull/kNotNull) while Velox has a bit per row, so
  // the flags are converted here. Kept alive until the transfer has copied
  // them.
  std::vector<BufferPtr> nullFlags_;

  // Rows of 'output_' that are read but not yet scheduled.
  int32_t pendingRows_{0};

  bool finished_{false};

  // Rows read from the file, including rows dropped by ScanSpec filters.
  int64_t scannedRows_{0};

  int64_t uploadedRows_{0};
  int64_t uploadedBytes_{0};
  int64_t decodeNanos_{0};
  int64_t numBatches_{0};
  int64_t numFlattened_{0};
};

} // namespace facebook::velox::wave
