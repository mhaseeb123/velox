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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReader.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDeletionHelpers.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergSplitReader.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/dwio/common/BufferUtil.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <folly/lang/Bits.h>

#include <cstring>
#include <unordered_set>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_hive = ::facebook::velox::connector::hive;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

namespace {

/// Returns true if a delete/update file should be skipped based on sequence
/// number conflict resolution. Per the Iceberg spec (V2+):
///   - Equality deletes apply when deleteSeqNum > dataSeqNum (i.e., skip when
///     deleteSeqNum <= dataSeqNum).
///   - Positional deletes, deletion vectors, and positional updates apply when
///     deleteSeqNum >= dataSeqNum (i.e., skip when deleteSeqNum < dataSeqNum),
///     because same-snapshot positional deletes SHOULD apply.
///   - A sequence number of 0 means "unassigned" (legacy V1 tables) and
///     disables filtering (never skip).
bool shouldSkipBySequenceNumber(
    int64_t fileSeqNum,
    int64_t dataSeqNum,
    bool isEqualityDelete) {
  if (fileSeqNum <= 0 || dataSeqNum <= 0) {
    return false;
  }
  return isEqualityDelete ? (fileSeqNum <= dataSeqNum)
                          : (fileSeqNum < dataSeqNum);
}

} // namespace

CudfIcebergSplitReader::CudfIcebergSplitReader(
    std::shared_ptr<CudfHiveConnectorSplit> split,
    std::shared_ptr<const velox_iceberg::HiveIcebergSplit> icebergSplit,
    std::shared_ptr<const velox_hive::HiveTableHandle> tableHandle,
    const RowTypePtr& outputType,
    const std::vector<std::string>& readColumnNames,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ::facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    bool useExperimentalCudfReader,
    cudf::ast::expression const* subfieldFilterExpr,
    std::unique_ptr<exec::ExprSet>* remainingFilterExprSet,
    std::shared_ptr<CudfExpression> cudfExpressionEvaluator,
    std::atomic<uint64_t>* totalRemainingFilterTime)
    : CudfSplitReader(
          std::move(split),
          std::move(tableHandle),
          outputType,
          readColumnNames,
          fileHandleFactory,
          executor,
          connectorQueryCtx,
          cudfHiveConfig,
          ioStatistics,
          ioStats,
          useExperimentalCudfReader,
          subfieldFilterExpr,
          remainingFilterExprSet,
          std::move(cudfExpressionEvaluator),
          totalRemainingFilterTime),
      icebergSplit_(std::move(icebergSplit)),
      hiveConfig_(hiveConfig) {}

void CudfIcebergSplitReader::prepareSplit() {
  deletionVectorReader_.reset();
  positionalDeleteFileReaders_.clear();
  equalityDeleteFileReaders_.clear();
  extraEqualityColumns_.clear();
  injectedColumns_.clear();
  baseReadOffset_ = 0;

  // Note: Must setup delete file readers before calling base `prepareSplit` so
  // that it can correctly determine the memory resource to construct the cuDF
  // reader.
  setupDeleteFileReaders();

  // Setup column projection to include any equality delete key columns that
  // are not already in the output projection. Must be called before base
  // `prepareSplit`
  setupColumnProjection();

  // Detect partition columns and schema-evolution missing columns.
  // Filters partition columns from readColumnNames_ so the parquet reader
  // doesn't try to read them, and records them for post-read injection.
  setupSchemaReconciliation();

  // Call base to setup stream, datasource, options, and the cudf reader
  CudfSplitReader::prepareSplit();
}

rmm::device_async_resource_ref
CudfIcebergSplitReader::determineCudfMemoryResource() {
  // If we will be applying any deletes, use temporary mr to read table chunks
  // from cuDF readers. Otherwise, use the output mr.
  return (deletionVectorReader_ or positionalDeleteFileReaders_.size() or
          equalityDeleteFileReaders_.size())
      ? get_temp_mr()
      : get_output_mr();
}

void CudfIcebergSplitReader::createCudfReader(
    rmm::device_async_resource_ref output_mr) {
  CudfSplitReader::createCudfReader(determineCudfMemoryResource());
}

std::optional<std::unique_ptr<cudf::table>>
CudfIcebergSplitReader::readNextChunk(
    rmm::device_async_resource_ref output_mr) {
  // Determine the memory resource to use for `readNextChunk`
  auto mr = determineCudfMemoryResource();

  // Read the next table chunk from the cuDF reader
  auto cudfTable = CudfSplitReader::readNextChunk(mr).value_or(nullptr);
  if (not cudfTable) {
    return std::nullopt;
  }

  // Number of table rows read by cuDF (before any deletes)
  const auto numRows = cudfTable->num_rows();

  // Determine if we are applying any deletes
  const auto isApplyingDeletes = numRows > 0 and
      (deletionVectorReader_ or positionalDeleteFileReaders_.size() or
       equalityDeleteFileReaders_.size());

  if (isApplyingDeletes) {
    // Allocate row mask column if needed
    if (not rowMask_ or rowMask_->size() < numRows) {
      auto true_scalar =
          cudf::numeric_scalar<bool>(true, true, stream_, get_temp_mr());
      rowMask_ = cudf::make_column_from_scalar(
          true_scalar, numRows, stream_, get_temp_mr());
    }

    // Apply deletion vector
    if (deletionVectorReader_) {
      applyDeletionVector(cudfTable->view());
    }
    // Apply positional deletes
    if (positionalDeleteFileReaders_.size()) {
      applyPositionalDeletes(cudfTable->view());
    }
    // Apply equality deletes
    if (equalityDeleteFileReaders_.size()) {
      applyEqualityDeletes(cudfTable->view());
    }

    cudfTable = cudf::apply_boolean_mask(
        cudfTable->view(), rowMask_->view(), stream_, get_output_mr());
  }

  // Inject partition columns and schema-evolution NULL columns.
  // Missing columns were detected during setupSchemaReconciliation().
  // This must run even for 0-row tables so post-delete empty chunks still
  // have the expected number and order of output columns.
  if (not injectedColumns_.empty()) {
    cudfTable = injectMissingColumns(std::move(cudfTable), output_mr);
  }

  // Strip any extra equality delete key columns that were added
  if (not extraEqualityColumns_.empty()) {
    VELOX_CHECK_EQ(
        extraEqualityColumns_.size(),
        cudfTable->num_columns() - outputType_->size(),
        "Unexpected number of extra equality delete key columns: {}",
        extraEqualityColumns_.size());
    auto columns = cudfTable->release();
    columns.resize(outputType_->size());
    cudfTable = std::make_unique<cudf::table>(std::move(columns));
  }

  // Update the base read offset
  baseReadOffset_ += numRows;

  return cudfTable;
}

void CudfIcebergSplitReader::setupDeleteFileReaders() {
  // TODO(mh): We currently read a data files as a single split.
  constexpr uint64_t splitOffset = 0;

  for (const auto& deleteFile : icebergSplit_->deleteFiles) {
    if (deleteFile.content == velox_iceberg::FileContent::kPositionalDeletes) {
      if (deleteFile.recordCount == 0) {
        continue;
      }
      if (shouldSkipBySequenceNumber(
              deleteFile.dataSequenceNumber,
              icebergSplit_->dataSequenceNumber,
              /*isEqualityDelete=*/false)) {
        continue;
      }

      // Skip the delete file if all delete positions are before this split.
      // TODO: Skip delete files where all positions are after the split, if
      // split row count becomes available.
      if (auto iter = deleteFile.upperBounds.find(
              velox_iceberg::IcebergMetadataColumn::kPosId);
          iter != deleteFile.upperBounds.end()) {
        auto decodedBound = encoding::Base64::decode(iter->second);
        VELOX_CHECK_EQ(
            decodedBound.size(),
            sizeof(uint64_t),
            "Unexpected decoded size for positional delete upper bound.");
        uint64_t posDeleteUpperBound;
        std::memcpy(
            &posDeleteUpperBound, decodedBound.data(), sizeof(uint64_t));
        posDeleteUpperBound = folly::Endian::little(posDeleteUpperBound);
        if (posDeleteUpperBound < splitOffset) {
          continue;
        }
      }

      positionalDeleteFileReaders_.push_back(
          std::make_unique<velox_iceberg::PositionalDeleteFileReader>(
              deleteFile,
              icebergSplit_->filePath,
              fileHandleFactory_,
              connectorQueryCtx_,
              executor_,
              hiveConfig_,
              ioStatistics_,
              ioStats_,
              runtimeStats_,
              splitOffset,
              icebergSplit_->connectorId));
    } else if (
        deleteFile.content == velox_iceberg::FileContent::kEqualityDeletes) {
      if (deleteFile.recordCount == 0 || deleteFile.equalityFieldIds.empty()) {
        continue;
      }
      if (shouldSkipBySequenceNumber(
              deleteFile.dataSequenceNumber,
              icebergSplit_->dataSequenceNumber,
              /*isEqualityDelete=*/true)) {
        continue;
      }

      // Resolve equalityFieldIds to column names and types. In Iceberg,
      // field IDs for top-level columns are assigned sequentially starting
      // from 1, matching the column order in the table schema.
      std::vector<std::string> equalityColumnNames;
      std::vector<TypePtr> equalityColumnTypes;

      const auto& dataColumns = tableHandle_->dataColumns();
      if (dataColumns) {
        for (const auto& eqFieldId : deleteFile.equalityFieldIds) {
          auto colIdx = static_cast<uint32_t>(eqFieldId - 1);
          VELOX_CHECK_LT(
              colIdx,
              dataColumns->size(),
              "Equality delete field ID out of range: {}",
              eqFieldId);
          equalityColumnNames.push_back(dataColumns->nameOf(colIdx));
          equalityColumnTypes.push_back(dataColumns->childAt(colIdx));
        }
      }

      if (!equalityColumnNames.empty()) {
        equalityDeleteFileReaders_.push_back(
            std::make_unique<CudfEqualityDeleteFileReader>(
                deleteFile,
                equalityColumnNames,
                equalityColumnTypes,
                icebergSplit_->filePath,
                fileHandleFactory_,
                connectorQueryCtx_,
                executor_,
                hiveConfig_,
                ioStatistics_,
                ioStats_,
                runtimeStats_,
                icebergSplit_->connectorId));
      }
    } else if (
        deleteFile.content == velox_iceberg::FileContent::kDeletionVector) {
      if (deleteFile.recordCount == 0) {
        continue;
      }
      if (shouldSkipBySequenceNumber(
              deleteFile.dataSequenceNumber,
              icebergSplit_->dataSequenceNumber,
              /*isEqualityDelete=*/false)) {
        continue;
      }
      deletionVectorReader_ =
          std::make_unique<CudfDeletionVectorReader>(deleteFile, splitOffset);
    } else {
      VELOX_NYI(
          "Unsupported delete file content type: {}",
          static_cast<int>(deleteFile.content));
    }
  }
}

void CudfIcebergSplitReader::applyDeletionVector(cudf::table_view input) {
  // Apply deletion vector into the rowMask_
  const auto numRows = input.num_rows();
  deletionVectorReader_->applyDeletes(
      rowMask_->mutable_view(),
      baseReadOffset_,
      numRows,
      stream_,
      get_temp_mr());

  // Reset the deletion vector reader if we have read the entire bitmap.
  if (deletionVectorReader_->noMoreData()) {
    deletionVectorReader_.reset();
  }
}

void CudfIcebergSplitReader::applyPositionalDeletes(cudf::table_view input) {
  // Apply positional deletes into the rowMask_
  const auto numRows = input.num_rows();

  // Initialize host and device delete bitmaps
  const auto numWords = cudf::num_bitmask_words(numRows);
  const auto numBitmaskBytes = numWords * sizeof(cudf::bitmask_type);
  dwio::common::ensureCapacity<int8_t>(
      deleteBitmap_,
      numBitmaskBytes,
      connectorQueryCtx_->memoryPool(),
      false,
      true);
  if (not deviceDeleteBitmap_ or
      deviceDeleteBitmap_->size() < numBitmaskBytes) {
    deviceDeleteBitmap_ = std::make_shared<rmm::device_buffer>(
        numBitmaskBytes, stream_, get_temp_mr());
  }

  VELOX_CHECK_NOT_NULL(deleteBitmap_->as<uint8_t>());
  VELOX_CHECK_GE(deleteBitmap_->size(), numBitmaskBytes);
  VELOX_CHECK_NOT_NULL(deviceDeleteBitmap_->data());
  VELOX_CHECK_GE(deviceDeleteBitmap_->size(), numBitmaskBytes);

  for (auto iter = positionalDeleteFileReaders_.begin();
       iter != positionalDeleteFileReaders_.end();) {
    (*iter)->readDeletePositions(baseReadOffset_, numRows, deleteBitmap_);
    if ((*iter)->noMoreData()) {
      iter = positionalDeleteFileReaders_.erase(iter);
    } else {
      ++iter;
    }
  }

  // Copy the deletion bitmap to device
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      deviceDeleteBitmap_->data(),
      deleteBitmap_->as<uint8_t>(),
      numBitmaskBytes,
      cudaMemcpyHostToDevice,
      stream_.value()));

  // Apply the deletion bitmap to the row mask
  applyDeletionBitmapToRowMask(
      cudf::device_span<cudf::bitmask_type>(
          static_cast<cudf::bitmask_type*>(deviceDeleteBitmap_->data()),
          numWords),
      rowMask_->mutable_view(),
      stream_,
      get_temp_mr());
}

void CudfIcebergSplitReader::applyEqualityDeletes(cudf::table_view input) {
  // Reset the row mask to all-true to start
  const auto numRows = input.num_rows();

  // Iteratively apply equality deletes, if any
  for (auto& reader : equalityDeleteFileReaders_) {
    reader->applyDeletes(
        input, readColumnNames_, rowMask_->mutable_view(), stream_);
  }
}

void CudfIcebergSplitReader::setupColumnProjection() {
  if (equalityDeleteFileReaders_.empty()) {
    return;
  }

  std::unordered_set<std::string> readColumnSet(
      readColumnNames_.begin(), readColumnNames_.end());
  const auto& dataColumns = tableHandle_->dataColumns();

  // For each equality delete file, find and append any columns that are not
  // already in the readColumnSet
  std::for_each(
      icebergSplit_->deleteFiles.begin(),
      icebergSplit_->deleteFiles.end(),
      [&](const auto& deleteFile) {
        if (deleteFile.content !=
                velox_iceberg::FileContent::kEqualityDeletes or
            deleteFile.equalityFieldIds.empty()) {
          return;
        }
        std::for_each(
            deleteFile.equalityFieldIds.begin(),
            deleteFile.equalityFieldIds.end(),
            [&](const auto& equalityFieldId) {
              const auto columnIdx = static_cast<uint32_t>(equalityFieldId - 1);
              if (dataColumns and columnIdx < dataColumns->size()) {
                const auto& columnName = dataColumns->nameOf(columnIdx);
                // Insert column name into readColumnSet if not already present
                if (readColumnSet.insert(columnName).second) {
                  extraEqualityColumns_.push_back(columnName);
                }
              }
            });
      });

  // Append extra columns to readColumnNames_ so the Parquet reader fetches
  // them.
  readColumnNames_.insert(
      readColumnNames_.end(),
      extraEqualityColumns_.begin(),
      extraEqualityColumns_.end());
}

void CudfIcebergSplitReader::setupSchemaReconciliation() {
  // Identify partition columns and schema-evolution missing columns.
  // Partition columns come from the split metadata; missing columns are
  // detected by reading the parquet file's schema (footer only).

  // 1. Detect partition columns
  for (size_t i = 0; i < outputType_->size(); ++i) {
    const auto& name = outputType_->nameOf(i);
    auto partIt = icebergSplit_->partitionKeys.find(name);
    if (partIt != icebergSplit_->partitionKeys.end()) {
      injectedColumns_.push_back(
          {i, name, partIt->second, outputType_->childAt(i)});
    }
  }

  // 2. Detect schema-evolution missing columns by reading the parquet
  //    file's schema. Read 0 rows to get just the column names from the
  //    footer without reading any data.
  {
    std::unordered_set<std::string> injectedNames;
    for (const auto& col : injectedColumns_) {
      injectedNames.insert(col.name);
    }

    // TODO(ducndh): This bypasses the BufferedInput/token-provider path used
    // by the main reader when useBufferedInputSession() is enabled. It will
    // fail on non-local filesystems (S3/HDFS). When adding remote filesystem
    // support, use the same datasource construction as the main reader.
    auto opts = cudf::io::parquet_reader_options::builder(
                    cudf::io::source_info{split_->filePath})
                    .num_rows(0)
                    .build();
    auto meta = cudf::io::read_parquet(opts, stream_, get_temp_mr());
    std::unordered_set<std::string> fileColumns;
    for (const auto& si : meta.metadata.schema_info) {
      fileColumns.insert(si.name);
    }

    for (size_t i = 0; i < outputType_->size(); ++i) {
      const auto& name = outputType_->nameOf(i);
      if (injectedNames.count(name) > 0) {
        continue; // Already handled as partition column
      }
      if (fileColumns.count(name) == 0) {
        // Column not in parquet file — schema evolution: inject NULL
        injectedColumns_.push_back(
            {i, name, std::nullopt, outputType_->childAt(i)});
        injectedNames.insert(name);
      }
    }
  }

  // 3. Remove all injected columns from readColumnNames_
  if (not injectedColumns_.empty()) {
    std::unordered_set<std::string> injectedNames;
    for (const auto& col : injectedColumns_) {
      injectedNames.insert(col.name);
    }
    std::vector<std::string> filtered;
    filtered.reserve(readColumnNames_.size());
    for (const auto& name : readColumnNames_) {
      if (injectedNames.count(name) == 0) {
        filtered.push_back(name);
      }
    }
    readColumnNames_ = std::move(filtered);
  }
}

std::unique_ptr<cudf::table> CudfIcebergSplitReader::injectMissingColumns(
    std::unique_ptr<cudf::table> table,
    rmm::device_async_resource_ref mr) {
  const auto numRows = table->num_rows();
  auto columns = table->release();

  // Total output columns = data columns from parquet + injected columns +
  // extra equality columns
  const auto totalColumns = columns.size() + injectedColumns_.size();
  std::vector<std::unique_ptr<cudf::column>> output;
  output.reserve(totalColumns);

  // Merge data columns and injected columns in the correct output order.
  // injectedColumns_ stores the output index where each injected column goes.
  size_t dataIdx = 0;
  size_t injIdx = 0;

  // Sort injected columns by output index for sequential insertion
  auto sorted = injectedColumns_;
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.outputIndex < b.outputIndex;
  });

  for (size_t outIdx = 0; outIdx < totalColumns; ++outIdx) {
    if (injIdx < sorted.size() && sorted[injIdx].outputIndex == outIdx) {
      // Inject a constant column
      auto cudfType = cudf_velox::veloxToCudfDataType(sorted[injIdx].veloxType);
      if (sorted[injIdx].partitionValue.has_value()) {
        // Partition column: create constant with the partition value.
        const auto& value = sorted[injIdx].partitionValue.value();
        std::unique_ptr<cudf::scalar> scalar;
        const auto& colName = sorted[injIdx].name;
        try {
          if (cudfType.id() == cudf::type_id::STRING) {
            scalar =
                std::make_unique<cudf::string_scalar>(value, true, stream_, mr);
          } else if (cudfType.id() == cudf::type_id::INT64) {
            scalar = std::make_unique<cudf::numeric_scalar<int64_t>>(
                std::stoll(value), true, stream_, mr);
          } else if (cudfType.id() == cudf::type_id::INT32) {
            scalar = std::make_unique<cudf::numeric_scalar<int32_t>>(
                std::stoi(value), true, stream_, mr);
          } else {
            VELOX_FAIL(
                "Unsupported partition column type for constant injection: "
                "column '{}', type {}",
                colName,
                sorted[injIdx].veloxType->toString());
          }
        } catch (const std::invalid_argument& e) {
          VELOX_FAIL(
              "Invalid partition value for column '{}' (type {}): '{}'",
              colName,
              sorted[injIdx].veloxType->toString(),
              value);
        } catch (const std::out_of_range& e) {
          VELOX_FAIL(
              "Partition value out of range for column '{}' (type {}): '{}'",
              colName,
              sorted[injIdx].veloxType->toString(),
              value);
        }
        output.push_back(
            cudf::make_column_from_scalar(*scalar, numRows, stream_, mr));
      } else {
        // Schema evolution: create all-NULL column
        auto scalar =
            cudf::make_default_constructed_scalar(cudfType, stream_, mr);
        scalar->set_valid_async(false, stream_);
        output.push_back(
            cudf::make_column_from_scalar(*scalar, numRows, stream_, mr));
      }
      ++injIdx;
    } else {
      // Data column from parquet
      VELOX_CHECK_LT(
          dataIdx,
          columns.size(),
          "Data column index out of range during schema reconciliation");
      output.push_back(std::move(columns[dataIdx++]));
    }
  }

  // Append any remaining data columns (extra equality columns)
  while (dataIdx < columns.size()) {
    output.push_back(std::move(columns[dataIdx++]));
  }

  return std::make_unique<cudf::table>(std::move(output));
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
