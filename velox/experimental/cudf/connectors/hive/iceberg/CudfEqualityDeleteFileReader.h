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

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <folly/Executor.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_connector = ::facebook::velox::connector;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;
namespace velox_hive = ::facebook::velox::connector::hive;

/// Reads an Iceberg equality delete file and filters base data rows on GPU
/// whose equality delete column values match any row in the delete file.
///
/// Iceberg equality delete files contain rows with values for one or more
/// columns (identified by equalityFieldIds). A base data row is deleted if
/// its values match ALL specified columns of ANY row in the delete file.
///
/// The delete file is read eagerly on CPU during construction and stored as
/// a Velox RowVector. On the first call to applyDeletes(), the delete rows
/// are lazily converted to a GPU-resident cudf table. Subsequent calls to
/// applyDeletes() reuse the cached cudf table. The actual row matching is
/// performed by cudf::detail::contains which builds a cuco::static_set
/// internally.
class CudfEqualityDeleteFileReader {
 public:
  /// Constructs a reader for a single equality delete file.
  ///
  /// Eagerly reads the entire delete file into a Velox RowVector. The delete
  /// file is fully consumed during construction.
  ///
  /// @param deleteFile Metadata about the equality delete file. Must have
  ///   content == FileContent::kEqualityDeletes and non-empty
  ///   equalityFieldIds.
  /// @param equalityColumnNames Ordered column names corresponding to
  ///   equalityFieldIds, resolved by the caller from the table schema.
  /// @param equalityColumnTypes Ordered column types corresponding to
  ///   equalityFieldIds.
  /// @param baseFilePath Path of the base data file being read.
  /// @param fileHandleFactory Factory for creating file handles.
  /// @param connectorQueryCtx Query context for memory and config.
  /// @param executor IO executor for async reads.
  /// @param hiveConfig Hive configuration.
  /// @param ioStatistics IO statistics collector.
  /// @param ioStats IO stats tracker.
  /// @param runtimeStats Runtime statistics for recording skipped bytes.
  /// @param connectorId Connector identifier.
  CudfEqualityDeleteFileReader(
      const velox_iceberg::IcebergDeleteFile& deleteFile,
      const std::vector<std::string>& equalityColumnNames,
      const std::vector<TypePtr>& equalityColumnTypes,
      const std::string& baseFilePath,
      FileHandleFactory* fileHandleFactory,
      const velox_connector::ConnectorQueryCtx* connectorQueryCtx,
      folly::Executor* executor,
      const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig,
      const std::shared_ptr<::facebook::velox::io::IoStatistics>& ioStatistics,
      const std::shared_ptr<::facebook::velox::IoStats>& ioStats,
      ::facebook::velox::dwio::common::RuntimeStatistics& runtimeStats,
      const std::string& connectorId);

  /// Applies equality deletes to the output CudfVector by clearing the
  /// row mask for rows whose equality column values match any delete key tuple.
  ///
  /// On the first call, lazily converts the Velox delete rows to a cudf
  /// table on GPU.
  ///
  /// @param table Base cudf table view to filter.
  /// @param inputColumnNames Column names of the input table, used to
  ///   resolve equality column indices.
  /// @param rowMask Device buffer of booleans; surviving rows are true.
  /// @param stream CUDA stream for kernel launches.
  /// @param mr Device memory resource for temporary allocations.
  void applyDeletes(
      cudf::table_view table,
      const std::vector<std::string>& inputColumnNames,
      std::shared_ptr<rmm::device_buffer> rowMask,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  /// Returns the number of delete key tuples loaded from the file.
  size_t numDeleteKeys() const {
    return numDeleteKeys_;
  }

  /// Returns true if this reader has no delete keys (file was skipped or
  /// empty). When true, applyDeletes() is a no-op.
  bool empty() const {
    return numDeleteKeys_ == 0;
  }

 private:
  /// Lazily converts the deleteRows_ to a cudf table at the first
  /// `applyDeletes` call
  void buildDeleteKeyTable(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  /// Lazily constructs the equality column indices in the input table
  /// on the first call to applyDeletes().
  void buildEqualityColumnIndices(
      const std::vector<std::string>& inputColumnNames);

  /// Column names and types for equality delete comparison.
  std::vector<std::string> equalityColumnNames_;

  /// Number of delete key tuples loaded from the file.
  size_t numDeleteKeys_;

  /// All rows read from the equality delete file, stored for equality
  /// comparison during probing.
  RowVectorPtr deleteRows_;

  /// Cudf table containing the delete key tuples.
  std::unique_ptr<cudf::table> deleteKeyTable_;

  /// Equality column indices in the input table
  std::vector<cudf::size_type> equalityColumnIndices_;

  /// Whether the delete key table and equality column indices have been loaded.
  bool isLoaded_{false};

  memory::MemoryPool* pool_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
