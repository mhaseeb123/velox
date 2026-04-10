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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// Transforms the deletion bitmap to a surviving (boolean) row mask
/// @param deviceBitmap Deletion bitmap.
/// @param rowMask Boolean row mask indicating surviving rows.
/// @param stream CUDA stream to use for the operation.
/// @param temp_mr Memory resource for temporary allocations.
void convertDeletionBitmapToRowMask(
    cudf::device_span<cudf::bitmask_type const> deviceBitmap,
    cudf::device_span<bool> rowMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr);

/// Scatters deletes to the row mask at positions indicated by `indices`
/// (anti-join semantics).
/// @param rowMask The row mask to scatter deletes to.
/// @param indices The indices to scatter deletes to.
/// @param stream The stream to use for the operation.
/// @param temp_mr Memory resource for temporary allocations.
void scatterDeletesToRowMask(
    cudf::device_span<bool> rowMask,
    cudf::device_span<cudf::size_type const> indices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
