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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDeletionHelpers.h"

#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/scatter.h>
#include <thrust/transform.h>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

/// Functor to check if the bit at `index` is clear. i.e., the row
/// is NOT deleted.
struct IsSurvivingRow {
  const cudf::bitmask_type* bitmask;
  __device__ bool operator()(cudf::size_type index) const noexcept {
    return not cudf::bit_is_set(bitmask, index);
  }
};

} // namespace

void convertDeletionBitmapToRowMask(
    cudf::device_span<cudf::bitmask_type const> deviceBitmap,
    cudf::device_span<bool> rowMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr) {
  // Alternate: Use `cudf::mask_to_bools` but it produces a new column.
  auto iter = cuda::counting_iterator{0};
  thrust::transform(
      rmm::exec_policy_nosync(stream, temp_mr),
      iter,
      iter + rowMask.size(),
      rowMask.begin(),
      IsSurvivingRow{deviceBitmap.data()});
}

void scatterDeletesToRowMask(
    cudf::device_span<bool> rowMask,
    cudf::device_span<cudf::size_type const> indices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr) {
  auto iter = cuda::constant_iterator<bool>(false);
  thrust::scatter(
      rmm::exec_policy_nosync(stream, temp_mr),
      iter,
      iter + indices.size(),
      indices.begin(),
      rowMask.begin());
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
