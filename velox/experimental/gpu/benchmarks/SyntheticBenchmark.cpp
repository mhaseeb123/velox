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

#include <folly/json.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace facebook::velox::gpu::benchmark {
namespace {

constexpr int64_t kDateCardinality = 2'500;
constexpr int64_t kValueCardinality = 1'000;

uint64_t mix(uint64_t hash, uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return hash ^ (value + (hash << 6) + (hash >> 2));
}

std::vector<RowVectorPtr> makeFact(
    test::VectorMaker& maker,
    const Options& options) {
  std::vector<RowVectorPtr> batches;
  for (int64_t begin = 0; begin < options.rows; begin += options.batchRows) {
    const auto size = static_cast<vector_size_t>(
        std::min<int64_t>(options.batchRows, options.rows - begin));
    auto make = [&](auto fn) {
      return maker.flatVector<int64_t>(
          size, [begin, fn](vector_size_t row) { return fn(begin + row); });
    };
    batches.push_back(maker.rowVector(
        {"key",
         "group_key",
         "value",
         "quantity",
         "discount",
         "shipdate",
         "extendedprice",
         "returnflag",
         "linestatus",
         "orderkey"},
        {make([](auto row) { return row; }),
         make([&](auto row) { return row % options.groupCardinality; }),
         make([](auto row) { return (row * 17 + 13) % kValueCardinality; }),
         make([](auto row) { return 1 + (row * 7) % 50; }),
         make([](auto row) { return (row * 3) % 11; }),
         make([](auto row) { return row % kDateCardinality; }),
         make([](auto row) { return 100 + (row * 19) % 100'000; }),
         make([](auto row) { return row % 3; }),
         make([](auto row) { return row % 2; }),
         make([](auto row) { return row / 4; })}));
  }
  return batches;
}

std::vector<RowVectorPtr> makeOrders(
    test::VectorMaker& maker,
    const Options& options) {
  const int64_t numOrders = std::max<int64_t>(1, (options.rows + 3) / 4);
  const auto size = static_cast<vector_size_t>(numOrders);
  auto make = [&](auto fn) {
    return maker.flatVector<int64_t>(
        size, [fn](vector_size_t row) { return fn(row); });
  };
  return {maker.rowVector(
      {"o_orderkey", "orderdate", "shippriority", "build_value"},
      {make([](auto row) { return row; }),
       make([](auto row) { return row % kDateCardinality; }),
       make([](auto row) { return row % 5; }),
       make([](auto row) { return 1 + (row * 23) % 10'000; })})};
}

} // namespace

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
    Workload workload) {
  if (workload != Workload::kJoinAggregate && workload != Workload::kQ3) {
    return makeData(pool, options);
  }
  VELOX_USER_CHECK_LE(options.rows, std::numeric_limits<vector_size_t>::max());
  auto joinOptions = options;
  joinOptions.batchRows = static_cast<int32_t>(options.rows);
  return makeData(pool, joinOptions);
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

std::string resultJson(const Result& result) {
  folly::dynamic samples = folly::dynamic::array;
  for (auto sample : result.warmMs) {
    samples.push_back(sample);
  }
  const auto median = percentile(result.warmMs, 0.5);
  const auto p95 = percentile(result.warmMs, 0.95);
  folly::dynamic json = folly::dynamic::object("backend", result.backend)(
      "workload", result.workload)("rows", result.rows)(
      "input_bytes", result.inputBytes)("filter_percent", result.filterPercent)(
      "group_cardinality", result.groupCardinality)("warmups", result.warmups)(
      "repetitions", result.repetitions)(
      "checksum", fmt::format("{:016x}", result.checksum))(
      "expected_checksum", fmt::format("{:016x}", result.expectedChecksum))(
      "correct", result.checksum == result.expectedChecksum)(
      "cold_ms", result.coldMs)("warm_median_ms", median)("warm_p95_ms", p95)(
      "rows_per_second", result.rows * 1'000.0 / median)(
      "input_gb_per_second", result.inputBytes / median / 1'000'000.0)(
      "warm_samples_ms", std::move(samples));
  return folly::toJson(json);
}

std::string csvHeader() {
  return "backend,workload,rows,input_bytes,filter_percent,group_cardinality,"
         "warmups,repetitions,checksum,expected_checksum,correct,cold_ms,"
         "warm_median_ms,warm_p95_ms,rows_per_second,input_gb_per_second";
}

std::string resultCsv(const Result& result) {
  const auto median = percentile(result.warmMs, 0.5);
  const auto p95 = percentile(result.warmMs, 0.95);
  return fmt::format(
      "{},{},{},{},{},{},{},{},{:016x},{:016x},{},{:.6f},{:.6f},{:.6f},"
      "{:.3f},{:.6f}",
      result.backend,
      result.workload,
      result.rows,
      result.inputBytes,
      result.filterPercent,
      result.groupCardinality,
      result.warmups,
      result.repetitions,
      result.checksum,
      result.expectedChecksum,
      result.checksum == result.expectedChecksum,
      result.coldMs,
      median,
      p95,
      result.rows * 1'000.0 / median,
      result.inputBytes / median / 1'000'000.0);
}

} // namespace facebook::velox::gpu::benchmark
