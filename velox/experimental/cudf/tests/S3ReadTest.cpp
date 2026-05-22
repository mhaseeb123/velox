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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/tests/S3Test.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <thread>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox::exec::test;
namespace {

// Environment variables used by KvikIO/libcurl to reach an S3-compatible
// endpoint. They are exported in SetUp() (for the KvikIO parameter) and
// cleared in TearDown() so the test does not leak state to the rest of the
// process.
constexpr const char* kAwsEnvVars[] = {
    "AWS_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
};

// Block until a TCP connection to `host:port` can be established or the
// timeout expires. KvikIO fails fast on connection-refused (it does not retry
// network errors), so we have to make sure MinIO is actually accepting
// connections before the test issues any S3 request through KvikIO.
bool waitForTcpListening(
    const std::string& host,
    uint16_t port,
    std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      return false;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
      ::close(sock);
      return false;
    }
    const int rc =
        ::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    ::close(sock);
    if (rc == 0) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}

// Parameterized on whether cuDF hive connector uses  BufferedInput (true) or
// falls back to KvikIO (false)
class S3ReadTest : public S3Test,
                   public ::test::VectorTestBase,
                   public ::testing::WithParamInterface<bool> {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    S3Test::SetUp();
    // Register cudf to enable the CudfDatasource creation from
    // CudfHiveConnector
    facebook::velox::cudf_velox::registerCudf();
    filesystems::registerS3FileSystem();

    // Register Hive connector
    auto hiveConfig = minioServer_->hiveConfig({
        {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
             kUseBufferedInput,
         GetParam() ? "true" : "false"},
    });

    // KvikIO source uses libcurl + AWS SigV4, ignore Velox's S3 configuration. Point it at the test MinIO server via standard AWS env vars.
    if (!GetParam()) {
      const auto endpoint =
          hiveConfig->get<std::string>("hive.s3.endpoint").value();
      const auto accessKey =
          hiveConfig->get<std::string>("hive.s3.aws-access-key").value();
      const auto secretKey =
          hiveConfig->get<std::string>("hive.s3.aws-secret-key").value();
      const auto endpointUrl = "http://" + endpoint;
      ::setenv("AWS_ENDPOINT_URL", endpointUrl.c_str(), /*overwrite=*/1);
      ::setenv("AWS_ACCESS_KEY_ID", accessKey.c_str(), /*overwrite=*/1);
      ::setenv("AWS_SECRET_ACCESS_KEY", secretKey.c_str(), /*overwrite=*/1);
      ::setenv("AWS_DEFAULT_REGION", "us-east-1", /*overwrite=*/1);

      // KvikIO does not retry on connection-refused, so block until MinIO is
      // actually accepting connections before any S3 request is issued.
      const auto colon = endpoint.find(':');
      VELOX_CHECK_NE(
          colon, std::string::npos, "Unexpected MinIO endpoint: {}", endpoint);
      const auto host = endpoint.substr(0, colon);
      const auto port = static_cast<uint16_t>(
          std::stoi(endpoint.substr(colon + 1)));
      VELOX_CHECK(
          waitForTcpListening(host, port, std::chrono::seconds(30)),
          "MinIO server at {} did not accept connections within 30s",
          endpoint);
    }

    facebook::velox::cudf_velox::connector::hive::CudfHiveConnectorFactory
        factory;
    auto hiveConnector = factory.newConnector(
        kCudfHiveConnectorId, std::move(hiveConfig), ioExecutor_.get());
    facebook::velox::connector::ConnectorRegistry::global().insert(
        hiveConnector->connectorId(), hiveConnector);
  }

  void TearDown() override {
    filesystems::finalizeS3FileSystem();
    facebook::velox::connector::ConnectorRegistry::global().erase(
        kCudfHiveConnectorId);
    if (!GetParam()) {
      for (const auto* var : kAwsEnvVars) {
        ::unsetenv(var);
      }
    }
    S3Test::TearDown();
  }
};
} // namespace

TEST_P(S3ReadTest, s3ReadTest) {
  const auto sourceFile = test::getDataFilePath(
      "velox/experimental/cudf/tests",
      "../../../dwio/parquet/tests/examples/int.parquet");
  const char* bucketName = "data";
  const auto destinationFile = S3Test::localPath(bucketName) + "/int.parquet";
  minioServer_->addBucket(bucketName);
  std::ifstream src(sourceFile, std::ios::binary);
  std::ofstream dest(destinationFile, std::ios::binary);
  // Copy source file to destination bucket.
  dest << src.rdbuf();
  ASSERT_GT(dest.tellp(), 0) << "Unable to copy from source " << sourceFile;
  dest.close();

  // Read the parquet file via the S3 bucket.
  auto rowType = ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
  auto tableHandle =
      std::make_shared<facebook::velox::connector::hive::HiveTableHandle>(
          kCudfHiveConnectorId,
          "int_table",
          common::SubfieldFilters{},
          nullptr);
  auto plan = PlanBuilder(pool())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();
  auto split = facebook::velox::connector::hive::HiveConnectorSplitBuilder(
                   filesystems::s3URI(bucketName, "int.parquet"))
                   .connectorId(kCudfHiveConnectorId)
                   .fileFormat(dwio::common::FileFormat::PARQUET)
                   .build();

  auto copy = AssertQueryBuilder(plan).split(split).copyResults(pool());

  // expectedResults is the data in int.parquet file.
  const int64_t kExpectedRows = 10;
  auto expectedResults = makeRowVector(
      {makeFlatVector<int32_t>(
           kExpectedRows, [](auto row) { return row + 100; }),
       makeFlatVector<int64_t>(
           kExpectedRows, [](auto row) { return row + 1000; })});
  assertEqualResults({expectedResults}, {copy});
}

INSTANTIATE_TEST_SUITE_P(
    S3ReadTestSuite,
    S3ReadTest,
    ::testing::Values(true, false),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "BufferedInput" : "KvikIO";
    });
