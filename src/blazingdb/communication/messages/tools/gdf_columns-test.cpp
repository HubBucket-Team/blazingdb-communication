#include "gdf_columns.h"

#include <array>

#include <cuda_runtime_api.h>

#include <blazingdb/uc/Context.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {
class GdfColumnFixture {
  using CudaBuffer =
      blazingdb::communication::messages::tools::gdf_columns::CudaBuffer;

public:
  explicit GdfColumnFixture(const void *const       data,
                            const std::size_t       dataSize,
                            const void *const       valid,
                            const std::size_t       validSize,
                            const std::size_t       size,
                            const std::int_fast32_t dtype)
      : data_{CudaBuffer::Make(data, dataSize)},
        valid_{CudaBuffer::Make(valid, validSize)},
        size_{size},
        dtype_{dtype} {}

  const CudaBuffer &
  data() const noexcept {
    return *data_;
  }

  const CudaBuffer &
  valid() const noexcept {
    return *valid_;
  }

  std::size_t
  size() const noexcept {
    return size_;
  }

  std::int_fast32_t
  dtype() const noexcept {
    return dtype_;
  }

private:
  std::unique_ptr<CudaBuffer> data_;
  std::unique_ptr<CudaBuffer> valid_;
  std::size_t                 size_;
  std::int_fast32_t           dtype_;
};
}  // namespace

static inline void *
CreateCudaSequence(const std::size_t size) {
  cudaError_t cudaError;

  void *data;
  cudaError = cudaMalloc(&data, size);
  assert(cudaSuccess == cudaError);

  std::vector<std::uint8_t> host(size);
  std::generate(host.begin(), host.end(), [n = 1]() mutable { return n++; });
  cudaError = cudaMemcpy(data, host.data(), size, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaError);

  cudaError = cudaDeviceSynchronize();
  assert(cudaSuccess == cudaError);

  return data;
}

static inline GdfColumnFixture
CreateBasicGdfColumnFixture() {
  const std::size_t dataSize = 2000;
  const void *const data     = CreateCudaSequence(dataSize);

  std::size_t validSize = 1000;
  void *      valid     = CreateCudaSequence(validSize);

  const std::size_t size = 500;

  const std::int_fast32_t dtype = 3;

  return GdfColumnFixture{data, dataSize, valid, validSize, size, dtype};
}

// Tests for gdf column builder

class MockAgent : public blazingdb::uc::Agent {
public:
  using Buffer = blazingdb::uc::Buffer;

  std::unique_ptr<Buffer>
  Register(const void *&data, std::size_t size) const noexcept final {
    MockRegister(data, size);
  }

  MOCK_CONST_METHOD2(MockRegister,
                     std::unique_ptr<Buffer>(const void *&, std::size_t));
};

TEST(GdfColumnBuilderTest, CheckPayload) {
  auto fixture = CreateBasicGdfColumnFixture();

  std::unique_ptr<blazingdb::uc::Context> context =
      blazingdb::uc::Context::IPC();
  // TODO: use mock instead this
  std::unique_ptr<blazingdb::uc::Agent> agent = context->Agent();

  using blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnBuilder;
  auto builder = GdfColumnBuilder::MakeWithHostAllocation(*agent);

  auto payload = builder->Data(fixture.data())
                     .Valid(fixture.valid())
                     .Size(fixture.size())
                     .DType(fixture.dtype())
                     .Build();

  auto &buffer = payload->Deliver();

  // TODO: do something with buffer
}

// Tests for gdf column collection builder

class MockPayload
    : public blazingdb::communication::messages::tools::gdf_columns::Payload {
public:
  using Buffer = blazingdb::communication::messages::tools::gdf_columns::Buffer;

  const Buffer &
  Deliver() const noexcept final {
    return MockDeliver();
  }

  MOCK_CONST_METHOD0(MockDeliver, const Buffer &());
};

class MockBuffer
    : public blazingdb::communication::messages::tools::gdf_columns::Buffer {
public:
  const void *
  Data() const noexcept final {
    return DataMember();
  }

  std::size_t
  Size() const noexcept final {
    return SizeMember();
  }

  MOCK_CONST_METHOD0(DataMember, const void *());
  MOCK_CONST_METHOD0(SizeMember, std::size_t());
};

static inline void
ExpectCall(MockBuffer &mock, const std::string &&content) {
  EXPECT_CALL(mock, DataMember)
      .WillRepeatedly(::testing::Return(content.c_str()));
  EXPECT_CALL(mock, SizeMember)
      .WillRepeatedly(::testing::Return(content.length()));
}

static inline void
ExpectCall(MockPayload &mock, const MockBuffer &buffer) {
  EXPECT_CALL(mock, MockDeliver).WillOnce(::testing::ReturnRef(buffer));
}

TEST(GdfColumnCollectorTest, AddPayloads) {
  using blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnCollector;
  auto collector = GdfColumnCollector::MakeInHost();

  MockPayload payload1, payload2, payload3;
  MockBuffer  buffer1, buffer2, buffer3;

  ExpectCall(buffer1, "11111");
  ExpectCall(buffer2, "22222");
  ExpectCall(buffer3, "33333");

  ExpectCall(payload1, buffer1);
  ExpectCall(payload2, buffer2);
  ExpectCall(payload3, buffer3);

  collector->Add(payload1);
  collector->Add(payload2);
  collector->Add(payload3);

  EXPECT_EQ(3, collector->Length());

  auto buffer = collector->Collect();
}
