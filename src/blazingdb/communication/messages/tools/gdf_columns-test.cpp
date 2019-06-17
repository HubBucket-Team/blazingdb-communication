#include "gdf_columns.h"

#include <array>
#include <cstring>
#include <numeric>

#include <cuda_runtime_api.h>

#include <blazingdb/uc/Context.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../GpuComponentMessage.h"

namespace {
class GdfColumnFixture {
  using CudaBuffer =
      blazingdb::communication::messages::tools::gdf_columns::CudaBuffer;

  using HostBuffer =
      blazingdb::communication::messages::tools::gdf_columns::HostBuffer;

public:
  explicit GdfColumnFixture(const void *const       data,
                            const std::size_t       dataSize,
                            const void *const       valid,
                            const std::size_t       validSize,
                            const std::size_t       size,
                            const std::int_fast32_t dtype,
                            const std::size_t       nullCount,
                            const std::string       columnName)
      : data_{CudaBuffer::Make(data, dataSize)},
        valid_{CudaBuffer::Make(valid, validSize)},
        size_{size},
        dtype_{dtype},
        nullCount_{nullCount},
        columnName_{HostBuffer::Make(columnName.data(), columnName.size())} {}

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

  std::size_t
  nullCount() const noexcept {
    return nullCount_;
  }

  const HostBuffer &
  columnName() const noexcept {
    return *columnName_;
  }

private:
  std::unique_ptr<CudaBuffer> data_;
  std::unique_ptr<CudaBuffer> valid_;
  std::size_t                 size_;
  std::int_fast32_t           dtype_;
  std::size_t                 nullCount_;
  std::unique_ptr<HostBuffer> columnName_;
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

  const std::size_t nullCount = 5;

  const std::string columnName = "ColName";

  return GdfColumnFixture{
      data, dataSize, valid, validSize, size, dtype, nullCount, columnName};
}

// Tests for gdf column builder

class MockBUCAgent : public blazingdb::uc::Agent {
public:
  using Buffer = blazingdb::uc::Buffer;

  std::unique_ptr<Buffer>
  Register(const void *&data, std::size_t size) const noexcept final {
    return RegisterMember(data, size);
  }

  MOCK_CONST_METHOD2(RegisterMember,
                     std::unique_ptr<Buffer>(const void *&, std::size_t));
};

class MockBUCBuffer : public blazingdb::uc::Buffer {
public:
  using Transport  = blazingdb::uc::Transport;
  using Serialized = blazingdb::uc::Record::Serialized;

  MOCK_CONST_METHOD1(Link, std::unique_ptr<Transport>(Buffer *));
  MOCK_CONST_METHOD0(SerializedRecordMember,
                     std::unique_ptr<const Serialized>());
  MOCK_METHOD1(Link, std::unique_ptr<Transport>(const std::uint8_t *));

  std::unique_ptr<const Serialized>
  SerializedRecord() const noexcept final {
    return SerializedRecordMember();
  }
};

class MockBUCSerialized : public blazingdb::uc::Record::Serialized {
public:
  const std::uint8_t *
  Data() const noexcept final {
    return DataMember();
  }

  std::size_t
  Size() const noexcept final {
    return SizeMember();
  }

  MOCK_CONST_METHOD0(DataMember, const std::uint8_t *());
  MOCK_CONST_METHOD0(SizeMember, std::size_t());
};

TEST(GdfColumnBuilderTest, CheckPayload) {
  auto fixture = CreateBasicGdfColumnFixture();

  MockBUCAgent agent;
  EXPECT_CALL(agent, RegisterMember(::testing::_, ::testing::_))
      .WillRepeatedly(::testing::InvokeWithoutArgs([]() {
        std::unique_ptr<MockBUCBuffer> buffer =
            std::make_unique<MockBUCBuffer>();
        EXPECT_CALL(*buffer, SerializedRecordMember)
            .WillRepeatedly(::testing::Invoke([]() {
              std::unique_ptr<const MockBUCSerialized> serialized =
                  std::make_unique<const MockBUCSerialized>();
              static auto generator = [n = 0]() mutable -> const std::string & {
                std::string *s = new std::string(5 + n++, 48);
                std::iota(s->begin(), s->end(), 49);
                return *s;
              };
              const auto &string = generator();
              EXPECT_CALL(*serialized, DataMember)
                  .WillRepeatedly(::testing::Return(
                      reinterpret_cast<const std::uint8_t *>(string.data())));
              EXPECT_CALL(*serialized, SizeMember)
                  .WillRepeatedly(::testing::Return(string.length()));
              return serialized;
            }));
        return buffer;
      }));

  using blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnBuilder;
  auto builder = GdfColumnBuilder::MakeInHost(agent);

  auto payload = builder->Data(fixture.data())
                     .Valid(fixture.valid())
                     .Size(fixture.size())
                     .DType(fixture.dtype())
                     .NullCount(fixture.nullCount())
                     .ColumnName(fixture.columnName())
                     .Build();

  auto &buffer = payload->Deliver();

  using blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnSpecialized;
  auto specialized = GdfColumnSpecialized::MakeInHost(buffer);

  using blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnPayload;
  auto resultPayload = specialized->Apply();

  const GdfColumnPayload &gdfColumnPayload =
      *static_cast<GdfColumnPayload *>(resultPayload.get());

  EXPECT_EQ(buffer.Size(), gdfColumnPayload.Deliver().Size());

  EXPECT_EQ(5, gdfColumnPayload.Data().Size());
  EXPECT_FALSE(std::memcmp("12345", gdfColumnPayload.Data().Data(), 5));

  EXPECT_EQ(6, gdfColumnPayload.Valid().Size());
  EXPECT_FALSE(std::memcmp("123456", gdfColumnPayload.Valid().Data(), 6));

  EXPECT_EQ(fixture.size(), gdfColumnPayload.Size());

  EXPECT_EQ(fixture.dtype(), gdfColumnPayload.DType());

  EXPECT_EQ(fixture.nullCount(), gdfColumnPayload.NullCount());

  EXPECT_EQ(7, gdfColumnPayload.ColumnName().Size());
  EXPECT_FALSE(std::memcmp("ColName", gdfColumnPayload.ColumnName().Data(), 7));

  // TODO: Check same values in payload and resultPayload
}

// Tests for gdf column tools

class gdf_column {
public:
  void *      data;
  void *      valid;
  std::size_t size;
};

static inline void
AddTo(std::vector<gdf_column> &gdfColumns,
      std::uintptr_t           data,
      std::uintptr_t           valid,
      std::size_t              size) {
  gdfColumns.push_back(gdf_column{
      reinterpret_cast<void *>(data), reinterpret_cast<void *>(valid), size});
}

class GdfColumnInfoDummy {
public:
  static inline std::size_t
  DataSize(const gdf_column &) noexcept {
    return 1;
  }
  static inline std::size_t
  ValidSize(const gdf_column &) noexcept {
    return 2;
  }
};

class BufferForTest
    : public blazingdb::communication::messages::tools::gdf_columns::Buffer {
private:
  const std::string &string;

public:
  BufferForTest(const std::string &str) : string(str) {}

  const void *
  Data() const noexcept final {
    return string.data();
  }

  std::size_t
  Size() const noexcept final {
    return string.length();
  }
};

TEST(GdfColumnsTest, DeliverAndCollect) {
  MockBUCAgent agent;
  EXPECT_CALL(agent, RegisterMember(::testing::_, ::testing::_))
      .WillRepeatedly(::testing::Invoke([](auto, auto) {
        std::unique_ptr<MockBUCBuffer> buffer =
            std::make_unique<MockBUCBuffer>();
        EXPECT_CALL(*buffer, SerializedRecordMember)
            .WillRepeatedly(::testing::Invoke([]() {
              std::unique_ptr<const MockBUCSerialized> serialized =
                  std::make_unique<const MockBUCSerialized>();
              EXPECT_CALL(*serialized, DataMember)
                  .WillRepeatedly(::testing::Return(
                      reinterpret_cast<const std::uint8_t *>("12345")));
              EXPECT_CALL(*serialized, SizeMember)
                  .WillRepeatedly(::testing::Return(5));
              return serialized;
            }));
        return buffer;
      }));

  std::vector<gdf_column> gdfColumns;
  AddTo(gdfColumns, 100, 200, 10);
  AddTo(gdfColumns, 101, 201, 25);
  AddTo(gdfColumns, 102, 202, 50);

  std::string result =
      blazingdb::communication::messages::tools::gdf_columns::DeliverFrom<
          GdfColumnInfoDummy>(gdfColumns, agent);

  BufferForTest resultBuffer(result);

  using blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnDispatcher;
  auto dispatcher = GdfColumnDispatcher::MakeInHost(resultBuffer);

  using blazingdb::communication::messages::tools::gdf_columns::Collector;
  std::unique_ptr<Collector> collector = dispatcher->Dispatch();

  using blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnPayload;

  EXPECT_EQ(3, collector->Length());

  EXPECT_EQ(10,
            static_cast<const GdfColumnPayload &>(collector->Get(0)).Size());
  EXPECT_EQ(25,
            static_cast<const GdfColumnPayload &>(collector->Get(1)).Size());
  EXPECT_EQ(50,
            static_cast<const GdfColumnPayload &>(collector->Get(2)).Size());

  // TODO: support diferent sizes in collector
}
