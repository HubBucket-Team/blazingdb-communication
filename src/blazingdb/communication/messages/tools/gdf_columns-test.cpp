#include "gdf_columns.h"

#include <array>
#include <cstring>
#include <numeric>

#include <cuda_runtime_api.h>

#include <blazingdb/uc/Context.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../ColumnDataMessage.h"
#include "../DataPivot.h"
#include "../DataScatterMessage.h"
#include "../GpuComponentMessage.h"
#include "../NodeDataMessage.h"
#include "../PartitionPivotsMessage.h"
#include "../SampleToNodeMasterMessage.h"

namespace {
class CategoryFixture {
  using CudaBuffer =
      blazingdb::communication::messages::tools::gdf_columns::CudaBuffer;

public:
  explicit CategoryFixture(const void *const strs,
                           const std::size_t strsSize,
                           const void *const mem,
                           const std::size_t memSize,
                           const void *const map,
                           const std::size_t mapSize,
                           std::size_t       count,
                           std::size_t       keys,
                           std::size_t       size,
                           std::size_t       base_address)
      : strs_{CudaBuffer::Make(strs, strsSize)},
        mem_{CudaBuffer::Make(mem, memSize)},
        map_{CudaBuffer::Make(map, mapSize)},
        count_{count},
        keys_{keys},
        size_{size},
        base_address_{base_address} {}

  const CudaBuffer &
  strs() const noexcept {
    return *strs_;
  }

  const CudaBuffer &
  mem() const noexcept {
    return *mem_;
  }

  const CudaBuffer &
  map() const noexcept {
    return *map_;
  }

  std::size_t
  count() const noexcept {
    return count_;
  }

  std::size_t
  keys() const noexcept {
    return keys_;
  }

  std::size_t
  size() const noexcept {
    return size_;
  }

  std::size_t
  baseAddress() const noexcept {
    return base_address_;
  }

  std::unique_ptr<CudaBuffer> strs_;
  std::unique_ptr<CudaBuffer> mem_;
  std::unique_ptr<CudaBuffer> map_;
  std::size_t                 count_;
  std::size_t                 keys_;
  std::size_t                 size_;
  std::size_t                 base_address_;
};

class DTypeInfoFixture {
public:
  explicit DTypeInfoFixture(const std::size_t time_unit,
                            CategoryFixture & category)
      : time_unit_{time_unit}, category_{category} {}

  std::size_t
  timeUnit() const noexcept {
    return time_unit_;
  }

  const CategoryFixture &
  category() const noexcept {
    return category_;
  }

  std::size_t      time_unit_;
  CategoryFixture &category_;
};

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
                            const std::string       columnName,
                            DTypeInfoFixture &      dtypeInfo)
      : data_{CudaBuffer::Make(data, dataSize)},
        valid_{CudaBuffer::Make(valid, validSize)},
        size_{size},
        dtype_{dtype},
        nullCount_{nullCount},
        columnNameStr_{columnName},
        columnName_{
            HostBuffer::Make(columnNameStr_.data(), columnNameStr_.size())},
        dtypeInfo_{dtypeInfo} {}

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

  const DTypeInfoFixture &
  dtypeInfo() const noexcept {
    return dtypeInfo_;
  }

private:
  std::unique_ptr<CudaBuffer> data_;
  std::unique_ptr<CudaBuffer> valid_;
  std::size_t                 size_;
  std::int_fast32_t           dtype_;
  std::size_t                 nullCount_;
  std::string                 columnNameStr_;
  std::unique_ptr<HostBuffer> columnName_;
  DTypeInfoFixture &          dtypeInfo_;
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

static inline CategoryFixture
CreateBasicCategoryFixture() {
  const std::size_t strsSize = 200;
  const void *const strs     = CreateCudaSequence(strsSize);

  const std::size_t memSize = 100;
  const void *const mem     = CreateCudaSequence(memSize);

  const std::size_t mapSize = 50;
  const void *const map     = CreateCudaSequence(mapSize);

  const std::size_t count = 500;

  const std::size_t keys = 10;

  const std::size_t size = 20;

  const std::size_t baseAddress = 0;

  return CategoryFixture{strs,
                         strsSize,
                         mem,
                         memSize,
                         map,
                         mapSize,
                         count,
                         keys,
                         size,
                         baseAddress};
}

static inline DTypeInfoFixture
CreateBasicDTypeInfoFixture(CategoryFixture &category) {
  const std::size_t timeUnit = 1;

  return DTypeInfoFixture{timeUnit, category};
}

static inline GdfColumnFixture
CreateBasicGdfColumnFixture(DTypeInfoFixture &dtypeInfo) {
  const std::size_t dataSize = 2000;
  const void *const data     = CreateCudaSequence(dataSize);

  std::size_t validSize = 1000;
  void *      valid     = CreateCudaSequence(validSize);

  const std::size_t size = 500;

  const std::int_fast32_t dtype = 3;

  const std::size_t nullCount = 5;

  const std::string columnName = "ColName";

  return GdfColumnFixture{data,
                          dataSize,
                          valid,
                          validSize,
                          size,
                          dtype,
                          nullCount,
                          columnName,
                          dtypeInfo};
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

class MockBUCTransport : public blazingdb::uc::Transport {
public:
  MOCK_METHOD0(Get, std::future<void>());
};

TEST(GdfColumnBuilderTest, CheckPayload) {
  auto categoryFixture  = CreateBasicCategoryFixture();
  auto dtypeInfoFixture = CreateBasicDTypeInfoFixture(categoryFixture);
  auto fixture          = CreateBasicGdfColumnFixture(dtypeInfoFixture);

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

  using blazingdb::communication::messages::tools::gdf_columns::CategoryBuilder;
  auto categoryBuilder = CategoryBuilder::MakeInHost(agent);

  auto categoryPayload = categoryBuilder->Strs(categoryFixture.strs())
                             .Mem(categoryFixture.mem())
                             .Map(categoryFixture.map())
                             .Count(categoryFixture.count())
                             .Keys(categoryFixture.keys())
                             .Size(categoryFixture.size())
                             .BaseAddress(categoryFixture.baseAddress())
                             .Build();

  using blazingdb::communication::messages::tools::gdf_columns::
      DTypeInfoBuilder;
  auto dtypeInfoBuilder = DTypeInfoBuilder::MakeInHost(agent);

  using blazingdb::communication::messages::tools::gdf_columns::CategoryPayload;
  auto dtypeInfoPayload =
      dtypeInfoBuilder->TimeUnit(dtypeInfoFixture.timeUnit())
          .Category(*categoryPayload)
          .Build();

  using blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnBuilder;
  auto builder = GdfColumnBuilder::MakeInHost(agent);

  auto payload = builder->Data(fixture.data())
                     .Valid(fixture.valid())
                     .Size(fixture.size())
                     .DType(fixture.dtype())
                     .NullCount(fixture.nullCount())
                     .ColumnName(fixture.columnName())
                     .DTypeInfo(*dtypeInfoPayload)
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

  using blazingdb::communication::messages::tools::gdf_columns::
      DTypeInfoPayload;
  EXPECT_EQ(fixture.dtypeInfo().timeUnit(),
            static_cast<const DTypeInfoPayload &>(
                *gdfColumnPayload.DTypeInfo().ToPayload())
                .TimeUnit());

  EXPECT_EQ(7, gdfColumnPayload.ColumnName().Size());
  EXPECT_FALSE(std::memcmp("ColName", gdfColumnPayload.ColumnName().Data(), 7));

  // TODO: Check same values in payload and resultPayload
}

// Tests for gdf column tools

class gdf_column {
public:
  const void *      data;
  const void *      valid;
  int               size;
  std::int_fast32_t dtype;
  int               null_count;
  class {
  } dtype_info;
  char *col_name;
};

static inline void
AddTo(std::vector<gdf_column> &gdfColumns,
      std::uintptr_t           data,
      std::uintptr_t           valid,
      std::size_t              size,
      std::int_fast32_t        dtype,
      std::size_t              null_count) {
  gdfColumns.push_back(gdf_column{reinterpret_cast<void *>(data),
                                  reinterpret_cast<void *>(valid),
                                  static_cast<int>(size),
                                  dtype,
                                  static_cast<int>(null_count),
                                  {},
                                  const_cast<char *>("column name")});
}

template <class gdf_column>
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
        EXPECT_CALL(*buffer, Link(::testing::_))
            .WillRepeatedly(::testing::InvokeWithoutArgs([]() {
              std::unique_ptr<MockBUCTransport> transport =
                  std::make_unique<MockBUCTransport>();
              EXPECT_CALL(*transport, Get)
                  .WillRepeatedly(::testing::InvokeWithoutArgs(
                      []() { return std::async([]() {}); }));
              return transport;
            }));
        return buffer;
      }));

  std::vector<gdf_column> gdfColumns;
  AddTo(gdfColumns, 100, 200, 10, 1, 0);
  AddTo(gdfColumns, 101, 201, 25, 2, 5);
  AddTo(gdfColumns, 102, 202, 50, 3, 10);

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

  std::vector<gdf_column> gdf_columns =
      blazingdb::communication::messages::tools::gdf_columns::CollectFrom<
          gdf_column>(result, agent);

  EXPECT_EQ(10, gdf_columns[0].size);
  EXPECT_EQ(25, gdf_columns[1].size);
  EXPECT_EQ(50, gdf_columns[2].size);
}