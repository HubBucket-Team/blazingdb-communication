#include "../../gdf_columns.h"

#include <cstring>

#include <cuda_runtime_api.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

TEST(CategoryBuilderTest, CheckPayload) {
  auto categoryFixture  = CreateBasicCategoryFixture();

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

  auto payload = categoryBuilder->Strs(categoryFixture.strs())
                             .Mem(categoryFixture.mem())
                             .Map(categoryFixture.map())
                             .Count(categoryFixture.count())
                             .Keys(categoryFixture.keys())
                             .Size(categoryFixture.size())
                             .BaseAddress(categoryFixture.baseAddress())
                             .Build();

  auto &buffer = payload->Deliver();

  using blazingdb::communication::messages::tools::gdf_columns::
      CategorySpecialized;
  auto specialized = CategorySpecialized::MakeInHost(buffer);

  using blazingdb::communication::messages::tools::gdf_columns::
      CategoryPayload;
  auto resultPayload = specialized->Apply();

  const CategoryPayload &categoryPayload =
      *static_cast<CategoryPayload *>(resultPayload.get());

  EXPECT_EQ(buffer.Size(), categoryPayload.Deliver().Size());

  EXPECT_EQ(5, categoryPayload.Strs().Size());
  EXPECT_FALSE(std::memcmp("12345", categoryPayload.Strs().Data(), 5));

  EXPECT_EQ(6, categoryPayload.Mem().Size());
  EXPECT_FALSE(std::memcmp("123456", categoryPayload.Mem().Data(), 6));

  EXPECT_EQ(7, categoryPayload.Map().Size());
  EXPECT_FALSE(std::memcmp("1234567", categoryPayload.Map().Data(), 7));

  EXPECT_EQ(categoryFixture.count(), categoryPayload.Count());
  EXPECT_EQ(categoryFixture.keys(), categoryPayload.Keys());
  EXPECT_EQ(categoryFixture.size(), categoryPayload.Size());
  EXPECT_EQ(categoryFixture.baseAddress(), categoryPayload.BaseAddress());
}
