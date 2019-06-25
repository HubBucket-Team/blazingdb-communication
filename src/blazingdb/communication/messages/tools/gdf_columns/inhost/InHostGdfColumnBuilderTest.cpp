#include "InHostGdfColumnBuilder.hpp"

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../common/test-helpers.hpp"

template <const std::size_t size>
static UC_INLINE auto
InvokeBuffer(const char (&data)[size]) {
  return testing::InvokeWithoutArgs([data]() {
    std::unique_ptr<uc::MockBuffer> buffer = std::make_unique<uc::MockBuffer>();
    EXPECT_CALL(*buffer, SerializedRecord)
        .WillOnce(testing::InvokeWithoutArgs([data = std::string{data}]() {
          std::unique_ptr<uc::MockSerialized> serialized =
              std::make_unique<uc::MockSerialized>();
          EXPECT_CALL(*serialized, Data)
              .WillOnce(testing::Return(
                  reinterpret_cast<const std::uint8_t *>(&data[0])));
          EXPECT_CALL(*serialized, Size)
              .WillRepeatedly(testing::Return(data.length()));
          return serialized;
        }));
    return buffer;
  });
}

TEST(InHostGdfColumnBuilder, Build) {
  static const void *data111 = reinterpret_cast<const void *>(111);
  static const void *data222 = reinterpret_cast<const void *>(222);

  using blazingdb::communication::messages::tools::gdf_columns::CudaBuffer;
  using blazingdb::communication::messages::tools::gdf_columns::HostBuffer;
  const std::unique_ptr<const CudaBuffer> dataCudaBuffer =
      CudaBuffer::Make(data111, 1);
  const std::unique_ptr<const CudaBuffer> validCudaBuffer =
      CudaBuffer::Make(data222, 2);
  const std::size_t                       size          = 333;
  const std::int_fast32_t                 dtype         = 444;
  const std::size_t                       nullCount     = 555;
  const std::string                       rawColumnName = "666";
  const std::unique_ptr<const HostBuffer> columnNameBuffer =
      HostBuffer::Make(rawColumnName.data(), rawColumnName.length());

  uc::MockAgent agent;
  {
    testing::InSequence inSequence;
    EXPECT_CALL(agent, Register(data111, 1)).WillOnce(InvokeBuffer("12345"));
    EXPECT_CALL(agent, Register(data222, 2)).WillOnce(InvokeBuffer("123456"));
  }

  using blazingdb::communication::messages::tools::gdf_columns::
      InHostGdfColumnBuilder;
  InHostGdfColumnBuilder builder{agent};

  using blazingdb::communication::messages::tools::gdf_columns::Payload;
  std::unique_ptr<const Payload> payload = builder.Data(*dataCudaBuffer)
                                               .Valid(*validCudaBuffer)
                                               .Size(size)
                                               .DType(dtype)
                                               .NullCount(nullCount)
                                               .ColumnName(*columnNameBuffer)
                                               .Build();

  using blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnPayload;
  const GdfColumnPayload &gdfColumnPayload = GdfColumnPayload::From(*payload);

  // Checks

  EXPECT_EQ(5, gdfColumnPayload.Data().Size());
  EXPECT_FALSE(std::memcmp("12345", gdfColumnPayload.Data().Data(), 5));

  EXPECT_EQ(6, gdfColumnPayload.Valid().Size());
  EXPECT_FALSE(std::memcmp("123456", gdfColumnPayload.Valid().Data(), 6));

  EXPECT_EQ(333, gdfColumnPayload.Size());

  EXPECT_EQ(444, gdfColumnPayload.DType());

  EXPECT_EQ(555, gdfColumnPayload.NullCount());

  EXPECT_FALSE(std::memcmp("666", gdfColumnPayload.ColumnName().Data(), 3));
}
