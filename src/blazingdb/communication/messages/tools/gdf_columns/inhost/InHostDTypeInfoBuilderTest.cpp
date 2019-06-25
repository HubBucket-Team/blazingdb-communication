#include "InHostDTypeInfoBuilder.hpp"

#include <cstring>

#include "InHostDTypeInfoPayload.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../buffers/StringBuffer.hpp"
#include "../common/test-helpers.hpp"

class MockAgent : public blazingdb::uc::Agent {
public:
  MOCK_CONST_METHOD2_NE(
      Register,
      std::unique_ptr<blazingdb::uc::Buffer>(const void *&, const std::size_t));
};

class MockCategoryPayload : public blazingdb::communication::messages::tools::
                                gdf_columns::CategoryPayload {
public:
  using Buffer = blazingdb::communication::messages::tools::gdf_columns::Buffer;
  using UCBuffer =
      blazingdb::communication::messages::tools::gdf_columns::UCBuffer;

  MOCK_CONST_METHOD0_NE(Deliver, const Buffer &());
  MOCK_CONST_METHOD0_NE(Strs, const UCBuffer &());
  MOCK_CONST_METHOD0_NE(Mem, const UCBuffer &());
  MOCK_CONST_METHOD0_NE(Map, const UCBuffer &());
  MOCK_CONST_METHOD0_NE(Count, std::size_t());
  MOCK_CONST_METHOD0_NE(Keys, std::size_t());
  MOCK_CONST_METHOD0_NE(Size, std::size_t());
  MOCK_CONST_METHOD0_NE(BaseAddress, std::size_t());
};

using blazingdb::communication::messages::tools::gdf_columns::
    InHostDTypeInfoBuilder;
using blazingdb::communication::messages::tools::gdf_columns::
    InHostDTypeInfoPayload;

TEST(InHostDTypeInfoBuilderTest, BuildWithoutCategory) {
  MockAgent agent;  // TODO(api): unnecessary agent

  InHostDTypeInfoBuilder builder{agent};

  auto  payload = builder.TimeUnit(12345).Build();
  auto &dtypeInfoPayload =
      static_cast<const InHostDTypeInfoPayload &>(*payload);

  EXPECT_EQ(12345, dtypeInfoPayload.TimeUnit());
}

TEST(InHostDTypeInfoBuilderTest, BuildWithCategory) {
  MockAgent agent;

  MockCategoryPayload categoryPayload;
  using blazingdb::communication::messages::tools::gdf_columns::StringBuffer;
  EXPECT_CALL(categoryPayload, Deliver)
      .WillOnce(testing::ReturnPointee(new StringBuffer{"12345"}));

  InHostDTypeInfoBuilder builder{agent};

  auto  payload = builder.TimeUnit(12345).Category(categoryPayload).Build();
  auto &dtypeInfoPayload =
      static_cast<const InHostDTypeInfoPayload &>(*payload);

  EXPECT_EQ(12345, dtypeInfoPayload.TimeUnit());
  EXPECT_EQ(0, std::memcmp("12345", dtypeInfoPayload.Category().Data(), 5));
}
