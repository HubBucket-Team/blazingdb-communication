#include "InHostDTypeInfoBuilder.hpp"

#include "InHostDTypeInfoPayload.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../common/test-helpers.hpp"

class MockAgent : public blazingdb::uc::Agent {
public:
  MOCK_CONST_METHOD2_NE(
      Register,
      std::unique_ptr<blazingdb::uc::Buffer>(const void *&, const std::size_t));
};

using blazingdb::communication::messages::tools::gdf_columns::
    InHostDTypeInfoBuilder;
using blazingdb::communication::messages::tools::gdf_columns::
    InHostDTypeInfoPayload;

TEST(InHostDTypeInfoBuilderTest, BuildAndDispatch) {
  MockAgent agent;

  InHostDTypeInfoBuilder builder{agent};

  auto  payload = builder.TimeUnit(12345).Build();
  auto &dtypeInfoPayload =
      static_cast<const InHostDTypeInfoPayload &>(*payload);

  EXPECT_EQ(12345, dtypeInfoPayload.TimeUnit());
}
