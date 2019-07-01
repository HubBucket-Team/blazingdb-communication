#include "interfaces.hpp"

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/test-helpers.hpp"

// Run `CollectorTest` to check the collector is working fine

TEST(DispatcherTest, Dispatch) {
  // Prepare data fixtures

  blazingdb::testing::MockBuffer fixtureBuffers[3];

  EXPECT_CALL(fixtureBuffers[0], Data).WillRepeatedly(testing::Return("12345"));
  EXPECT_CALL(fixtureBuffers[0], Size).WillRepeatedly(testing::Return(5));

  EXPECT_CALL(fixtureBuffers[1], Data)
      .WillRepeatedly(testing::Return("123456"));
  EXPECT_CALL(fixtureBuffers[1], Size).WillRepeatedly(testing::Return(6));

  EXPECT_CALL(fixtureBuffers[2], Data)
      .WillRepeatedly(testing::Return("1234567"));
  EXPECT_CALL(fixtureBuffers[2], Size).WillRepeatedly(testing::Return(7));

  // Generate buffer under test

  auto collector = blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnCollector::MakeInHost();

  for (auto &buffer : fixtureBuffers) { collector->Add(buffer); }

  auto resultBuffer = collector->Collect();

  auto dispatcher = blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnDispatcher::MakeInHost(*resultBuffer);

  auto resultCollector = dispatcher->Dispatch();

  // Check results

  EXPECT_EQ(3, resultCollector->Length());

  using blazingdb::communication::messages::tools::gdf_columns::Buffer;
  std::vector<const Buffer *> resultBuffers;
  resultBuffers.reserve(resultCollector->Length());

  std::transform(collector->begin(),
                 collector->end(),
                 std::back_inserter(resultBuffers),
                 [](const Buffer &buffer) { return &buffer; });

  EXPECT_EQ(5, resultBuffers[0]->Size());
  EXPECT_EQ(6, resultBuffers[1]->Size());
  EXPECT_EQ(7, resultBuffers[2]->Size());

  EXPECT_FALSE(std::memcmp("12345", resultBuffers[0]->Data(), 5));
  EXPECT_FALSE(std::memcmp("123456", resultBuffers[1]->Data(), 6));
  EXPECT_FALSE(std::memcmp("1234567", resultBuffers[2]->Data(), 7));
}
