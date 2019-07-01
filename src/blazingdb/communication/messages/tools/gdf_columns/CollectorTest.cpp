#include "interfaces.hpp"

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/test-helpers.hpp"

class CollectorTest : public testing::Test {
protected:
  void
  SetUp() final {
    // Prepare mock buffers
    EXPECT_CALL(buffer[0], Data).WillRepeatedly(testing::Return("12345"));
    EXPECT_CALL(buffer[0], Size).WillRepeatedly(testing::Return(5));

    EXPECT_CALL(buffer[1], Data).WillRepeatedly(testing::Return("123456"));
    EXPECT_CALL(buffer[1], Size).WillRepeatedly(testing::Return(6));

    EXPECT_CALL(buffer[2], Data).WillRepeatedly(testing::Return("1234567"));
    EXPECT_CALL(buffer[2], Size).WillRepeatedly(testing::Return(7));

    // Prepare mock payloads
    EXPECT_CALL(payload[0], Deliver)
        .WillRepeatedly(testing::ReturnRef(buffer[0]));

    EXPECT_CALL(payload[1], Deliver)
        .WillRepeatedly(testing::ReturnRef(buffer[1]));

    EXPECT_CALL(payload[2], Deliver)
        .WillRepeatedly(testing::ReturnRef(buffer[2]));

    // Create collector
    // Use a parametrized test to perform unit test about different
    // implementation (not InHost)
    collector = blazingdb::communication::messages::tools::gdf_columns::
        GdfColumnCollector::MakeInHost();
  }

  using Collector =
      blazingdb::communication::messages::tools::gdf_columns::Collector;
  std::unique_ptr<Collector> collector;

  blazingdb::testing::MockBuffer  buffer[3];
  blazingdb::testing::MockPayload payload[3];

  using Buffer = blazingdb::communication::messages::tools::gdf_columns::Buffer;
};

TEST_F(CollectorTest, Add) {
  EXPECT_EQ(0, collector->Length());
  collector->Add(payload[0].Deliver());
  EXPECT_EQ(1, collector->Length());
  collector->Add(payload[1].Deliver());
  EXPECT_EQ(2, collector->Length());
  collector->Add(payload[2].Deliver());
  EXPECT_EQ(3, collector->Length());
}

TEST_F(CollectorTest, Iterate) {
  collector->Add(payload[0].Deliver());
  collector->Add(payload[1].Deliver());
  collector->Add(payload[2].Deliver());

  std::vector<const Buffer *> collectedBuffers;
  collectedBuffers.reserve(3);

  for (const Buffer &buffer : *collector) {
    collectedBuffers.push_back(&buffer);
  }

  EXPECT_EQ(5, collectedBuffers[0]->Size());
  EXPECT_EQ(6, collectedBuffers[1]->Size());
  EXPECT_EQ(7, collectedBuffers[2]->Size());

  EXPECT_FALSE(std::memcmp("12345", collectedBuffers[0]->Data(), 5));
  EXPECT_FALSE(std::memcmp("123456", collectedBuffers[1]->Data(), 6));
  EXPECT_FALSE(std::memcmp("1234567", collectedBuffers[2]->Data(), 7));
}
