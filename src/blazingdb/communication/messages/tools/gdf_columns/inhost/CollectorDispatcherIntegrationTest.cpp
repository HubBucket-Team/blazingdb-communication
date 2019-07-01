#include "../../gdf_columns.h"

#include <cstring>

#include <blazingdb/uc/internal/macros.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

class MockPayload
    : public blazingdb::communication::messages::tools::gdf_columns::Payload {
public:
  using Buffer = blazingdb::communication::messages::tools::gdf_columns::Buffer;

  const Buffer &
  Deliver() const noexcept final {
    return DeliverMember();
  }

  MOCK_CONST_METHOD0(DeliverMember, const Buffer &());
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

static UC_INLINE void
ExpectCall(MockBuffer &mock, const std::string &content) {
  EXPECT_CALL(mock, DataMember)
      .WillRepeatedly(::testing::Return(content.c_str()));
  EXPECT_CALL(mock, SizeMember)
      .WillRepeatedly(::testing::Return(content.length()));
}

static UC_INLINE void
ExpectCall(MockPayload &mock, const MockBuffer &buffer) {
  EXPECT_CALL(mock, DeliverMember).WillRepeatedly(::testing::ReturnRef(buffer));
}

static UC_INLINE void
ExpectCall(MockPayload &       mockPayload,
           MockBuffer &        mockBuffer,
           const std::string &&content) {
  ExpectCall(mockBuffer, std::move(content));
  ExpectCall(mockPayload, mockBuffer);
}

static UC_INLINE auto
CheckCollect() {
  using blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnCollector;
  auto collector = GdfColumnCollector::MakeInHost();

  MockPayload payload1, payload2, payload3;
  MockBuffer  buffer1, buffer2, buffer3;

  ExpectCall(payload1, buffer1, "11111");
  ExpectCall(payload2, buffer2, "222222");
  ExpectCall(payload3, buffer3, "3333333");

  collector->Add(payload1.Deliver());
  collector->Add(payload2.Deliver());
  collector->Add(payload3.Deliver());

  EXPECT_EQ(3, collector->Length());

  auto buffer = collector->Collect();

  EXPECT_LE(18, buffer->Size());

  std::string content{
      static_cast<const std::string::value_type *>(buffer->Data()),
      buffer->Size()};

  std::vector<std::string> subs{{"11111", "222222", "3333333"}};
  for (const auto &sub : subs) {
    auto result = std::find_end(
        content.cbegin(), content.cend(), sub.cbegin(), sub.cend());

    EXPECT_NE(result, content.cend());
  }

  return buffer;
}

static UC_INLINE void
CheckDispatch(
    const blazingdb::communication::messages::tools::gdf_columns::Buffer
        &buffer) {
  using blazingdb::communication::messages::tools::gdf_columns::
      GdfColumnDispatcher;
  auto dispatcher = GdfColumnDispatcher::MakeInHost(buffer);

  auto collector = dispatcher->Dispatch();

  EXPECT_EQ(3, collector->Length());
}

TEST(CollectorDispatcherIntegrationTest, CollectAndDispatch) {
  auto buffer = CheckCollect();
  CheckDispatch(*buffer);
}
