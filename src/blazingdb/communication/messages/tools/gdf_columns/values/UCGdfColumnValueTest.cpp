#include "UCGdfColumnValue.hpp"

#include "../buffers/NullBuffer.hpp"
#include "../buffers/ViewBuffer.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../common/test-helpers.hpp"

using namespace blazingdb::communication::messages::tools::gdf_columns;

class UC_NOEXPORT MockMemoryRuntime : public MemoryRuntime {
public:
  MOCK_METHOD1(Allocate, void *(const std::size_t));
  MOCK_METHOD0(Synchronize, void());
};

class MockGdfColumnPayload : public GdfColumnPayload {
public:
  MOCK_CONST_METHOD0_NE(Deliver, const Buffer &());
  MOCK_CONST_METHOD0_NE(Data, const UCBuffer &());
  MOCK_CONST_METHOD0_NE(Valid, const UCBuffer &());
  MOCK_CONST_METHOD0_NE(Size, std::size_t());
  MOCK_CONST_METHOD0_NE(DType, std::int_fast32_t());
  MOCK_CONST_METHOD0_NE(NullCount, std::size_t());
  MOCK_CONST_METHOD0_NE(DTypeInfo, const PayloadableBuffer &());
  MOCK_CONST_METHOD0_NE(ColumnName, const UCBuffer &());
};

TEST(UCGdfColumnValueTest, MakeValue) {
  std::unique_ptr<MockMemoryRuntime> memoryRuntime =
      std::make_unique<MockMemoryRuntime>();
  {
    testing::InSequence inSequence;
    EXPECT_CALL(*memoryRuntime, Allocate(testing::_))
        .WillOnce(testing::Return(reinterpret_cast<void *const>(111)));
    EXPECT_CALL(*memoryRuntime, Allocate(testing::_))
        .WillOnce(testing::Return(reinterpret_cast<void *const>(222)));
  }
  EXPECT_CALL(*memoryRuntime, Synchronize).WillRepeatedly(testing::Return());

  MockGdfColumnPayload gdfColumnPayload;
  EXPECT_CALL(gdfColumnPayload, Deliver)
      .WillRepeatedly(testing::ReturnPointee(std::make_shared<NullBuffer>()));
  EXPECT_CALL(gdfColumnPayload, Data)
      .WillRepeatedly(
          testing::ReturnPointeeCast<UCBuffer>(std::make_shared<ViewBuffer>(
              reinterpret_cast<const void *const>(-1), 1)));
  EXPECT_CALL(gdfColumnPayload, Valid)
      .WillRepeatedly(
          testing::ReturnPointeeCast<UCBuffer>(std::make_shared<ViewBuffer>(
              reinterpret_cast<const void *const>(-1), 2)));
  EXPECT_CALL(gdfColumnPayload, Size).WillRepeatedly(testing::Return(333));
  EXPECT_CALL(gdfColumnPayload, DType).WillRepeatedly(testing::Return(444));
  EXPECT_CALL(gdfColumnPayload, NullCount).WillRepeatedly(testing::Return(555));

  MockUCAgent ucAgent;
  EXPECT_CALL(ucAgent, Register(testing::_, testing::_))
      .WillRepeatedly(testing::InvokeWithoutArgs([]() {
        std::unique_ptr<MockUCBuffer> ucBuffer =
            std::make_unique<MockUCBuffer>();
        EXPECT_CALL(*ucBuffer, Link(testing::_))
            .WillRepeatedly(testing::InvokeWithoutArgs([]() {
              std::unique_ptr<MockUCTransport> ucTransport =
                  std::make_unique<MockUCTransport>();
              EXPECT_CALL(*ucTransport, Get).WillOnce([]() {
                return std::async([]() {});
              });
              return ucTransport;
            }));
        return ucBuffer;
      }));

  auto gdfColumnValue = std::make_unique<UCGdfColumnValue>(
      std::move(memoryRuntime), gdfColumnPayload, ucAgent);

  EXPECT_EQ(111, reinterpret_cast<std::intptr_t>(gdfColumnValue->data()));
  EXPECT_EQ(222, reinterpret_cast<std::intptr_t>(gdfColumnValue->valid()));
  EXPECT_EQ(333, gdfColumnValue->size());
  EXPECT_EQ(444, gdfColumnValue->dtype());
  EXPECT_EQ(555, gdfColumnValue->null_count());
}
