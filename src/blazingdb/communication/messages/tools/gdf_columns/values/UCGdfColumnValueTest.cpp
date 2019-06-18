#include "UCGdfColumnValue.hpp"

#include "../buffers/NullBuffer.hpp"
#include "../buffers/ViewBuffer.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../common/test-helpers.hpp"

using namespace blazingdb::communication::messages::tools::gdf_columns;

class MockGdfColumnPayload : public GdfColumnPayload {
public:
  MOCK_CONST_METHOD0_NE(Deliver, const Buffer &());
  MOCK_CONST_METHOD0_NE(Data, const UCBuffer &());
  MOCK_CONST_METHOD0_NE(Valid, const UCBuffer &());
  MOCK_CONST_METHOD0_NE(Size, std::size_t());
  MOCK_CONST_METHOD0_NE(DType, std::int_fast32_t());
  MOCK_CONST_METHOD0_NE(NullCount, std::size_t());
  MOCK_CONST_METHOD0_NE(DTypeInfo, const DTypeInfoPayload &());
  MOCK_CONST_METHOD0_NE(ColumnName, const UCBuffer &());
};

TEST(UCGdfColumnValueTest, MakeValue) {
  MockGdfColumnPayload gdfColumnPayload;
  EXPECT_CALL(gdfColumnPayload, Deliver)
      .WillRepeatedly(testing::ReturnPointee(std::make_shared<NullBuffer>()));
  EXPECT_CALL(gdfColumnPayload, Data)
      .WillRepeatedly(
          testing::ReturnPointeeCast<UCBuffer>(std::make_shared<ViewBuffer>(
              reinterpret_cast<const void *const>(111), 1)));
  EXPECT_CALL(gdfColumnPayload, Valid)
      .WillRepeatedly(
          testing::ReturnPointeeCast<UCBuffer>(std::make_shared<ViewBuffer>(
              reinterpret_cast<const void *const>(222), 2)));
  EXPECT_CALL(gdfColumnPayload, Size).WillRepeatedly(testing::Return(12345));

  MockUCAgent ucAgent;
  EXPECT_CALL(ucAgent, Register(testing::_, testing::_))
      .WillRepeatedly(testing::InvokeWithoutArgs([]() {
        std::unique_ptr<MockUCBuffer> ucBuffer =
            std::make_unique<MockUCBuffer>();
        EXPECT_CALL(*ucBuffer, Link(testing::_))
            .WillRepeatedly(testing::InvokeWithoutArgs([]() {
              std::unique_ptr<MockUCTransport> ucTransport =
                  std::make_unique<MockUCTransport>();
              EXPECT_CALL(*ucTransport, Get)
                  .Times(testing::Exactly(2))
                  .WillRepeatedly([]() { return std::async([]() {}); });
              return ucTransport;
            }));
        return ucBuffer;
      }));

  auto gdfColumnValue = GdfColumnValue::Make(gdfColumnPayload, ucAgent);

  EXPECT_EQ(12345, gdfColumnValue->size());
}
