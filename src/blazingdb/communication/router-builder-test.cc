#include "RouterBuilder.h"

#include <blazingdb/communication/Router.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

class FooMockListener : public blazingdb::communication::Listener {
public:
  MOCK_CONST_METHOD0(Process, void());
};

class BarMockListener : public blazingdb::communication::Listener {
public:
  MOCK_CONST_METHOD0(Process, void());
};

TEST(RouterBuilderTest, Build) {
  FooMockListener fooMockListener;
  BarMockListener barMockListener;

  EXPECT_CALL(fooMockListener, Process()).Times(1);

  blazingdb::communication::RouterBuilder builder;

  auto fooMessageToken = blazingdb::communication::MessageToken::Make();
  auto barMessageToken = blazingdb::communication::MessageToken::Make();
  auto bazMessageToken = blazingdb::communication::MessageToken::Make();

  builder.Append(*fooMessageToken, fooMockListener);
  builder.Append(*barMessageToken, barMockListener);

  std::unique_ptr<blazingdb::communication::Router> router = builder.build();

  router->Call(*fooMessageToken);

  EXPECT_THROW(router->Call(*bazMessageToken), std::runtime_error);
}
