#include "Node.h"

#include <gtest/gtest.h>

TEST(NodeTest, NodeCreationFromCreatedBuffer) {
  const std::shared_ptr<blazingdb::communication::Address> address =
      blazingdb::communication::Address::Make("1.2.3.4", 1234);

  const std::shared_ptr<blazingdb::communication::Node> node =
      blazingdb::communication::Node::Make(address);

  const std::shared_ptr<blazingdb::communication::Buffer> buffer =
      node->ToBuffer();

  const std::shared_ptr<blazingdb::communication::Node> resultNode =
      blazingdb::communication::Node::Make(*buffer);

  EXPECT_EQ(*resultNode, *node);
}
