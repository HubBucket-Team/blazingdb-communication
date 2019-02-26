#include "Manager.h"
#include "blazingdb/communication/messages/NodeDataMessage.h"
#include "network/Client.h"
#include "network/ClientExceptions.h"

#include <thread>

#include <gtest/gtest.h>

TEST(TestManager, ConnectionAndGenerateContext) {
  // Start manager
  std::unique_ptr<blazingdb::communication::Manager> manager =
      blazingdb::communication::Manager::Make();

  std::thread managerRunThread{[&manager]() { manager->Run(); }};

  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Register node
  std::unique_ptr<blazingdb::communication::network::Client> client =
      blazingdb::communication::network::Client::Make();

  blazingdb::communication::Node node(
      blazingdb::communication::Address::Make("1.2.3.4", 1234));

  // Create message
  blazingdb::communication::messages::NodeDataMessage nodeDataMessage(node);

  try {
    const std::unique_ptr<blazingdb::communication::network::Status> status =
        client->SendNodeData("localhost", 9000, nodeDataMessage);
    EXPECT_TRUE(status->IsOk());
  } catch (const blazingdb::communication::network::Client::SendError &error) {
    FAIL() << error.what();
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  EXPECT_EQ(manager->getCluster().getTotalNodes(), 1);

  blazingdb::communication::Context *context =
      manager->generateContext("plan", {"file"});

  EXPECT_EQ("plan", context->getLogicalPlan());
  EXPECT_EQ(std::vector<std::string>{"file"}, context->getDataFiles());

  // End
  manager->Close();
  managerRunThread.join();
}
