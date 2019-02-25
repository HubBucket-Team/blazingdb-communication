#include "Manager.h"
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

  const std::string node =
      "{\"node_ip\": \"192.168.0.1\", \"node_port\": 8000}";
  blazingdb::communication::Buffer buffer{
      reinterpret_cast<const std::uint8_t *>(node.data()), node.size()};

  try {
    const std::unique_ptr<blazingdb::communication::network::Status> status =
        client->SendNodeData("localhost", 9000, buffer);
    EXPECT_TRUE(status->IsOk());
  } catch (const blazingdb::communication::network::Client::SendError &error) {
    FAIL() << error.what();
  }

  // End
  manager->Close();
  managerRunThread.join();
}
