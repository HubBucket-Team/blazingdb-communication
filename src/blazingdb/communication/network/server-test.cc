#include "Server.h"
#include "Client.h"

#include <chrono>
#include <memory>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

class MockNodeToken : public blazingdb::communication::NodeToken {
public:
  MOCK_CONST_METHOD1(SameValueAs, bool(const NodeToken &));
};

class MockAddress : public blazingdb::communication::Address {
public:
  MOCK_CONST_METHOD1(SameValueAs, bool(const Address &));
};

TEST(ServerTest, Ehlo) {
  std::unique_ptr<blazingdb::communication::network::Server> server =
      blazingdb::communication::network::Server::Make();

  std::thread serverThread([&server]() { server->Run(); });

  std::this_thread::sleep_for(std::chrono::seconds(1));

  std::unique_ptr<blazingdb::communication::network::Client> client =
      blazingdb::communication::network::Client::Make();

  const std::uint8_t content[] = "Server-and-Client Test";
  blazingdb::communication::Buffer buffer(content, sizeof(content));

  std::shared_ptr<blazingdb::communication::NodeToken> nodeToken =
      std::make_shared<MockNodeToken>();

  std::shared_ptr<blazingdb::communication::Address> address =
      std::make_shared<MockAddress>();

  blazingdb::communication::Node node{std::move(nodeToken), std::move(address)};

  client->Send(node, buffer);

  server->Close();
  serverThread.join();
}
