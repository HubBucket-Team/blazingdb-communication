#include "Client.h"
#include "ClientExceptions.h"
#include "Server.h"
#include "ServerFrame.h"

#include <blazingdb/communication/Message.h>

#include <chrono>
#include <memory>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

class MockNodeToken : public blazingdb::communication::NodeToken {
public:
  MOCK_CONST_METHOD1(SameValueAs, bool(const NodeToken &));
  MOCK_CONST_METHOD1(serializeToJson, void(JsonSerializable::Writer &));
};

class MockAddress : public blazingdb::communication::Address {
public:
  MOCK_CONST_METHOD1(SameValueAs, bool(const Address &));
};

class MockMessage : public blazingdb::communication::Message {
public:
  using blazingdb::communication::Message::Message;

  MOCK_CONST_METHOD0(serializeToJson, const std::string());
  MOCK_CONST_METHOD0(serializeToBinary, const std::string());
};

TEST(IntegrationServerClientTest, SendMessageToServerFromClient) {
  // Run server
  std::unique_ptr<blazingdb::communication::network::Server> server =
      blazingdb::communication::network::Server::Make();

  std::thread serverRunThread([&server]() { server->Run(); });

  std::this_thread::sleep_for(std::chrono::seconds(1));

  std::thread serverGetMessageThread([&server]() {
    std::shared_ptr<blazingdb::communication::network::Server::Frame> frame =
        server->GetFrame();

    EXPECT_EQ("{}", frame->data());
    EXPECT_EQ("data", frame->BufferString());
  });

  // Create message
  std::unique_ptr<blazingdb::communication::MessageToken> messageToken =
      blazingdb::communication::MessageToken::Make("sample");

  MockMessage mockMessage{std::move(messageToken)};

  EXPECT_CALL(mockMessage, serializeToJson).WillOnce(testing::Return("{}"));
  EXPECT_CALL(mockMessage, serializeToBinary).WillOnce(testing::Return("data"));

  // Create node info
  std::shared_ptr<blazingdb::communication::NodeToken> nodeToken =
      std::make_shared<MockNodeToken>();

  std::shared_ptr<blazingdb::communication::Address> address =
      blazingdb::communication::Address::Make("localhost", 8000);

  blazingdb::communication::Node node{std::move(nodeToken), std::move(address)};

  // Send message
  std::unique_ptr<blazingdb::communication::network::Client> client =
      blazingdb::communication::network::Client::Make();
  try {
    std::unique_ptr<blazingdb::communication::network::Status> status =
        client->Send(node, "sample", mockMessage);
    EXPECT_TRUE(status->IsOk());
  } catch (const blazingdb::communication::network::Client::SendError &e) {
    FAIL() << e.what();
  }

  serverGetMessageThread.join();
  server->Close();
  serverRunThread.join();
}
