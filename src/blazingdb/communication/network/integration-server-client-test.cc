#include "Client.h"
#include "ClientExceptions.h"
#include "Server.h"

#include <blazingdb/communication/messages/Message.h>

#include <chrono>
#include <memory>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

class MockAddress : public blazingdb::communication::Address {
public:
  MOCK_CONST_METHOD1(SameValueAs, bool(const Address &));
};

class MockMessage : public blazingdb::communication::messages::Message {
public:
  MockMessage(
      std::unique_ptr<blazingdb::communication::messages::MessageToken> &&messageToken,
      const std::size_t pages, const std::string &model)
      : Message{std::forward<
            std::unique_ptr<blazingdb::communication::messages::MessageToken>>(
            messageToken)},
        pages_{pages},
        model_{model} {}

  MOCK_CONST_METHOD0(serializeToJson, const std::string());
  MOCK_CONST_METHOD0(serializeToBinary, const std::string());

  static std::shared_ptr<Message> Make(const std::string &json_data,
                                       const std::string &binary_data) {
    const std::string expected_json = "{\"pages\": 12, \"model\": \"qwerty\"}";
    const std::string expected_binary = "";

    EXPECT_EQ(expected_json, json_data);
    EXPECT_EQ(expected_binary, binary_data);

    std::unique_ptr<blazingdb::communication::messages::MessageToken> messageToken =
        blazingdb::communication::messages::MessageToken::Make("sample");
    return std::shared_ptr<Message>(new MockMessage(std::move(messageToken), 12, "qwerty"));
  }

  std::size_t pages() const { return pages_; }

  const std::string model() const { return model_; }

private:
  const std::size_t pages_;
  const std::string model_;
};

class MockFlag {
public:
  MOCK_METHOD0(Flag, void());
};

TEST(IntegrationServerClientTest, SendMessageToServerFromClient) {
  // Alias
  using Server= blazingdb::communication::network::Server;
  using Message = blazingdb::communication::messages::Message;

  // Create endpoint
  const std::string endpoint {"sample"};

  // Create ContextToken
  Server::TokenValue context_token = 10;

  // Create server
  std::unique_ptr<Server> server = Server::Make();

  // Configure server
  server->registerEndPoint(endpoint, Server::Methods::Post);
  server->registerContext(context_token);
  server->registerDeserializer(endpoint, MockMessage::Make);

  // Run server
  std::thread serverRunThread([&server]() { server->Run(); });
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Create message
  std::unique_ptr<blazingdb::communication::messages::MessageToken> messageToken =
      blazingdb::communication::messages::MessageToken::Make("sample");

  MockMessage mockMessage{std::move(messageToken), 12, "qwerty"};

  const std::string json_data = "{\"pages\": 12, \"model\": \"qwerty\"}";
  const std::string binary_data = "";

  EXPECT_CALL(mockMessage, serializeToJson)
      .WillOnce(testing::Return(json_data));
  EXPECT_CALL(mockMessage, serializeToBinary)
      .WillOnce(testing::Return(binary_data));

  // Create node info
  std::shared_ptr<blazingdb::communication::Address> address =
      blazingdb::communication::Address::Make("localhost", 8000);

  blazingdb::communication::Node node{std::move(address)};

  // Send message
  std::unique_ptr<blazingdb::communication::network::Client> client =
      blazingdb::communication::network::Client::Make();
  try {
    std::unique_ptr<blazingdb::communication::network::Status> status =
        client->Send(node, endpoint, mockMessage);
    EXPECT_TRUE(status->IsOk());
  } catch (const blazingdb::communication::network::Client::SendError &e) {
    FAIL() << e.what();
  }

  // Get a frame
  MockFlag mockFlag;
  EXPECT_CALL(mockFlag, Flag()).Times(1);

  std::thread serverGetMessageThread([&server, &mockFlag, context_token]() {
    std::shared_ptr<Message> message = server->getMessage(context_token);

    auto mock_message = std::dynamic_pointer_cast<MockMessage>(message);
    EXPECT_EQ(12, mock_message->pages());
    EXPECT_EQ("qwerty", mock_message->model());
    mockFlag.Flag();
  });

  // Clean context
  serverGetMessageThread.join();
  server->Close();
  serverRunThread.join();
}
