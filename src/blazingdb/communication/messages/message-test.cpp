#include <blazingdb/communication/messages/Message.h>
#include <blazingdb/communication/network/Client.h>
#include <blazingdb/communication/network/ClientExceptions.h>
#include <blazingdb/communication/network/Server.h>

#include <blazingdb/uc/API.hpp>

#include <cuda.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

static constexpr std::size_t length = 100;

static void *
Malloc() {
  cudaError_t cudaStatus;

  void *data = nullptr;

  cudaStatus = cudaMalloc(&data, length);
  assert(cudaSuccess == cudaStatus);

  std::uint8_t content[length];
  for (std::size_t i = 0; i < length; i++) { content[i] = i; }

  cudaStatus = cudaMemcpy(data, content, length, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStatus);

  return data;
}

class DataContainer {
public:
  DataContainer() : context_{blazingdb::uc::Context::IPC()} {
    auto agent = context_->Agent();
    const void* d_ptr = (const void* )Malloc();
    buffers_.emplace_back(agent->Register(d_ptr, length));

    data_.resize(buffers_.size() * 104);

    std::memcpy(&data_[0], buffers_[0]->SerializedRecord()->Data(), 104);
  }

  const std::string &
  data() const noexcept {
    return data_;
  }

private:
  std::unique_ptr<blazingdb::uc::Context>             context_;
  std::vector<std::unique_ptr<blazingdb::uc::Buffer>> buffers_;
  std::string                                         data_;
};

class MockMessage : public blazingdb::communication::messages::Message {
public:
  MockMessage(
      std::shared_ptr<blazingdb::communication::ContextToken> &&contextToken,
      std::unique_ptr<blazingdb::communication::messages::MessageToken>
          &&messageToken)
      : Message{std::forward<std::unique_ptr<
                    blazingdb::communication::messages::MessageToken>>(
                    messageToken),
                std::move(contextToken)} {}

  MOCK_CONST_METHOD0(serializeToJson, const std::string());
  MOCK_CONST_METHOD0(serializeToBinary, const std::string());

  static std::shared_ptr<Message>
  Make(const std::string & /*jsonData*/, const std::string &binaryData) {
    std::cout << ">>> " << binaryData << std::endl;
    return MockMessage::CreateMessage();
  }
  
  static std::shared_ptr<MockMessage>
  CreateMessage() {
    using blazingdb::communication::ContextToken;
    using blazingdb::communication::messages::MessageToken;
    using blazingdb::communication::network::Server;
  
    static constexpr char                      endpoint[]     = "testEndpoint";
    static constexpr Server::ContextTokenValue contextTokenId = 1230;
  
    auto messageToken = MessageToken::Make(endpoint);
    auto contextToken = ContextToken::Make(contextTokenId);
  
    return std::make_shared<MockMessage>(std::move(contextToken),
                                         std::move(messageToken));
  }
};

static std::shared_ptr<blazingdb::communication::Node>
CreateNode() {
  auto address = blazingdb::communication::Address::Make("127.0.0.1", 8000, 5678);
  return blazingdb::communication::Node::Make(std::move(address));
}

TEST(MessageTest, SerializingWithUCX) {
  using blazingdb::communication::network::Server;

  static constexpr char                      endpoint[]   = "testEndpoint";
  static constexpr Server::ContextTokenValue contextToken = 1230;

  cuInit(0);
  auto server = Server::Make();
  server->registerEndPoint(endpoint, Server::Methods::Post);
  server->registerContext(contextToken);
  server->registerDeserializer(endpoint, MockMessage::Make);

  std::thread serverRunThread{&Server::Run, server.get(), 8000};
  std::this_thread::sleep_for(std::chrono::seconds(1));

  using blazingdb::communication::network::Client;
  auto client  = Client::Make();
  auto node    = CreateNode();
  auto message = MockMessage::CreateMessage();

  EXPECT_CALL(*message, serializeToJson()).WillOnce(testing::Return(""));
  EXPECT_CALL(*message, serializeToBinary()).WillOnce(testing::Return(""));

  try {
    auto status = client->Send(*node, endpoint, *message);
    EXPECT_TRUE(status->IsOk());
  } catch (const Client::SendError &error) { FAIL() << error.what(); }

  std::thread serverGetMessageThread([&server]() {
    auto message = server->getMessage(contextToken, "testEndpoint");
  });

  serverGetMessageThread.join();
  server->Close();
  serverRunThread.join();
}
