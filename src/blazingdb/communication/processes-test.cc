#include <blazingdb/communication/Address-Internal.h>
#include <blazingdb/communication/messages/Message.h>
#include <blazingdb/communication/messages/MessageToken.h>
#include <blazingdb/communication/network/Client.h>
#include <blazingdb/communication/network/ClientExceptions.h>
#include <blazingdb/communication/network/Server.h>
#include <blazingdb/uc/API.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

static const void *
Malloc(const std::string &&payload) {
  void *data;

  cudaError_t cudaError;

  cudaError = cudaMalloc(&data, payload.length() + 100);
  assert(cudaSuccess == cudaError);

  cudaError = cudaMemcpy(
      data, payload.data(), payload.length(), cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaError);

  return data;
}

namespace {
using namespace blazingdb::communication;

class MockAddress : public Address {
public:
  MOCK_CONST_METHOD1(SameValueAs, bool(const Address &));
};

class MockMessage : public messages::Message {
public:
  MockMessage(std::shared_ptr<ContextToken> &&contextToken,
              std::unique_ptr<blazingdb::communication::messages::MessageToken>
                  &&messageToken)
      : Message{std::forward<std::unique_ptr<
                    blazingdb::communication::messages::MessageToken>>(
                    messageToken),
                std::move(contextToken)} {}

  MOCK_CONST_METHOD0(serializeToJson, const std::string());
  MOCK_CONST_METHOD0(serializeToBinary, const std::string());

  static std::shared_ptr<Message>
  Make(const std::string & /*jsonData*/, const std::string & /*binaryData*/) {
    return nullptr;
  }
};

class DataContainer {
public:
  DataContainer() : context_{blazingdb::uc::Context::IPC()} {
    agent_ = context_->Agent();
    buffers_.emplace_back(agent_->Register(Malloc("ownData"), 108));

    data_.resize(buffers_.size() * 104);

    std::memcpy(&data_[0], buffers_[0]->SerializedRecord()->Data(), 104);
  }

  const std::string &
  data() const noexcept {
    return data_;
  }

private:
  std::unique_ptr<blazingdb::uc::Context>             context_;
  std::unique_ptr<blazingdb::uc::Agent>               agent_;
  std::vector<std::unique_ptr<blazingdb::uc::Buffer>> buffers_;
  std::string                                         data_;
};
}  // namespace

static constexpr char endpoint[] = "testEndpoint";
static constexpr blazingdb::communication::network::Server::ContextTokenValue
    contextTokenValueId = 1230;

static void
ExecServer() {
  using namespace blazingdb::communication::messages;
  using namespace blazingdb::communication::network;

  cuInit(0);
  std::unique_ptr<Server> server = Server::Make();
  server->registerEndPoint(endpoint, Server::Methods::Post);
  server->registerContext(contextTokenValueId);
  server->registerDeserializer(endpoint, MockMessage::Make);

  std::thread serverThread{&Server::Run, server.get(), 8000};
  std::this_thread::sleep_for(std::chrono::seconds(1));

  std::shared_ptr<Address> address = Address::Make("127.0.0.1", 8001);
  Node                     node{std::move(address)};
  std::shared_ptr<Message> message = server->getMessage(contextTokenValueId);

  server->Close();
  serverThread.join();
}

static void
ExecClient() {
  using namespace blazingdb::communication;
  using namespace blazingdb::communication::messages;
  using namespace blazingdb::communication::network;

  cuInit(0);
  std::unique_ptr<Server> server = Server::Make();

  std::thread serverThread{&Server::Run, server.get(), 8001};
  std::this_thread::sleep_for(std::chrono::seconds(1));

  std::unique_ptr<Client> client = Client::Make();

  std::shared_ptr<Address> address = Address::Make("127.0.0.1", 8000);
  Node                     node{std::move(address)};

  std::shared_ptr<ContextToken> contextToken =
      ContextToken::Make(contextTokenValueId);
  std::unique_ptr<MessageToken> messageToken = MessageToken::Make(endpoint);
  MockMessage message{std::move(contextToken), std::move(messageToken)};

  DataContainer dataContainer;

  EXPECT_CALL(message, serializeToJson()).WillOnce(testing::Return(""));
  EXPECT_CALL(message, serializeToBinary())
      .WillOnce(testing::Return(dataContainer.data()));
  testing::Mock::AllowLeak(&message);

  try {
    const std::shared_ptr<Status> status =
        client->Send(node, endpoint, message);
    EXPECT_TRUE(status->IsOk());
  } catch (const Client::SendError &e) { FAIL() << e.what(); }

  std::this_thread::sleep_for(std::chrono::seconds(1));
  server->Close();
  serverThread.join();
}

TEST(DISABLED_ProcessesTest, TwoProcesses) {
  pid_t pid = fork();
  if (pid) {
    ExecServer();
  } else {
    ExecClient();
    std::exit(0);
  }
}
