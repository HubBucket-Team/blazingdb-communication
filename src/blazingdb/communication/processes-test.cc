#include <blazingdb/communication/Address-Internal.h>
#include <blazingdb/communication/messages/Message.h>
#include <blazingdb/communication/messages/MessageToken.h>
#include <blazingdb/communication/network/Client.h>
#include <blazingdb/communication/network/ClientExceptions.h>
#include <blazingdb/communication/network/Server.h>
#include <blazingdb/uc/API.h>

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

  void
  CreateRemoteBuffer(const Node &node) const final {
    auto &concreteAddress = *static_cast<
        const blazingdb::communication::internal::ConcreteAddress *>(
        node.address());

    auto context = blazingdb::uc::Context::IPC(concreteAddress.trader());

    auto ownAgent  = context->OwnAgent();
    auto peerAgent = context->PeerAgent();

    const std::size_t length = 100;

    const void *ownData  = Malloc("ownText");
    const void *peerData = Malloc("peerText");

    auto ownBuffer  = ownAgent->Register(ownData, length);
    auto peerBuffer = peerAgent->Register(peerData, length);

    ownBuffer->Link(peerBuffer.get());
  }
};
}  // namespace

static void
ExecServer() {
  using namespace blazingdb::communication::messages;
  using namespace blazingdb::communication::network;

  cuInit(0);
  std::unique_ptr<Server> server = Server::Make();

  std::thread serverThread{&Server::Run, server.get(), 8000};
  std::this_thread::sleep_for(std::chrono::seconds(1));

  std::shared_ptr<Address> address = Address::Make("127.0.0.1", 8001);
  Node                     node{std::move(address)};
  std::shared_ptr<Message> message = server->getMessage(1243, node);

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

  std::shared_ptr<ContextToken> contextToken = ContextToken::Make(1243);
  std::unique_ptr<MessageToken> messageToken = MessageToken::Make("id");
  MockMessage message{std::move(contextToken), std::move(messageToken)};

  EXPECT_CALL(message, serializeToJson).WillOnce(testing::Return(""));
  EXPECT_CALL(message, serializeToBinary).WillOnce(testing::Return(""));
  testing::Mock::AllowLeak(&message);

  try {
    const std::shared_ptr<Status> status = client->Send(node, "id", message);
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
