#include "Message.h"

#include "../internal/Trader.hpp"
#include "../Address-Internal.h"
#include <blazingdb/uc/API.hpp>
#include <cuda_runtime_api.h>

namespace blazingdb {
namespace communication {
namespace messages {

Message::Message(std::unique_ptr<MessageToken> &&messageToken,
                 std::shared_ptr<ContextToken> &&contextToken)
: messageToken_{std::move(messageToken)},
  contextToken_{std::move(contextToken)}
{ }

Message::Message(std::unique_ptr<MessageToken>&& messageToken,
                 std::shared_ptr<ContextToken>&& contextToken,
                 const Node& sender_node)
: messageToken_{std::move(messageToken)},
  contextToken_{std::move(contextToken)},
  sender_node_{sender_node}
{ }

Message::~Message() = default;

ContextToken::TokenType Message::getContextTokenValue() const {
    return contextToken_->getIntToken();
}

const MessageToken::TokenType Message::getMessageTokenValue() const {
    return messageToken_->toString();
}

const blazingdb::communication::Node& Message::getSenderNode() const {
    return sender_node_;
}

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

void
Message::GetRemoteBuffer(const Node &node) {
  auto &concreteAddress =
      *static_cast<const blazingdb::communication::internal::ConcreteAddress *>(
          node.address());

  auto context = blazingdb::uc::Context::IPC(concreteAddress.trader());

  auto ownAgent  = context->OwnAgent();
  auto peerAgent = context->PeerAgent();

  const std::size_t length = 100;

  const void *ownData  = Malloc("ownText");
  const void *peerData = Malloc("peerText");

  auto ownBuffer  = ownAgent->Register(ownData, length);
  auto peerBuffer = peerAgent->Register(peerData, length);

  auto transport = ownBuffer->Link(peerBuffer.get());

  auto future = transport->Get();
  future.wait();

}

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
