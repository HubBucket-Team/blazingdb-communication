#include "Client.h"

namespace blazingdb {
namespace communication {
namespace network {

namespace {
class ConcreteClient : public Client {
public:
  void Send(const Node &node, const Buffer &buffer) const final {}

  void Send(const Node &node, const Message &message,
            const MessageSerializer &messageSerializer) const final {}
};
}  // namespace

std::unique_ptr<Client> Client::Make() { return nullptr; }

}  // namespace network
}  // namespace communication
}  // namespace blazingdb
