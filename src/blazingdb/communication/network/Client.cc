#include "Client.h"

#include <simple-web-server/client_http.hpp>

namespace blazingdb {
namespace communication {
namespace network {

namespace {
class ConcreteClient : public Client {
public:
  using HttpClient = SimpleWeb::Client<SimpleWeb::HTTP>;

  void Send(const Node &node, const Buffer &buffer) const final {

  }

  void Send(const Node &node, const Message &message,
            const MessageSerializer &messageSerializer) const final {}

private:
  HttpClient httpClient{"localhost:8000"};
};
}  // namespace

std::unique_ptr<Client> Client::Make() {
  return std::unique_ptr<Client>(new ConcreteClient);
}

}  // namespace network
}  // namespace communication
}  // namespace blazingdb
