#include "Client.h"

#include <iostream>

#include <simple-web-server/client_http.hpp>

namespace blazingdb {
namespace communication {
namespace network {

namespace {
class ConcreteClient : public Client {
public:
  using HttpClient = SimpleWeb::Client<SimpleWeb::HTTP>;

  void Send(const Node &node, const Buffer &buffer) /*const*/ final {
    std::string body{reinterpret_cast<const char *>(buffer.data()),
                     buffer.size()};
      auto request = httpClient_.request("POST", "/ehlo", body);
      std::cout << request->content.rdbuf() << std::endl;
  }

  void Send(const Node &node, const Message &message,
            const MessageSerializer &messageSerializer) /*const*/ final {
    Send(node, messageSerializer.serialize(message));
  }

  void SendNodeData(std::string ip, uint16_t port, const Buffer &buffer) final {

  }

private:
  HttpClient httpClient_{"localhost:8000"};
};
}  // namespace

std::unique_ptr<Client> Client::Make() {
  return std::unique_ptr<Client>(new ConcreteClient);
}

}  // namespace network
}  // namespace communication
}  // namespace blazingdb
