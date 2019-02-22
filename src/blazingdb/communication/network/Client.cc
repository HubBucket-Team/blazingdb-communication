#include "Client.h"
#include "ClientExceptions.h"

#include <blazingdb/communication/Address-Internal.h>

#include <iostream>
#include <map>

#include <simple-web-server/client_http.hpp>

namespace blazingdb {
namespace communication {
namespace network {

namespace {
class ConcreteClient : public Client {
public:
  using HttpClient = SimpleWeb::Client<SimpleWeb::HTTP>;

  void Send(const Node &node, const std::string &endpoint,
            const std::string &data,
            const std::string &buffer) /*const*/ final {
    const internal::ConcreteAddress *concreteAddress =
        static_cast<const internal::ConcreteAddress *>(node.address().get());

    const std::string serverPortPath =
        concreteAddress->ip() + ":" + std::to_string(concreteAddress->port());

    HttpClient httpClient{serverPortPath};

    std::string body{reinterpret_cast<const char *>(buffer.data()),
                     buffer.size()};

    std::map<std::string, std::string> headers{{"json_data", data}};

    try {
      std::shared_ptr<HttpClient::Response> response =
          httpClient.request("POST", "/" + endpoint, body, headers);
      std::cout << response->content.rdbuf() << std::endl;
    } catch (const boost::system::system_error &error) {
      throw SendError(endpoint);
    }
  }

  void Send(const Node &node, const std::string &endpoint,
            const Message &message) final {
    Send(node, endpoint, message.serializeToJson(),
         message.serializeToBinary());
  }

  void SendNodeData(std::string ip, uint16_t port, const Buffer &buffer) final {

  }
};
}  // namespace

std::unique_ptr<Client> Client::Make() {
  return std::unique_ptr<Client>(new ConcreteClient);
}

}  // namespace network
}  // namespace communication
}  // namespace blazingdb
