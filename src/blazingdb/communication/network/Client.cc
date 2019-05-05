#include "Client.h"
#include "ClientExceptions.h"
#include "Status.h"

#include <blazingdb/communication/Address-Internal.h>

#include <map>

#include <simple-web-server/client_http.hpp>

#include <blazingdb/uc/UCPool.hpp>

namespace blazingdb {
namespace communication {
namespace network {

namespace {
using HttpClient = SimpleWeb::Client<SimpleWeb::HTTP>;
using messages::Message;

class ConcreteStatus : public Status {
public:
  ConcreteStatus(const std::shared_ptr<HttpClient::Response> &response)
      : response_{response} {}

  bool IsOk() const noexcept { return "200 OK" == response_->status_code; }

  const std::string ToString() const noexcept {
    std::ostringstream oss;
    oss << response_->content.rdbuf();
    return oss.str();
  }

private:
  std::shared_ptr<HttpClient::Response> response_;
};

class ConcreteClient : public Client {
public:
  std::shared_ptr<Status> Send(const Node &node, const std::string &endpoint,
                               const std::string &data,
                               const std::string &buffer) /*const*/ final {
    const internal::ConcreteAddress *concreteAddress =
        static_cast<const internal::ConcreteAddress *>(node.address());

    const std::string serverPortPath =
        concreteAddress->ip() + ":" + std::to_string(concreteAddress->port());

    HttpClient httpClient{serverPortPath};

    std::map<std::string, std::string> headers{{"json_data", data}};

    try {
      std::shared_ptr<HttpClient::Response> response =
          httpClient.request("POST", "/message/" + endpoint, buffer, headers);
      return std::shared_ptr<Status>(new ConcreteStatus{response});
    } catch (const std::exception& e) {
      throw SendError(e.what(), endpoint, data, buffer.size());
    }
  }

  std::shared_ptr<Status> Send(const Node &node, const std::string &endpoint,
                               const Message &message) final {
    return Send(node, endpoint, message.serializeToJson(),
                message.serializeToBinary());
  }

  std::shared_ptr<Status> send(const Node& node,
                                 std::shared_ptr<messages::Message>& message) override {

        const auto server_address = getAddress(node);
        HttpClient httpClient{server_address};


        auto &concreteAddress = *static_cast<
              const blazingdb::communication::internal::ConcreteAddress *>(
              node.address());

        auto context = blazingdb::uc::Context::IPC();

        auto agent  = context->Agent();

        const std::string head_json = message->serializeToJson();
        const std::string buffer_descriptors = message->serializeToBinary(agent.get());

        UCPool::getInstance().push(agent.release());
        UCPool::getInstance().push(context.release());

        std::map<std::string, std::string> headers{{"json_data", head_json}};

        return sendPost(httpClient, message->getMessageTokenValue(), headers, buffer_descriptors);
    }

  std::shared_ptr<Status> SendNodeData(const std::string &ip,
                                       const std::uint16_t port,
                                       const Message &message) final {
    const std::string serverPortPath = ip + ":" + std::to_string(port);
    HttpClient httpClient{serverPortPath};

    std::map<std::string, std::string> headers{
        {"json_data", message.serializeToJson()}};

    try {
      std::shared_ptr<HttpClient::Response> response =
          httpClient.request("POST", "/register_node", "", headers);
      return std::shared_ptr<Status>(new ConcreteStatus{response});
    } catch (const std::exception& e) {
      const std::string data = message.serializeToJson();
      throw SendError(e.what(), "/register_node", data, data.size());
    }
  }

    const std::string getAddress(const Node& node) {
        const auto* concreteAddress = static_cast<const internal::ConcreteAddress*>(node.address());
        return std::string{concreteAddress->ip() + ":" + std::to_string(concreteAddress->port())};
    }

    std::shared_ptr<Status> sendPost(HttpClient& httpClient,
                                     const std::string& endpoint,
                                     const std::map<std::string, std::string>& headers,
                                     const std::string& body) {
        try {
            std::shared_ptr<HttpClient::Response> response = httpClient.request("POST", "/message/" + endpoint, body, headers);

            //
            //  concreteAddress = toConcrete(node.address)
            //
            //  serializeToBinary(node.address.trader())
            //     c = Context(trader)
            //     sender = c.OwnAgent()
            //     receiver = c.PeerAgent()
            //
            //     sendBuf = sender.Register(MyPtr, MySize);
            //     recvBuf = receiver.Register(nullptr, 0);
            //
            //     sendBuf.Link(recvBuf)
            //
            //     // send message

            // context_pointer_pool
            return std::shared_ptr<Status>(new ConcreteStatus{response});
        }
        catch (const std::exception& e) {
            const std::string& data = headers.at("json_data");
            throw SendError(e.what(), "/message/" + endpoint, data, data.size());
        }
    }
};
}  // namespace

std::unique_ptr<Client> Client::Make() {
  return std::unique_ptr<Client>(new ConcreteClient);
}


    // std::shared_ptr<Client::Status> Client::send(const Node& node,
    //                                              std::shared_ptr<Message>& message) {
        // auto client = blazingdb::communication::network::Client::Make();
    //     return client->send(node, message);
    // }

    // std::shared_ptr<Client::Status> Client::sendNodeData(const std::string& orchestratorIp,
    //                                                     int16_t orchestratorPort,
    //                                                     std::shared_ptr<Message>& message) {
    //     auto client = blazingdb::communication::network::Client::Make();
    //     return client->SendNodeData(orchestratorIp, orchestratorPort, *message);
    // }


}  // namespace network
}  // namespace communication
}  // namespace blazingdb
