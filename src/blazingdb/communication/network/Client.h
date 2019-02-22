#ifndef BLAZINGDB_COMMUNICATION_NETWORK_CLIENT_H_
#define BLAZINGDB_COMMUNICATION_NETWORK_CLIENT_H_

#include <exception>
#include <memory>

#include <blazingdb/communication/Buffer.h>
#include <blazingdb/communication/Message.h>
#include <blazingdb/communication/Node.h>

namespace blazingdb {
namespace communication {
namespace network {

class Client {
public:
  class SendError;

  virtual void Send(const Node &node, const std::string &endpoint,
                    const std::string &data,
                    const std::string &buffer) /*const*/
      = 0;

  virtual void Send(const Node &node, const std::string &endpoint,
                    const Message &message) = 0;

  virtual void SendNodeData(std::string ip, uint16_t port,
                            const Buffer &buffer) /*const*/
      = 0;

  static std::unique_ptr<Client> Make();
};

class Client::SendError {
public:
  SendError(const std::string &endpoint) : endpoint_{endpoint} {}

  const char *what() const noexcept {
    return ("Communication::Client: Bad endpoint \"" + endpoint_ + "\"")
        .c_str();
  }

private:
  const std::string endpoint_;
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif
