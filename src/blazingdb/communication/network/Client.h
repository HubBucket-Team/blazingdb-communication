#ifndef BLAZINGDB_COMMUNICATION_NETWORK_CLIENT_H_
#define BLAZINGDB_COMMUNICATION_NETWORK_CLIENT_H_

#include <memory>

#include <blazingdb/communication/Buffer.h>
#include <blazingdb/communication/MessageSerializer.h>
#include <blazingdb/communication/Node.h>

namespace blazingdb {
namespace communication {
namespace network {

class Client {
public:
  // TODO: cualifier has a internal discard
  virtual void Send(const Node &node, const Buffer &buffer) /*const*/ = 0;
  virtual void Send(const Node &node, const Message &message,
                    const MessageSerializer &messageSerializer) /*const*/ = 0;

  static std::unique_ptr<Client> Make();
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif
