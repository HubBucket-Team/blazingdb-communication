#ifndef BLAZINGDB_COMMUNICATION_NETWORK_SERVER_H_
#define BLAZINGDB_COMMUNICATION_NETWORK_SERVER_H_

#include <memory>
#include <sstream>

#include <blazingdb/communication/Message.h>

namespace blazingdb {
namespace communication {
namespace network {

class Server {
public:
  Server() = default;

  virtual std::shared_ptr<Message> GetMessage() = 0;

  virtual void Run() = 0;
  virtual void Close() noexcept = 0;

  static std::unique_ptr<Server> Make();

  class Frame;
  virtual std::shared_ptr<Frame> GetFrame() /*const*/ = 0;
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif
