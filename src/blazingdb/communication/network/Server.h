#ifndef BLAZINGDB_COMMUNICATION_NETWORK_SERVER_H_
#define BLAZINGDB_COMMUNICATION_NETWORK_SERVER_H_

#include <memory>
#include <sstream>

#include <blazingdb/communication/Message.h>

namespace blazingdb {
namespace communication {
namespace network {

class Frame {
public:
  virtual const std::string &data() const = 0;
  virtual std::streambuf &buffer() /*const*/ = 0;

  const std::string BufferString() /*const*/ {
    std::stringstream ss;
    ss << &buffer();
    return ss.str();
  }
};

class Server {
public:
  Server() = default;

  virtual std::shared_ptr<Frame> GetMessage() /*const*/ = 0;

  virtual void Run() = 0;
  virtual void Close() noexcept = 0;

  static std::unique_ptr<Server> Make();
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif
