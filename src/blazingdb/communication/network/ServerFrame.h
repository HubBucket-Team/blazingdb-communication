#ifndef BLAZINGDB_COMMUNICATION_NETWORK_SERVER_FRAME_H_
#define BLAZINGDB_COMMUNICATION_NETWORK_SERVER_FRAME_H_

#include <blazingdb/communication/network/Server.h>

namespace blazingdb {
namespace communication {
namespace network {

class Server::Frame {
public:
  virtual const std::string &data() const = 0;
  virtual std::streambuf &buffer() /*const*/ = 0;

  const std::string BufferString() /*const*/ {
    std::stringstream ss;
    ss << &buffer();
    return ss.str();
  }
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif
