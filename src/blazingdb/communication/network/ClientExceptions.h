#ifndef BLAZINGDB_COMMUNICATION_NETWORK_CLIENT_EXCEPTIONS_H_
#define BLAZINGDB_COMMUNICATION_NETWORK_CLIENT_EXCEPTIONS_H_

#include "Client.h"

namespace blazingdb {
namespace communication {
namespace network {

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
