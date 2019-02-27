#ifndef BLAZINGDB_COMMUNICATION_NETWORK_CLIENT_EXCEPTIONS_H_
#define BLAZINGDB_COMMUNICATION_NETWORK_CLIENT_EXCEPTIONS_H_

#include <exception>
#include "Client.h"

namespace blazingdb {
namespace communication {
namespace network {

class Client::SendError : public std::exception {
public:
  SendError(const std::string &endpoint, const std::string &data,
            const std::size_t bufferSize)
      : endpoint_{endpoint}, data_{data}, bufferSize_{bufferSize} {}

  const char* what() const noexcept override {
    return ("Communication::Client: Bad endpoint \"" + endpoint_ +
            "\" with buffer size = " + std::to_string(bufferSize_) +
            " and data = " + data_)
        .c_str();
  }

private:
  const std::string endpoint_;
  const std::string data_;
  const std::size_t bufferSize_;
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif
