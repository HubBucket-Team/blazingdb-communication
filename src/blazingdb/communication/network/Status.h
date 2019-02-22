#ifndef BLAZINGDB_COMMUNICATION_NETWORK_STATUS_H_
#define BLAZINGDB_COMMUNICATION_NETWORK_STATUS_H_

#include <string>

namespace blazingdb {
namespace communication {
namespace network {

class Status {
public:
  virtual bool IsOk() const noexcept = 0;
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif
