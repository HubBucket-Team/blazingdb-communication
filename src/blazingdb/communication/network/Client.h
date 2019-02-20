#ifndef BLAZINGDB_COMMUNICATION_NETWORK_CLIENT_H_
#define BLAZINGDB_COMMUNICATION_NETWORK_CLIENT_H_

#include <memory>

namespace blazingdb {
namespace communication {
namespace network {

class Client {
public:
  static std::unique_ptr<Client> Make();
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif
