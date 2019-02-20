#ifndef BLAZINGDB_COMMUNICATION_ADDRESS_H_
#define BLAZINGDB_COMMUNICATION_ADDRESS_H_

#include <string>

namespace blazingdb {
namespace communication {

class Address {
public:
  explicit Address(std::string ip, int port);
  Address(const Address& other) = default;
  std::string getIp() const;
  int getPort() const;

private:
  std::string ip_;
  int port_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
