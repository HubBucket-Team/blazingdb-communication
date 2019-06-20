#ifndef BLAZINGDB_COMMUNICATION_ADDRESS_INTERNAL_H_
#define BLAZINGDB_COMMUNICATION_ADDRESS_INTERNAL_H_

#include "Address.h"
#include "internal/Trader.hpp"

namespace blazingdb {
namespace communication {
namespace internal {

class ConcreteAddress : public Address {
public:
  ConcreteAddress(const std::string &ip, const std::int16_t communication_port, const std::int16_t protocol_port)
      : ip_{std::move(ip)}, communication_port_{communication_port}, protocol_port_{protocol_port} {}

  const std::string &
  ip() const noexcept {
    return ip_;
  }
  
  bool SameValueAs(const Address &address) const final {
    const ConcreteAddress &concreteAddress =
        *static_cast<const ConcreteAddress *>(&address);
    return (ip_ == concreteAddress.ip_) && (communication_port_ == concreteAddress.communication_port_) && (protocol_port_ == concreteAddress.protocol_port_);
  }

  const blazingdb::uc::Trader &
  trader() const noexcept {
    return trader_;
  }

  std::int16_t communication_port() const noexcept { return communication_port_; }
  
  std::int16_t protocol_port() const noexcept { return protocol_port_; }

  void serializeToJson(JsonSerializable::Writer& writer) const {
      writer.Key("addressIp");
      writer.String(ip_.c_str());

      writer.Key("addressCommunicationPort");
      writer.Int(communication_port_);
      
      writer.Key("addressProtocolPort");
      writer.Int(protocol_port_);
  };

private:
  Trader trader_;
  const std::string ip_;
  const std::int16_t communication_port_;
  const std::int16_t protocol_port_; // For TCP socket // TODO percy c.gonzales JP
};

}  // namespace internal
}  // namespace communication
}  // namespace blazingdb

#endif
