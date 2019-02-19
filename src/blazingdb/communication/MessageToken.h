#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_TOKEN_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_TOKEN_H_

#include <cstdint>

namespace blazingdb {
namespace communication {

class MessageToken {
public:
  explicit MessageToken(const std::uint64_t rawToken);

private:
  std::uint64_t rawToken_;
};

}  // namespace communication
}  // namespace blazgindb

#endif
