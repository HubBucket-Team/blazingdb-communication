#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_TOKEN_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_TOKEN_H_

#include <cstdint>

#include <blazingdb/communication/shared/Identity.h>

namespace blazingdb {
namespace communication {

class MessageToken : public Identity<MessageToken> {
public:
  explicit MessageToken(const std::uint64_t rawToken);


  bool SameAs(const MessageToken &other) const noexcept final;

  std::uint64_t getToken();

private:
  const std::uint64_t rawToken_;
};

}  // namespace communication
}  // namespace blazgindb

#endif
