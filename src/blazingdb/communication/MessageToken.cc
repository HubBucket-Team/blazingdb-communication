#include "MessageToken.h"

#include <cstdint>

namespace blazingdb {
namespace communication {

class Int64MessageToken : public MessageToken {
public:
  explicit Int64MessageToken(const std::uint64_t int64Token)
      : int64Token_{int64Token} {}

  bool Is(const MessageToken &other) const noexcept {
    return int64Token_ ==
           static_cast<const Int64MessageToken &>(other).int64Token_;
  }

private:
  std::uint64_t int64Token_;
};

std::unique_ptr<MessageToken> MessageToken::Make() {
  static std::uint64_t counter = 0;

  return std::unique_ptr<MessageToken>(new Int64MessageToken{counter++});
}

}  // namespace communication
}  // namespace blazingdb
