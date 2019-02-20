#include "MessageToken.h"

namespace blazingdb {
namespace communication {

MessageToken::MessageToken(const std::uint64_t rawToken)
    : rawToken_{rawToken} {}

bool MessageToken::SameAs(const MessageToken &other) const noexcept {
    return rawToken_ == other.rawToken_;
}

std::uint64_t MessageToken::getToken() {
    return rawToken_;
}

}  // namespace communication
}  // namespace blazingdb
