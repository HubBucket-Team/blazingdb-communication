#include "MessageToken.h"

namespace blazgindb {
namespace communication {

MessageToken::MessageToken(const std::uint64_t rawToken)
    : rawToken_{rawToken} {}

}  // namespace communication
}  // namespace blazgindb
