#include "Message.h"

namespace blazingdb {
namespace communication {

Message::Message(const std::shared_ptr<MessageToken> &messageToken)
    : messageToken_{messageToken} {}

Message::~Message() = default;

}  // namespace communication
}  // namespace blazingdb
