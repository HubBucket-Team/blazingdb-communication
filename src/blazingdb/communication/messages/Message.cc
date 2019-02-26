#include "Message.h"

namespace blazingdb {
namespace communication {
namespace messages {

Message::Message(std::unique_ptr<MessageToken> &&messageToken)
    : messageToken_{std::move(messageToken)} {}

Message::~Message() = default;

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
