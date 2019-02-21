#include "Message.h"

namespace blazingdb {
namespace communication {

Message::Message(std::unique_ptr<MessageToken> &&messageToken)
    : messageToken_{std::move(messageToken)} {}

Message::~Message() = default;

}  // namespace communication
}  // namespace blazingdb
