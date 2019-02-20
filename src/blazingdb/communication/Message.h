#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_H_

#include <memory>

#include <blazingdb/communication/MessageToken.h>

namespace blazingdb {
namespace communication {

class Message {
public:
  explicit Message(const std::shared_ptr<MessageToken> &messageToken);

  MessageToken getMessageToken();

private:
  std::shared_ptr<MessageToken> messageToken_;
};

}  // namespace communication
}  // namespace blazgindb

#endif
