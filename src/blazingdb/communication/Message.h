#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_H_

#include <blazingdb/communication/MessageToken.h>

namespace blazingdb {
namespace communication {

class Message {
public:
  explicit Message(const MessageToken &messageToken);

  MessageToken getMessageToken();

private:
  MessageToken messageToken_;
};

}  // namespace communication
}  // namespace blazgindb

#endif
