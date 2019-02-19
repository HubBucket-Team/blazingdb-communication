#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_H_

#include <blazgindb/communication/MessageToken.h>

namespace blazgindb {
namespace communication {

class Message {
public:
  explicit Message(const MessageToken &messageToken);

private:
  MessageToken messageToken_;
};

}  // namespace communication
}  // namespace blazgindb

#endif
