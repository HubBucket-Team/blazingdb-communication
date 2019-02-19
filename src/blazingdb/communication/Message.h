#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_H_

#include <blazingdb/communication/MessageToken.h>

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
