#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_H_

#include <memory>

#include <blazingdb/communication/MessageToken.h>
#include <blazingdb/communication/shared/Entity.h>

namespace blazingdb {
namespace communication {

class Message : public Entity<Message, MessageToken> {
public:
  explicit Message(const std::shared_ptr<MessageToken> &messageToken);

  ~Message();

private:
  std::shared_ptr<MessageToken> messageToken_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
