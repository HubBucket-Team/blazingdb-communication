#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_H_

#include <memory>

#include <blazingdb/communication/MessageToken.h>
#include <blazingdb/communication/shared/Entity.h>

namespace blazingdb {
namespace communication {

class Message : public Entity<Message, MessageToken> {
public:
  explicit Message(std::unique_ptr<MessageToken> &&messageToken);

  ~Message();

private:
  const std::unique_ptr<MessageToken> messageToken_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
