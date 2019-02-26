#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_H_

#include <memory>

#include <blazingdb/communication/messages/MessageToken.h>
#include <blazingdb/communication/shared/Entity.h>

namespace blazingdb {
namespace communication {
namespace messages {

class Message {
public:
  explicit Message(std::unique_ptr<MessageToken> &&messageToken);

  ~Message();

  virtual const std::string serializeToJson() const = 0;
  virtual const std::string serializeToBinary() const = 0;

private:
  const std::unique_ptr<MessageToken> messageToken_;
};

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
