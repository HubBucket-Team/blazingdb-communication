#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_H_

#include <memory>

#include <blazingdb/communication/messages/MessageToken.h>
#include <blazingdb/communication/shared/Entity.h>
#include "blazingdb/communication/ContextToken.h"

namespace blazingdb {
namespace communication {
namespace messages {

class Message {
public:
  explicit Message(std::unique_ptr<MessageToken> &&messageToken,
                   std::unique_ptr<ContextToken> &&contextToken);

  ~Message();

  virtual const std::string serializeToJson() const = 0;
  virtual const std::string serializeToBinary() const = 0;

public:
    const ContextToken::TokenType getContextTokenValue() const;
    const MessageToken::TokenType getMessageTokenValue() const;

private:
  const std::unique_ptr<MessageToken> messageToken_;
  const std::unique_ptr<ContextToken> contextToken_;

private:
    template <typename Writer>
    friend void serializeMessage(Writer& writer, const Message* message);
};

template <typename Writer>
void serializeMessage(Writer& writer, const Message* message) {
    writer.StartObject();
    {
        message->messageToken_->serializeToJson(writer);
        message->contextToken_->serializeToJson(writer);
    }
    writer.EndObject();
}

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
