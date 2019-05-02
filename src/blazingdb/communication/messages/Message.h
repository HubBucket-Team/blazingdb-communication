#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_H_

#include <memory>

#include "blazingdb/communication/ContextToken.h"
#include "blazingdb/communication/Node.h"
#include "blazingdb/communication/messages/MessageToken.h"

namespace blazingdb {
namespace communication {
namespace messages {

class Message {
public:
  explicit Message(std::unique_ptr<MessageToken> &&messageToken,
                   std::shared_ptr<ContextToken> &&contextToken);

  explicit Message(std::unique_ptr<MessageToken>&& messageToken,
                   std::shared_ptr<ContextToken>&& contextToken,
                   const Node& sender_node);

  virtual ~Message();

public:
  virtual const std::string serializeToJson() const = 0;
  virtual const std::string serializeToBinary() const = 0;

  virtual void CreateRemoteBuffer(const Node &) const {
    throw std::runtime_error("Message#CreateRemoteBuffer not implemented");
  }

  /*virtual void GetRemoteBuffer(const Node &) const {*/
    /*throw std::runtime_error("Message#GetRemoteBuffer not implemented");*/
  /*}*/

  static void
  GetRemoteBuffer(const Node& node);

public:
    ContextToken::TokenType getContextTokenValue() const;

    const MessageToken::TokenType getMessageTokenValue() const;

    const blazingdb::communication::Node& getSenderNode() const;

private:
  const std::unique_ptr<MessageToken> messageToken_;
  const std::shared_ptr<ContextToken> contextToken_;

private:
    const blazingdb::communication::Node sender_node_;

private:
    template <typename Writer>
    friend void serializeMessage(Writer& writer, const Message* message);

private:
    template <typename Object>
    friend void deserializeMessage(Object& object,
                                   std::unique_ptr<MessageToken>& messageToken,
                                   std::shared_ptr<ContextToken>& contextToken,
                                   Node& node);
};

template <typename Writer>
void serializeMessage(Writer& writer, const Message* message) {
    writer.Key("message");
    writer.StartObject();
    {
        message->messageToken_->serializeToJson(writer);
        message->contextToken_->serializeToJson(writer);
        if (message->sender_node_.isAvailable()) {
            message->sender_node_.serializeToJson(writer);
        }
    }
    writer.EndObject();
}

template <typename Object>
void deserializeMessage(const Object& object,
                        std::unique_ptr<MessageToken>& messageToken,
                        std::shared_ptr<ContextToken>& contextToken,
                        std::unique_ptr<Node>& node) {
    // Get message token;
    const auto& message_token_value = object["messageToken"];
    MessageToken::TokenType message_token{message_token_value.GetString(), message_token_value.GetStringLength()};
    messageToken = MessageToken::Make(message_token);

    // Get context token;
    ContextToken::TokenType context_token = object["contextToken"].GetInt();
    contextToken = ContextToken::Make(context_token);

    // Get sender node
    if (object.HasMember("node")) {
        node = blazingdb::communication::Node::makeUnique(object["node"].GetObject());
    }
}

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
