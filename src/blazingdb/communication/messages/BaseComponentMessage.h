#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_BASECOMPONENTMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_BASECOMPONENTMESSAGE_H

#include <string>
#include <rapidjson/writer.h>
#include "blazingdb/communication/messages/Message.h"
#include "blazingdb/communication/Node.h"

namespace blazingdb {
namespace communication {
namespace messages {

    class BaseComponentMessage : public Message {
    protected:
        using Node = blazingdb::communication::Node;

        using StringBuffer = rapidjson::StringBuffer;
        using Writer = typename rapidjson::Writer<StringBuffer>;

    protected:
        BaseComponentMessage(std::unique_ptr<MessageToken>&& message_token,
                             std::unique_ptr<ContextToken>&& context_token,
                             const Node& sender_node)
        : Message(std::move(message_token), std::move(context_token), sender_node)
        { }

    protected:
        BaseComponentMessage(std::unique_ptr<ContextToken>&& context_token, const MessageToken::TokenType& message_token)
        : Message(MessageToken::Make(message_token), std::move(context_token))
        { }

        BaseComponentMessage(const ContextToken& context_token, const MessageToken::TokenType& message_token)
        : Message(MessageToken::Make(message_token), ContextToken::Make(context_token.getIntToken()))
        { }

    protected:
        static Node makeNode(rapidjson::Value::Object&& object) {
            return Node::make(object);
        }
    };

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_BASECOMPONENTMESSAGE_H
