#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_BASECOMPONENTMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_BASECOMPONENTMESSAGE_H

#include <string>
#include <rapidjson/writer.h>
#include "blazingdb/communication/messages/Message.h"

namespace blazingdb {
namespace communication {
namespace messages {

    class BaseComponentMessage : public Message {
    protected:
        using StringBuffer = rapidjson::StringBuffer;
        using Writer = typename rapidjson::Writer<StringBuffer>;

    protected:
        BaseComponentMessage(const std::string& messageID)
        : Message(MessageToken::Make(messageID))
        { }

    protected:
        static Node makeNode(rapidjson::Value::Object&& object) {
            using blazingdb::communication::Node;
            return Node::make(object);
        }
    };

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_BASECOMPONENTMESSAGE_H
