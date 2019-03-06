#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_COLUMNDATAMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_COLUMNDATAMESSAGE_H

#include <vector>
#include <rapidjson/writer.h>
#include "blazingdb/communication/messages/Message.h"
#include "blazingdb/communication/messages/GpuComponentMessage.h"

namespace blazingdb {
namespace communication {
namespace messages {

template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
class ColumnDataMessage : public GpuComponentMessage<RalColumn, CudfColumn, GpuFunctions> {
private:
    using BaseClass = GpuComponentMessage<RalColumn, CudfColumn, GpuFunctions>;

public:
    using MessageType = ColumnDataMessage<RalColumn, CudfColumn, GpuFunctions>;

public:
    ColumnDataMessage(std::unique_ptr<MessageToken>&& message_token,
                      std::unique_ptr<ContextToken>&& context_token,
                      const Node& sender_node,
                      std::vector<RalColumn>&& columns)
    : BaseClass(std::move(message_token), std::move(context_token), sender_node),
      columns{std::move(columns)}
    { }

    ColumnDataMessage(std::unique_ptr<MessageToken>&& message_token,
                      std::unique_ptr<ContextToken>&& context_token,
                      const Node& sender_node,
                      const std::vector<RalColumn>& columns)
    : BaseClass(std::move(message_token), std::move(context_token), sender_node),
      columns{columns}
    { }

public:
    const std::vector<RalColumn>& getColumns() const {
        return columns;
    }

public:
    const std::string serializeToJson() const override {
        typename BaseClass::StringBuffer stringBuffer;
        typename BaseClass::Writer writer(stringBuffer);

        writer.StartObject();
        {
            // Serialize Message
            serializeMessage(writer, this);

            // Serialize columns
            writer.Key("columns");
            writer.StartArray();
            {
                for (const auto &column : columns) {
                    BaseClass::serializeRalColumn(writer, const_cast<RalColumn&>(column));
                }
            }
            writer.EndArray();
        }
        writer.EndObject();

        return std::string(stringBuffer.GetString(), stringBuffer.GetSize());
    }

    const std::string serializeToBinary() const override {
        return BaseClass::serializeToBinary(const_cast<std::vector<RalColumn>&>(columns));
    }

public:
    static const std::string getMessageID() {
        return MessageID;
    }

    static std::shared_ptr<Message> Make(const std::string& json, const std::string& binary) {
        // Parse json
        rapidjson::Document document;
        document.Parse(json.c_str());

        // Get main object
        const auto& object = document.GetObject();

        // Get message values;
        std::unique_ptr<Node> node;
        std::unique_ptr<MessageToken> messageToken;
        std::unique_ptr<ContextToken> contextToken;
        deserializeMessage(object["message"].GetObject(), messageToken, contextToken, node);

        // Get array columns (payload)
        std::size_t binary_pointer = 0;
        std::vector<RalColumn> columns;
        const auto& gpu_data_array = object["columns"].GetArray();
        for (const auto& gpu_data : gpu_data_array) {
            columns.emplace_back(BaseClass::deserializeRalColumn(binary_pointer, binary, gpu_data.GetObject()));
        }

        // Create the message
        return std::make_shared<MessageType>(std::move(messageToken), std::move(contextToken), *node, std::move(columns));
    }

private:
    const std::vector<RalColumn> columns;

private:
    static const std::string MessageID;
};

template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
const std::string ColumnDataMessage<RalColumn, CudfColumn, GpuFunctions>::MessageID {"ColumnDataMessage"};

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_COLUMNDATAMESSAGE_H
