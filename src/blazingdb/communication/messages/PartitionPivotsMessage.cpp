#include "blazingdb/communication/messages/PartitionPivotsMessage.h"

namespace blazingdb {
namespace communication {
namespace messages {

    const std::string PartitionPivotsMessage::MessageID {"PartitionPivotsMessage"};

    PartitionPivotsMessage::PartitionPivotsMessage(const ContextToken& context_token,
                                                   std::vector<DataPivot>&& data)
    : BaseComponentMessage(context_token, MessageID),
      data_pivot_array{std::move(data)}
    { }

    PartitionPivotsMessage::PartitionPivotsMessage(const ContextToken& context_token,
                                                   const std::vector<DataPivot>& data)
    : BaseComponentMessage(context_token, MessageID),
      data_pivot_array{data}
    { }

    PartitionPivotsMessage::PartitionPivotsMessage(std::shared_ptr<ContextToken>&& context_token,
                                                   std::vector<DataPivot>&& data)
    : BaseComponentMessage(std::move(context_token), MessageID),
      data_pivot_array{std::move(data)}
    { }

    PartitionPivotsMessage::PartitionPivotsMessage(std::shared_ptr<ContextToken>&& context_token,
                                                   const std::vector<DataPivot>& data)
    : BaseComponentMessage(std::move(context_token), MessageID),
      data_pivot_array{data}
    { }

    const std::vector<DataPivot>& PartitionPivotsMessage::getDataPivots() {
        return data_pivot_array;
    }

    const std::string PartitionPivotsMessage::serializeToJson() const {
        BaseComponentMessage::StringBuffer stringBuffer;
        BaseComponentMessage::Writer writer(stringBuffer);

        writer.StartObject();
        {
            serializeMessage(writer, this);

            writer.Key("pivots");
            writer.StartArray();
            {
                for (const auto& data_pivot : data_pivot_array) {
                    const_cast<DataPivot&>(data_pivot).serialize(writer);
                }
            }
            writer.EndArray();
        }
        writer.EndObject();

        return std::string(stringBuffer.GetString(), stringBuffer.GetSize());
    }

    const std::string PartitionPivotsMessage::serializeToBinary() const {
        return std::string();
    }
 
    const std::string PartitionPivotsMessage::getMessageID() {
        return MessageID;
    }

    std::shared_ptr<Message> PartitionPivotsMessage::Make(const std::string& json, const std::string& binary) {
        // Parse json
        rapidjson::Document document;
        document.Parse(json.c_str());

        // Get main object
        const auto& object = document.GetObject();

        // Get context token value;
        ContextToken::TokenType context_token = object["message"]["contextToken"].GetInt();

        // Get array pivots (payload)
        std::vector<DataPivot> data_pivot_array;
        const auto& pivots = object["pivots"].GetArray();
        for (auto& pivot : pivots) {
            data_pivot_array.emplace_back(DataPivot::make(pivot.GetObject()));
        }

        // Create message
        return std::make_shared<MessageType>(ContextToken::Make(context_token), data_pivot_array);
    }

} // namespace messages
} // namespace communication
} // namespace blazingdb
