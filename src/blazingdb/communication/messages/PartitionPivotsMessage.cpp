#include "blazingdb/communication/messages/PartitionPivotsMessage.h"
#include <rapidjson/writer.h>

namespace blazingdb {
namespace communication {
namespace messages {

    using blazingdb::communication::MessageToken;

    const std::string PartitionPivotsMessage::MessageID {"PartitionPivotsMessage"};

    PartitionPivotsMessage::PartitionPivotsMessage(const std::vector<DataPivot>& data)
    : Message(MessageToken::Make(MessageID)),
      data_pivot_array{data}
    { }

    const std::vector<DataPivot>& PartitionPivotsMessage::getDataPivots() {
        return data_pivot_array;
    }

    const std::string PartitionPivotsMessage::serializeToJson() const {
        rapidjson::StringBuffer string_buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(string_buffer);

        writer.StartObject();
        {
            writer.StartArray();
            {
                for (const auto &data_pivot : data_pivot_array) {
                    const_cast<DataPivot&>(data_pivot).serialize(writer);
                }
            }
            writer.EndArray();
        }
        writer.EndObject();

        return std::string(string_buffer.GetString(), string_buffer.GetSize());
    }

    const std::string PartitionPivotsMessage::serializeToBinary() const {
        return std::string();
    }

} // namespace messages
} // namespace communication
} // namespace blazingdb
