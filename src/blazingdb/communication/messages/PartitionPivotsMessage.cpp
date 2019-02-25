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

        writer.StartArray();
        {
            for (const auto &data_pivot : data_pivot_array) {
                const_cast<DataPivot&>(data_pivot).serialize(writer);
            }
        }
        writer.EndArray();

        return std::string(string_buffer.GetString(), string_buffer.GetSize());
    }

    const std::string PartitionPivotsMessage::serializeToBinary() const {
        return std::string();
    }

    std::shared_ptr<PartitionPivotsMessage> PartitionPivotsMessage::make(const std::string& data) {
        rapidjson::Document document;
        document.Parse(data.c_str());

        std::vector<DataPivot> data_pivot_array;

        const auto& pivots = document.GetArray();
        for (auto& pivot : pivots) {
            data_pivot_array.emplace_back(DataPivot::make(pivot.GetObject()));
        }

        return std::make_unique<PartitionPivotsMessage>(data_pivot_array);
    }

} // namespace messages
} // namespace communication
} // namespace blazingdb
