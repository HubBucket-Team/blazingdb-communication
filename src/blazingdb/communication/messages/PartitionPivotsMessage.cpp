#include "blazingdb/communication/messages/PartitionPivotsMessage.h"

namespace blazingdb {
namespace communication {
namespace messages {

    const std::string PartitionPivotsMessage::MessageID {"PartitionPivotsMessage"};

    PartitionPivotsMessage::PartitionPivotsMessage(std::vector<DataPivot>&& data)
    : BaseComponentMessage(MessageID),
      data_pivot_array{std::move(data)}
    { }

    PartitionPivotsMessage::PartitionPivotsMessage(const std::vector<DataPivot>& data)
    : BaseComponentMessage(MessageID),
      data_pivot_array{data}
    { }

    const std::vector<DataPivot>& PartitionPivotsMessage::getDataPivots() {
        return data_pivot_array;
    }

    const std::string PartitionPivotsMessage::serializeToJson() const {
        BaseComponentMessage::StringBuffer stringBuffer;
        BaseComponentMessage::Writer writer(stringBuffer);

        writer.StartArray();
        {
            for (const auto &data_pivot : data_pivot_array) {
                const_cast<DataPivot&>(data_pivot).serialize(writer);
            }
        }
        writer.EndArray();

        return std::string(stringBuffer.GetString(), stringBuffer.GetSize());
    }

    const std::string PartitionPivotsMessage::serializeToBinary() const {
        return std::string();
    }

    const std::string PartitionPivotsMessage::getMessageID() {
        return MessageID;
    }

    std::shared_ptr<PartitionPivotsMessage> PartitionPivotsMessage::Make(const std::string& json, const std::string& binary) {
        rapidjson::Document document;
        document.Parse(json.c_str());

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
