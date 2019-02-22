#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_DATASCATTERMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_DATASCATTERMESSAGE_H

#include <vector>
#include <rapidjson/writer.h>
#include "blazingdb/communication/Message.h"
#include "blazingdb/communication/messages/Serializer.h"

namespace blazingdb {
namespace communication {
namespace messages {

    using blazingdb::communication::Message;

    template <typename GpuData, typename GpuFunctions>
    class DataScatterMessage : public Message {
    public:
        DataScatterMessage(const std::vector<GpuData>& columns)
        : Message(MessageToken::Make(MessageID)),
          columns{columns}
        { }

    public:
        const std::vector<GpuData>& getColumns() const {
            return columns;
        }

    public:
        const std::string serializeToJson() const override {
            rapidjson::StringBuffer string_buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(string_buffer);

            writer.StartObject();
            {
                writer.Key("columns");
                writer.StartArray();
                {
                    for (const auto &column : columns) {
                        serializeGdfColumnCpp(writer, const_cast<GpuData&>(column));
                    }
                }
                writer.EndArray();
            }
            writer.EndObject();

            return std::string(string_buffer.GetString(), string_buffer.GetSize());
        }

        const std::string serializeToBinary() const override {
            std::string result;

            std::size_t capacity = 0;
            for (const auto& column : columns) {
                capacity += const_cast<GpuData&>(column).size();
            }
            result.reserve(capacity);

            for (const auto& column : columns) {
                GpuFunctions::CopyGpuToCpu(result, const_cast<GpuData&>(column));
            }

            return result;
        }

    private:
        const std::vector<GpuData> columns;

    private:
        static const std::string MessageID;
    };

    template <typename GpuData, typename GpuFunctions>
    const std::string DataScatterMessage<GpuData, GpuFunctions>::MessageID {"DataScatterMessage"};

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_DATASCATTERMESSAGE_H
