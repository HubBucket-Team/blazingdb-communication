#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_DATASCATTERMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_DATASCATTERMESSAGE_H

#include <vector>
#include <rapidjson/writer.h>
#include "blazingdb/communication/messages/Message.h"
#include "blazingdb/communication/messages/Serializer.h"

namespace blazingdb {
namespace communication {
namespace messages {

    template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
    class DataScatterMessage : public Message {
    private:
        using MessageType = DataScatterMessage<RalColumn, CudfColumn, GpuFunctions>;

    public:
        DataScatterMessage(std::vector<RalColumn>&& columns)
        : Message(MessageToken::Make(MessageID)),
          columns{std::move(columns)}
        { }

        DataScatterMessage(const std::vector<RalColumn>& columns)
        : Message(MessageToken::Make(MessageID)),
          columns{columns}
        { }

    public:
        const std::vector<RalColumn>& getColumns() const {
            return columns;
        }

    public:
        const std::string serializeToJson() const override {
            rapidjson::StringBuffer string_buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(string_buffer);

            writer.StartArray();
            {
                for (const auto &column : columns) {
                    serializeGdfColumnCpp(writer, const_cast<RalColumn&>(column));
                }
            }
            writer.EndArray();

            return std::string(string_buffer.GetString(), string_buffer.GetSize());
        }

        const std::string serializeToBinary() const override {
            std::string result;

            std::size_t capacity = 0;
            for (const auto& column : columns) {
                capacity += GpuFunctions::getDataCapacity(column.get_gdf_column());
                capacity += GpuFunctions::getValidCapacity(column.get_gdf_column());
            }
            result.resize(capacity);

            std::size_t binary_pointer = 0;
            for (const auto& column : columns) {
                GpuFunctions::copyGpuToCpu(binary_pointer, result, const_cast<RalColumn&>(column));
            }

            return result;
        }

    public:
        static std::shared_ptr<MessageType> make(const std::string& json, const std::string& binary) {
            // Parse
            rapidjson::Document document;
            document.Parse(json.c_str());

            // The gdf_column_cpp container
            std::vector<RalColumn> columns;

            // Make the deserialization
            std::size_t binary_pointer = 0;
            const auto& gpu_data_array = document.GetArray();
            for (const auto& gpu_data : gpu_data_array) {
                columns.emplace_back(deserializeRalColumn<RalColumn, CudfColumn, GpuFunctions>(binary_pointer, binary, gpu_data.GetObject()));
            }

            // Create the message
            return std::make_shared<MessageType>(std::move(columns));
        }

    private:
        const std::vector<RalColumn> columns;

    private:
        static const std::string MessageID;
    };

    template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
    const std::string DataScatterMessage<RalColumn, CudfColumn, GpuFunctions>::MessageID {"DataScatterMessage"};

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_DATASCATTERMESSAGE_H
