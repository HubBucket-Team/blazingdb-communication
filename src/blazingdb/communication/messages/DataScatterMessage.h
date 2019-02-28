#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_DATASCATTERMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_DATASCATTERMESSAGE_H

#include <vector>
#include <rapidjson/writer.h>
#include "blazingdb/communication/messages/Message.h"
#include "blazingdb/communication/messages/GpuComponentMessage.h"

namespace blazingdb {
namespace communication {
namespace messages {

    template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
    class DataScatterMessage : public GpuComponentMessage<RalColumn, CudfColumn, GpuFunctions> {
    private:
        using BaseClass = GpuComponentMessage<RalColumn, CudfColumn, GpuFunctions>;

    public:
        using MessageType = DataScatterMessage<RalColumn, CudfColumn, GpuFunctions>;

    public:
        DataScatterMessage(std::vector<RalColumn>&& columns)
        : BaseClass(MessageID),
          columns{std::move(columns)}
        { }

        DataScatterMessage(const std::vector<RalColumn>& columns)
        : BaseClass(MessageID),
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

            writer.StartArray();
            {
                for (const auto &column : columns) {
                    BaseClass::serializeRalColumn(writer, const_cast<RalColumn&>(column));
                }
            }
            writer.EndArray();

            return std::string(stringBuffer.GetString(), stringBuffer.GetSize());
        }

        const std::string serializeToBinary() const override {
            return BaseClass::serializeToBinary(const_cast<std::vector<RalColumn>&>(columns));
        }

    public:
        static const std::string getMessageID() {
            return MessageID;
        }

        static std::shared_ptr<MessageType> Make(const std::string& json, const std::string& binary) {
            // Parse
            rapidjson::Document document;
            document.Parse(json.c_str());

            // The gdf_column_cpp container
            std::vector<RalColumn> columns;

            // Make the deserialization
            std::size_t binary_pointer = 0;
            const auto& gpu_data_array = document.GetArray();
            for (const auto& gpu_data : gpu_data_array) {
                columns.emplace_back(BaseClass::deserializeRalColumn(binary_pointer, binary, gpu_data.GetObject()));
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
