#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_SAMPLETONODEMASTERMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_SAMPLETONODEMASTERMESSAGE_H

#include <vector>
#include "blazingdb/communication/Node.h"
#include "blazingdb/communication/messages/GpuComponentMessage.h"

namespace blazingdb {
namespace communication {
namespace messages {

    template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
    class SampleToNodeMasterMessage : public GpuComponentMessage<RalColumn, CudfColumn, GpuFunctions> {
    private:
        using BaseClass = GpuComponentMessage<RalColumn, CudfColumn, GpuFunctions>;

    public:
        using MessageType = SampleToNodeMasterMessage<RalColumn, CudfColumn, GpuFunctions>;

    public:
        SampleToNodeMasterMessage(const Node& node, std::vector<RalColumn>&& samples)
        : BaseClass(MessageID),
          node{node},
          samples{std::move(samples)}
        { }

        SampleToNodeMasterMessage(const Node& node, const std::vector<RalColumn>& samples)
        : BaseClass(MessageID),
          node{node},
          samples{samples}
        { }

    public:
        const Node& getNode() const {
            return node;
        }

        const std::vector<RalColumn>& getSamples() const {
            return samples;
        }

    public:
        const std::string serializeToJson() const override {
            typename BaseClass::StringBuffer stringBuffer;
            typename BaseClass::Writer writer(stringBuffer);

            writer.StartObject();
            {
                // Serialize Node
                node.serializeToJson(writer);

                // Serialize RalColumns
                writer.Key("samples");
                writer.StartArray();
                {
                    for (const auto& sample : samples) {
                        BaseClass::serializeRalColumn(writer, const_cast<RalColumn&>(sample));
                    }
                }
                writer.EndArray();
            }
            writer.EndObject();

            return std::string(stringBuffer.GetString(), stringBuffer.GetSize());
        }

        const std::string serializeToBinary() const override {
            return BaseClass::serializeToBinary(const_cast<std::vector<RalColumn>&>(samples));
        }

    public:
        static std::shared_ptr<MessageType> Make(const std::string& json, const std::string& binary) {
            // Parse
            rapidjson::Document document;
            document.Parse(json.c_str());

            // The gdf_column_cpp container
            std::vector<RalColumn> columns;

            // Deserialize Node class
            Node node = BaseClass::makeNode(document["node"].GetObject());

            // Make the deserialization
            std::size_t binary_pointer = 0;
            const auto& gpu_data_array = document["samples"].GetArray();
            for (const auto& gpu_data : gpu_data_array) {
                columns.emplace_back(BaseClass::deserializeRalColumn(binary_pointer, binary, gpu_data.GetObject()));
            }

            // Create the message
            return std::make_shared<MessageType>(node, std::move(columns));
        }

    private:
        const Node node;
        const std::vector<RalColumn> samples;

    private:
        static const std::string MessageID;
    };

    template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
    const std::string SampleToNodeMasterMessage<RalColumn, CudfColumn, GpuFunctions>::MessageID {"SampleToNodeMasterMessage"};

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_SAMPLETONODEMASTERMESSAGE_H
