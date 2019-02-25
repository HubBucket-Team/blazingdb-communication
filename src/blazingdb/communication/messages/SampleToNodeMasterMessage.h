#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_SAMPLETONODEMASTERMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_SAMPLETONODEMASTERMESSAGE_H

#include <vector>
#include <rapidjson/writer.h>
#include "blazingdb/communication/Message.h"
#include "blazingdb/communication/Node.h"
#include "blazingdb/communication/messages/Serializer.h"

namespace blazingdb {
namespace communication {
namespace messages {

    using blazingdb::communication::Message;

    template <typename GpuData, typename GpuFunctions>
    class SampleToNodeMasterMessage : public Message {
    public:
        SampleToNodeMasterMessage(const Node& node, const std::vector<GpuData>& samples)
        : Message(MessageToken::Make(MessageID)),
          node{node},
          samples{samples}
        { }

    public:
        const Node& getNode() const {
            return node;
        }

        const std::vector<GpuData>& getSamples() const {
            return samples;
        }

    public:
        const std::string serializeToJson() const override {
            rapidjson::StringBuffer string_buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(string_buffer);

            writer.StartObject();
            {
                writer.Key("node");
                node.serializeToJson(writer);

                writer.Key("samples");
                writer.StartArray();
                {
                    for (const auto &sample : samples) {
                        serializeGdfColumnCpp(writer, const_cast<GpuData&>(sample));
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
            for (const auto& sample : samples) {
                capacity += const_cast<GpuData&>(sample).size();
            }
            result.reserve(capacity);

            for (const auto& sample : samples) {
                GpuFunctions::CopyGpuToCpu(result, const_cast<GpuData&>(sample));
            }

            return result;
        }

    private:
        const Node node;
        const std::vector<GpuData> samples;

    private:
        static const std::string MessageID;
    };

    template <typename GpuData, typename GpuFunctions>
    const std::string SampleToNodeMasterMessage<GpuData, GpuFunctions>::MessageID {"SampleToNodeMasterMessage"};

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_SAMPLETONODEMASTERMESSAGE_H
