#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_SAMPLETONODEMASTERMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_SAMPLETONODEMASTERMESSAGE_H

#include <vector>
#include "blazingdb/communication/Node.h"
#include "blazingdb/communication/messages/GpuComponentMessage.h"

#include <blazingdb/communication/Configuration.h>

namespace blazingdb {
namespace communication {
namespace messages {

template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
class SampleToNodeMasterMessage
    : public GpuComponentMessage<RalColumn, CudfColumn, GpuFunctions> {
private:
  using BaseClass = GpuComponentMessage<RalColumn, CudfColumn, GpuFunctions>;

public:
  using MessageType =
      SampleToNodeMasterMessage<RalColumn, CudfColumn, GpuFunctions>;

public:
  SampleToNodeMasterMessage(std::unique_ptr<MessageToken>&& messageToken,
                            std::shared_ptr<ContextToken>&& contextToken,
                            const Node&                     sender_node,
                            std::uint64_t                   total_row_size,
                            std::vector<RalColumn>&&        samples)
      : BaseClass(
            std::move(messageToken), std::move(contextToken), sender_node),
        total_row_size_{total_row_size},
        samples_{std::move(samples)} {}

  SampleToNodeMasterMessage(std::unique_ptr<MessageToken>&& messageToken,
                            std::shared_ptr<ContextToken>&& contextToken,
                            const Node&                     sender_node,
                            std::uint64_t                   total_row_size,
                            const std::vector<RalColumn>&   samples)
      : BaseClass(
            std::move(messageToken), std::move(contextToken), sender_node),
        total_row_size_{total_row_size},
        samples_{samples} {}

public:
  std::uint64_t
  getTotalRowSize() const {
    return total_row_size_;
  }

  std::vector<RalColumn>
  getSamples() {
    return std::move(samples_);
  }

public:
  const std::string
  serializeToJson() const override {
    typename BaseClass::StringBuffer stringBuffer;
    typename BaseClass::Writer       writer(stringBuffer);

    writer.StartObject();
    {
      // Serialize Message
      serializeMessage(writer, this);

      // Serialize total_data_size
      writer.Key("total_row_size");
      writer.Uint64(total_row_size_);

      // Serialize RalColumns
      writer.Key("samples");
      writer.StartArray();
      {
        for (const auto& sample : samples_) {
          BaseClass::serializeRalColumn(writer, const_cast<RalColumn&>(sample));
        }
      }
      writer.EndArray();
    }
    writer.EndObject();

    return std::string(stringBuffer.GetString(), stringBuffer.GetSize());
  }

  const std::string
  serializeToBinary() const override {
    return BaseClass::serializeToBinary(
        const_cast<std::vector<RalColumn>&>(samples_));
  }

public:
  static const std::string
  getMessageID() {
    return MessageID;
  }

  static std::shared_ptr<Message>
  Make(const std::string& json, const std::string& binary) {
    // Parse json
    rapidjson::Document document;
    document.Parse(json.c_str());

    // Get main object
    const auto& object = document.GetObject();

    // Get message values;
    std::unique_ptr<Node>         sender_node;
    std::unique_ptr<MessageToken> messageToken;
    std::shared_ptr<ContextToken> contextToken;
    deserializeMessage(
        object["message"].GetObject(), messageToken, contextToken, sender_node);

    // Get total row size
    std::uint64_t total_row_size = document["total_row_size"].GetUint64();

    // blazingdb-uc
    const Configuration& configuration =
        blazingdb::communication::Configuration::Instance();

    std::unique_ptr<blazingdb::uc::Context> context;

    if (configuration.WithGDR()) {
      context = blazingdb::uc::Context::GDR();
    } else {
      context = blazingdb::uc::Context::IPCView();
    }

    auto agent = context->Agent();

    // Get samples
    std::hash<std::string> hasher;
    auto                   hashed = hasher(binary);
    std::cout << "****Make message from bin: " << binary << std::endl;
    std::cout << "****Make message from: " << hashed << std::endl;

    std::vector<RalColumn> columns =
        BaseClass::deserializeRalColumns(binary, *agent);

    /// TODO(workaround): remove this.
    /// Write code to manage the life cycle of a UC context.
    agent.release();
    context.release();

    // Create the message
    return std::make_shared<MessageType>(std::move(messageToken),
                                         std::move(contextToken),
                                         *sender_node,
                                         total_row_size,
                                         std::move(columns));
  }

private:
  const uint64_t         total_row_size_;
  std::vector<RalColumn> samples_;

private:
  static const std::string MessageID;
};

template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
const std::string
    SampleToNodeMasterMessage<RalColumn, CudfColumn, GpuFunctions>::MessageID{
        "SampleToNodeMasterMessage"};

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif  // BLAZINGDB_COMMUNICATION_MESSAGES_SAMPLETONODEMASTERMESSAGE_H
