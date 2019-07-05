#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_DATASCATTERMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_DATASCATTERMESSAGE_H

#include <rapidjson/writer.h>
#include <vector>
#include "blazingdb/communication/messages/GpuComponentMessage.h"
#include "blazingdb/communication/messages/Message.h"

namespace blazingdb {
namespace communication {
namespace messages {

template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
class DataScatterMessage
    : public GpuComponentMessage<RalColumn, CudfColumn, GpuFunctions> {
private:
  using BaseClass = GpuComponentMessage<RalColumn, CudfColumn, GpuFunctions>;

public:
  using MessageType = DataScatterMessage<RalColumn, CudfColumn, GpuFunctions>;

public:
  DataScatterMessage(const ContextToken&      context_token,
                     std::vector<RalColumn>&& columns)
      : BaseClass(context_token, MessageID), columns{std::move(columns)} {}

  DataScatterMessage(const ContextToken&           context_token,
                     const std::vector<RalColumn>& columns)
      : BaseClass(context_token, MessageID), columns{columns} {}

  DataScatterMessage(std::shared_ptr<ContextToken>&& context_token,
                     std::vector<RalColumn>&&        columns)
      : BaseClass(std::move(context_token), MessageID),
        columns{std::move(columns)} {}

  DataScatterMessage(std::shared_ptr<ContextToken>&& context_token,
                     const std::vector<RalColumn>&   columns)
      : BaseClass(std::move(context_token), MessageID), columns{columns} {}

public:
  const std::vector<RalColumn>&
  getColumns() const {
    return columns;
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

      // Serialize columns
      writer.Key("columns");
      writer.StartArray();
      {
        for (const auto& column : columns) {
          BaseClass::serializeRalColumn(writer, const_cast<RalColumn&>(column));
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
        const_cast<std::vector<RalColumn>&>(columns));
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

    // Get context token value;
    ContextToken::TokenType context_token =
        object["message"]["contextToken"].GetInt();

    // blazingdb-uc
    const Configuration& configuration =
        blazingdb::communication::Configuration::Instance();

    std::unique_ptr<blazingdb::uc::Context> context =
        configuration.WithGDR() ? blazingdb::uc::Context::GDR()
                                : blazingdb::uc::Context::IPC();

    auto agent = context->Agent();

    // Get array columns (payload)
    std::vector<RalColumn> columns =
        BaseClass::deserializeRalColumns(binary, *agent);

    agent.release();
    context.release();

    // Create the message
    return std::make_shared<MessageType>(ContextToken::Make(context_token),
                                         std::move(columns));
  }

private:
  const std::vector<RalColumn> columns;

private:
  static const std::string MessageID;
};

template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
const std::string
    DataScatterMessage<RalColumn, CudfColumn, GpuFunctions>::MessageID{
        "DataScatterMessage"};

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif  // BLAZINGDB_COMMUNICATION_MESSAGES_DATASCATTERMESSAGE_H
