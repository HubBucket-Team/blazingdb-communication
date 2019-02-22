#include "NodeDataMessage.h"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace blazingdb {
namespace communication {
namespace messages {

NodedataMessage::NodedataMessage(const Node& node)
    : Message{MessageToken::Make("NodedataMessage")}, node{node} {}

const std::string NodedataMessage::serializeToJson() const {
  rapidjson::StringBuffer stringBuffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(stringBuffer);

  node.serializeToJson(writer);

  return std::string(stringBuffer.GetString());
};

const std::string NodedataMessage::serializeToBinary() const { return ""; };

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
