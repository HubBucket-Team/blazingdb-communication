#include "NodeDataMessage.h"
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <cstring>
#include <iostream>

namespace blazingdb {
namespace communication {
namespace messages {

NodeDataMessage::NodeDataMessage(const Node& node)
    : Message{MessageToken::Make("NodeDataMessage")}, node{node} {}

const std::string NodeDataMessage::serializeToJson() const {
  rapidjson::StringBuffer stringBuffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(stringBuffer);

  writer.StartObject();
  node.serializeToJson(writer);
  writer.EndObject();

  return std::string(stringBuffer.GetString());
};

const std::string NodeDataMessage::serializeToBinary() const { return ""; };

std::shared_ptr<NodeDataMessage> NodeDataMessage::make(const std::string& jsonBuffer,
                                               const std::string& binBuffer) {
  rapidjson::Document doc;

  // if (!document.HasMember("node")) {
  //// TODO: raise exception
  //}

  std::unique_ptr<char[]> jsonChars(new char[jsonBuffer.size() + 1]);
  std::strcpy(jsonChars.get(), jsonBuffer.c_str());
  if (doc.ParseInsitu(jsonChars.get()).HasParseError()) {
    std::cerr << "NodeDataMessage::make => rapidjson::ParseInsitu Error(offset "
              << (unsigned)doc.GetErrorOffset()
              << "): " << rapidjson::GetParseError_En(doc.GetParseError())
              << "\n";

    return std::shared_ptr<NodeDataMessage>{};
  }

  return std::make_shared<NodeDataMessage>(Node::make(doc["node"].GetObject()));
}

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
