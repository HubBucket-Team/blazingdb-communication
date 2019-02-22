#ifndef BLAZINGDB_COMMUNICATION_NODETOKEN_H_
#define BLAZINGDB_COMMUNICATION_NODETOKEN_H_

#include <blazingdb/communication/shared/JsonSerializable.h>
#include <rapidjson/document.h>
#include <memory>
#include <string>
namespace blazingdb {
namespace communication {

class NodeToken : public JsonSerializable {
public:
  virtual ~NodeToken() = default;

  // TODO: See MessageToken
  virtual bool SameValueAs(const NodeToken& rhs) const = 0;
  virtual void serializeToJson(JsonSerializable::Writer& writer) const = 0;

  static std::unique_ptr<NodeToken> Make(std::string ip, int port);
  static std::unique_ptr<NodeToken> Make(rapidjson::Document& doc);
};

}  // namespace communication
}  // namespace blazingdb

#endif
