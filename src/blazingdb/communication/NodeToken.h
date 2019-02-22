#ifndef BLAZINGDB_COMMUNICATION_NODETOKEN_H_
#define BLAZINGDB_COMMUNICATION_NODETOKEN_H_

#include <memory>
#include <string>
#include <blazingdb/communication/shared/JsonSerializable.h>

namespace blazingdb {
namespace communication {

class NodeToken : public JsonSerializable {
public:
  virtual ~NodeToken() = default;

  // TODO: See MessageToken
  virtual bool SameValueAs(const NodeToken& rhs) const = 0;
  virtual void serializeToJson(JsonSerializable::Writer& writer) const = 0;

  static std::unique_ptr<NodeToken> Make(std::string ip, int port);
};

}  // namespace communication
}  // namespace blazingdb

#endif
