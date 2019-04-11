#ifndef BLAZINGDB_COMMUNICATION_JSONSERIALIZABLE_H_
#define BLAZINGDB_COMMUNICATION_JSONSERIALIZABLE_H_

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace blazingdb {
namespace communication {

class JsonSerializable {
public:
  using Writer = typename rapidjson::Writer<rapidjson::StringBuffer>;
  virtual void serializeToJson(Writer& writer) const = 0;
};

}  // namespace communication
}  // namespace blazingdb

#endif
