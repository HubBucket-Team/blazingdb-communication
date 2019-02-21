#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_SERIALIZER_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_SERIALIZER_H_

#include <memory>

#include <blazingdb/communication/Buffer.h>
#include <blazingdb/communication/Message.h>

namespace blazingdb {
namespace communication {

class MessageSerializer {
public:
  virtual Buffer serialize(const Message &message) const = 0;
  virtual std::shared_ptr<Message> deserialize(const Buffer &buffer) const = 0;
};

}  // namespace communication
}  // namespace blazingdb

#endif
