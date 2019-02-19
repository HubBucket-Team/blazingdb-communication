#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_SERIALIZER_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_SERIALIZER_H_

#include <memory>

#include <blazingdb/communication/Message.h>
#include <blazingdb/communication/Buffer.h>

namespace blazingdb {
namespace communication {

class MessageSerializer {
public:
  virtual Buffer serialize(const Message &message) = 0;
  virtual std::unique_ptr<Message> deserialize(const Buffer &buffer) = 0;
};

}  // namespace communication
}  // namespace blazingdb

#endif
