#ifndef BLAZINGDB_COMMUNICATION_NODEDATAMESSAGE_H_
#define BLAZINGDB_COMMUNICATION_NODEDATAMESSAGE_H_

#include <blazingdb/communication/Message.h>
#include <blazingdb/communication/Node.h>

namespace blazingdb {
namespace communication {
namespace messages {

class NodedataMessage : public Message {
public:
  NodedataMessage(const Node& node);

  const std::string serializeToJson() const override;
  const std::string serializeToBinary() const override;
  
  const Node node;
};

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
