#ifndef BLAZINGDB_COMMUNICATION_NODEDATAMESSAGE_H_
#define BLAZINGDB_COMMUNICATION_NODEDATAMESSAGE_H_

#include <blazingdb/communication/messages/Message.h>
#include <blazingdb/communication/Node.h>

namespace blazingdb {
namespace communication {
namespace messages {

class NodeDataMessage : public Message {
public:
  NodeDataMessage(const Node& node);

  const std::string serializeToJson() const override;
  const std::string serializeToBinary() const override;
  const std::string serializeToBinary(const blazingdb::uc::Agent* agent) const override;

  const Node node;

  static std::shared_ptr<NodeDataMessage> Make(const std::string& jsonBuffer, const std::string& binBuffer);

  private:
    static const std::string MessageID;
};

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_NODEDATAMESSAGE_H_
