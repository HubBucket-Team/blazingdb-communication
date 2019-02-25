#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_PARTITIONPIVOTSMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_PARTITIONPIVOTSMESSAGE_H

#include <vector>
#include "src/blazingdb/communication/Message.h"
#include "blazingdb/communication/messages/DataPivot.h"

namespace blazingdb {
namespace communication {
namespace messages {

    using blazingdb::communication::Message;

    class PartitionPivotsMessage : public Message {
    public:
        PartitionPivotsMessage(const std::vector<DataPivot>& data);

    public:
        const std::vector<DataPivot>& getDataPivots();

    public:
        const std::string serializeToJson() const override;

        const std::string serializeToBinary() const override;

    public:
        static std::shared_ptr<PartitionPivotsMessage> make(const std::string& data);

    private:
        std::vector<DataPivot> data_pivot_array;

    private:
        static const std::string MessageID;
    };

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_PARTITIONPIVOTSMESSAGE_H
