#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_PARTITIONPIVOTSMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_PARTITIONPIVOTSMESSAGE_H

#include <vector>
#include "blazingdb/communication/messages/BaseComponentMessage.h"
#include "blazingdb/communication/messages/DataPivot.h"

namespace blazingdb {
namespace communication {
namespace messages {

    class PartitionPivotsMessage : public BaseComponentMessage {
    public:
        using MessageType = PartitionPivotsMessage;

    public:
        PartitionPivotsMessage(const std::vector<DataPivot>& data);

    public:
        const std::vector<DataPivot>& getDataPivots();

    public:
        const std::string serializeToJson() const override;

        const std::string serializeToBinary() const override;

    public:
        static std::shared_ptr<PartitionPivotsMessage> make(const std::string& json, const std::string& binary);

    private:
        std::vector<DataPivot> data_pivot_array;

    private:
        static const std::string MessageID;
    };

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_PARTITIONPIVOTSMESSAGE_H
