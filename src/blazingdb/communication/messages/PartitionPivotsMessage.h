#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_PARTITIONPIVOTSMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_PARTITIONPIVOTSMESSAGE_H

#include <vector>
#include "blazingdb/communication/messages/DataPivot.h"

namespace blazingdb {
namespace communication {
namespace messages {

    class PartitionPivotsMessage {
    public:
        PartitionPivotsMessage(const std::vector<DataPivot>& data);

    public:
        const std::vector<DataPivot>& getDataPivots();

    public:
        template <typename Writer>
        void serialize(Writer& writer) {
            writer.StartObject();
            {
                writer.StartArray();
                {
                    for (const auto &data_pivot : data_pivot_array) {
                        data_pivot.serialize(writer);
                    }
                }
                writer.EndArray();
            }
            writer.EndObject();
        }

    private:
        std::vector<DataPivot> data_pivot_array;
    };

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_PARTITIONPIVOTSMESSAGE_H
