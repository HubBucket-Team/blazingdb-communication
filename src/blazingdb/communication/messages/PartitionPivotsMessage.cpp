#include "blazingdb/communication/messages/PartitionPivotsMessage.h"

namespace blazingdb {
namespace communication {
namespace messages {

    PartitionPivotsMessage::PartitionPivotsMessage(const std::vector<DataPivot>& data)
    : data_pivot_array{data}
    { }

    const std::vector<DataPivot>& PartitionPivotsMessage::getDataPivots() {
        return data_pivot_array;
    }

} // namespace messages
} // namespace communication
} // namespace blazingdb
