#include "blazingdb/communication/messages/DataPivot.h"

namespace blazingdb {
namespace communication {
namespace messages {

    DataPivot::DataPivot(const Node& node, const std::string& mix_range, const std::string& max_range)
    : node {node}, min_range{min_range}, max_range{max_range}
    { }

    const Node& DataPivot::getNode() const {
        return node;
    }

    const std::string& DataPivot::getMinRange() const {
        return min_range;
    }

    const std::string& DataPivot::getMaxRange() const {
        return max_range;
    }

} // namespace messages
} // namespace communication
} // namespace blazingdb
