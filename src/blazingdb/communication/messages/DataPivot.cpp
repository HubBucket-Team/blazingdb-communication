#include "blazingdb/communication/messages/DataPivot.h"

namespace blazingdb {
namespace communication {
namespace messages {

    DataPivot::DataPivot(const Node& node, const std::string& min_range, const std::string& max_range)
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

    DataPivot DataPivot::make(rapidjson::Value::Object&& object) {
        // TODO: change hardcode data
        Node node(Address::Make("1.2.3.4", 1234));

        const auto& value_min_range = object["min_range"];
        std::string min_range(value_min_range.GetString(), value_min_range.GetStringLength());

        const auto& value_max_range = object["max_range"];
        std::string max_range(value_max_range.GetString(), value_max_range.GetStringLength());

        return DataPivot(node, min_range, max_range);
    }

} // namespace messages
} // namespace communication
} // namespace blazingdb
