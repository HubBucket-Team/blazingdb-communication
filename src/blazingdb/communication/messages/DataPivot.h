#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_DATAPIVOT_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_DATAPIVOT_H

#include <string>
#include "blazingdb/communication/Node.h"

namespace blazingdb {
namespace communication {
namespace messages {

    class DataPivot {
    public:
        explicit DataPivot(const Node& node, const std::string& max_range, const std::string& min_range);

    public:
        const Node& getNode() const;

        const std::string& getMinRange() const;

        const std::string& getMaxRange() const;

    public:
        template <typename Writer>
        void serialize(Writer& writer) {
            writer.StartObject();
            {
                // writer.Key("node");
                // node.serialize(writer);

                writer.Key("min_range");
                writer.String(min_range);

                writer.Key("max_range");
                writer.String(max_range);
            }
            writer.EndObject();
        }

    private:
        const Node node;
        const std::string min_range;
        const std::string max_range;
    };

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_DATAPIVOT_H
