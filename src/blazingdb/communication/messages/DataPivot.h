#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_DATAPIVOT_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_DATAPIVOT_H

#include <string>
#include "blazingdb/communication/Node.h"

namespace blazingdb {
namespace communication {
namespace messages {

    class DataPivot {
    public:
        explicit DataPivot(const Node& node, std::string&& min_range, std::string&& max_range);

        explicit DataPivot(const Node& node, const std::string& min_range, const std::string& max_range);

    public:
        const Node& getNode() const;

        const std::string& getMinRange() const;

        const std::string& getMaxRange() const;

    public:
        template <typename Writer>
        void serialize(Writer& writer) {
            writer.StartObject();
            {
                node.serializeToJson(writer);

                writer.Key("min_range");
                writer.String(min_range.c_str());

                writer.Key("max_range");
                writer.String(max_range.c_str());
            }
            writer.EndObject();
        }

    public:
        static DataPivot make(rapidjson::Value::Object&& object);

    private:
        const Node node;
        const std::string min_range;
        const std::string max_range;
    };

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_DATAPIVOT_H
