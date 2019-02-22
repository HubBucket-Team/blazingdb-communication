#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_SERIALIZER_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_SERIALIZER_H

#include <string>
#include <rapidjson/stringbuffer.h>

namespace blazingdb {
namespace communication {
namespace messages {

    template <typename Writer, typename Column>
    void serializeGdfColumn(Writer& writer, Column* column) {
        writer.StartObject();
        {
            writer.Key("size");
            writer.Uint64(column->size);

            writer.Key("dtype");
            writer.Uint(column->dtype);

            writer.Key("null_count");
            writer.Uint64(column->null_count);

            writer.Key("dtype_info");
            writer.Uint(column->dtype_info.time_unit);

            writer.Key("col_name");
            writer.String(column->col_name, std::strlen(column->col_name));
        }
        writer.EndObject();
    }

    template <typename Writer, typename Column>
    void serializeGdfColumnCpp(Writer& writer, Column& column) {
        writer.StartObject();
        {
            writer.Key("column");
            serializeGdfColumn(writer, column.get_gdf_column());

            writer.Key("column_name");
            auto column_name = column.name();
            writer.String(column_name.c_str(), column_name.length());

            writer.Key("is_ipc");
            writer.Bool(column.is_ipc());

            writer.Key("get_column_token");
            writer.Uint64(column.get_column_token());
        }
        writer.EndObject();
    }

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_SERIALIZER_H
