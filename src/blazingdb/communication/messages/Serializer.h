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
            if (column->col_name == nullptr) {
                writer.String("");
            }
            else {
                writer.String(column->col_name, std::strlen(column->col_name));
            }
        }
        writer.EndObject();
    }

    template <typename Writer, typename Column>
    void serializeGdfColumnCpp(Writer& writer, Column& column) {
        writer.StartObject();
        {
            writer.Key("is_ipc");
            writer.Bool(column.is_ipc());

            writer.Key("column_token");
            writer.Uint64(column.get_column_token());

            writer.Key("column_name");
            auto column_name = column.name();
            writer.String(column_name.c_str(), column_name.length());

            writer.Key("cudf_column");
            serializeGdfColumn(writer, column.get_gdf_column());
        }
        writer.EndObject();
    }

    template <typename CudfColumn, typename GpuFunctions>
    CudfColumn deserializeCudfColumn(rapidjson::Value::ConstObject&& object) {
        CudfColumn column;

        column.size = object["size"].GetUint64();

        column.dtype = (typename GpuFunctions::DType)object["dtype"].GetUint();

        column.null_count = object["null_count"].GetUint64();

        column.dtype_info = (typename GpuFunctions::DTypeInfo) { (typename GpuFunctions::TimeUnit)object["dtype_info"].GetUint() };

        const auto& col_name = object["col_name"];
        if (col_name.GetStringLength() == 0) {
            column.col_name = nullptr;
        }
        else {
            column.col_name = new char[col_name.GetStringLength() + 1];
            std::strncpy(column.col_name, col_name.GetString(), col_name.GetStringLength());
            column.col_name[col_name.GetStringLength()] = '\0';
        }

        return column;
    }

    template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
    RalColumn deserializeRalColumn(std::size_t& binary_pointer, const std::string& binary_data, rapidjson::Value::ConstObject&& object) {
        const auto& column_name_data = object["column_name"];
        std::string column_name(column_name_data.GetString(), column_name_data.GetStringLength());

        bool is_ipc = object["is_ipc"].GetBool();

        std::uint64_t column_token = object["column_token"].GetUint64();

        auto cudf_column = deserializeCudfColumn<CudfColumn, GpuFunctions>(object["cudf_column"].GetObject());

        // Calculate pointers and update binary_pointer
        std::size_t dtype_size = GpuFunctions::getDTypeSize(cudf_column.dtype);
        std::size_t data_pointer = binary_pointer;
        std::size_t valid_pointer = data_pointer + GpuFunctions::getDataCapacity(&cudf_column);
        binary_pointer = valid_pointer + GpuFunctions::getValidCapacity(&cudf_column);

        RalColumn ral_column;
        if (!is_ipc) {
            ral_column.create_gdf_column(cudf_column.dtype,
                                         cudf_column.size,
                                         (typename GpuFunctions::DataTypePointer)&binary_data[data_pointer],
                                         (typename GpuFunctions::ValidTypePointer)&binary_data[valid_pointer],
                                         dtype_size,
                                         column_name);
        }
        else {
            ral_column.create_gdf_column_for_ipc(cudf_column.dtype,
                                                 (typename GpuFunctions::DataTypePointer)&binary_data[data_pointer],
                                                 (typename GpuFunctions::ValidTypePointer)&binary_data[valid_pointer],
                                                 cudf_column.size,
                                                 column_name);
        }

        ral_column.set_column_token(column_token);
        ral_column.get_gdf_column()->null_count = cudf_column.null_count;
        ral_column.get_gdf_column()->dtype_info = cudf_column.dtype_info;
        if (cudf_column.col_name != nullptr) {
            std::strcpy(ral_column.get_gdf_column()->col_name, cudf_column.col_name);
        }

        return ral_column;
    }

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_SERIALIZER_H
