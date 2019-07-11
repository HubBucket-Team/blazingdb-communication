#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H

#include "blazingdb/communication/messages/BaseComponentMessage.h"

namespace blazingdb {
namespace communication {
namespace messages {

namespace tools {

template <class GpuFunctions>
class GdfColumnInHost {
public:
  explicit GdfColumnInHost(const std::string& body,
                           const std::size_t lastPosition,
                           rapidjson::Value::ConstObject &ralcolumn_info) {}

  const void* data() const noexcept { return nullptr; }

  const void* valid() const noexcept { return nullptr; }

private:
  void* data_;
  void* valid_;
  void* category_1;
  void* category_2;
};

}  // namespace tools

    template <typename RalColumn, typename CudfColumn, typename GpuFunctions>
    class GpuComponentMessage : public BaseComponentMessage {
    protected:
        GpuComponentMessage(std::unique_ptr<MessageToken>&& message_token,
                            std::shared_ptr<ContextToken>&& context_token,
                            const Node& sender_node)
        : BaseComponentMessage(std::move(message_token), std::move(context_token), sender_node)
        { }

    protected:
        GpuComponentMessage(const ContextToken& context_token, const MessageToken::TokenType& message_token)
        : BaseComponentMessage(context_token, message_token)
        { }

        GpuComponentMessage(std::shared_ptr<ContextToken>&& context_token, const MessageToken::TokenType& message_token)
        : BaseComponentMessage(std::move(context_token), message_token)
        { }

    protected:
        static void serializeCudfColumn(BaseComponentMessage::Writer& writer, CudfColumn* column) {
            writer.StartObject();
            {
                writer.Key("size");
                writer.Uint64(column->size);

                writer.Key("dtype");
                writer.Uint(column->dtype);

                writer.Key("null_count");
                writer.Uint64(column->null_count);

                writer.Key("dtype_info-time_unit");
                writer.Uint(column->dtype_info.time_unit);
            }
            writer.EndObject();
        }

        static void serializeRalColumn(BaseComponentMessage::Writer& writer, RalColumn& column) {
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
                serializeCudfColumn(writer, column.get_gdf_column());

                writer.Key("category_info-size-1");
                writer.Uint64(100);  // TODO:

                writer.Key("category_info-size-2");
                writer.Uint64(200);  // TODO:
            }
            writer.EndObject();
        }

        static std::string serializeToBinary(std::vector<RalColumn>& columns) {
            std::string result;

            std::size_t capacity = 0;
            for (const auto& column : columns) {
                capacity += GpuFunctions::getDataCapacity(column.get_gdf_column());
                capacity += GpuFunctions::getValidCapacity(column.get_gdf_column());
            }
            result.resize(capacity);

            std::size_t binary_pointer = 0;
            for (const auto& column : columns) {
                GpuFunctions::copyGpuToCpu(binary_pointer, result, const_cast<RalColumn&>(column));
            }

            return result;
        }

        static CudfColumn deserializeCudfColumn(rapidjson::Value::ConstObject&& object) {
            CudfColumn column;

            column.size = object["size"].GetUint64();

            column.dtype = (typename GpuFunctions::DType)object["dtype"].GetUint();

            column.null_count = object["null_count"].GetUint64();

            column.dtype_info = (typename GpuFunctions::DTypeInfo){
                (typename GpuFunctions::TimeUnit)object["dtype_info-time_unit"]
                    .GetUint()};

            return column;
        }

        static RalColumn deserializeRalColumn(std::size_t& binary_pointer, const std::string& binary_data, rapidjson::Value::ConstObject&& object) {
            const auto& column_name_data = object["column_name"];
            std::string column_name(column_name_data.GetString(), column_name_data.GetStringLength());

            bool is_ipc = object["is_ipc"].GetBool();

            std::uint64_t column_token = object["column_token"].GetUint64();

            auto cudf_column = deserializeCudfColumn(object["cudf_column"].GetObject());

            // Calculate pointers and update binary_pointer
            std::size_t dtype_size = GpuFunctions::getDTypeSize(cudf_column.dtype);
            std::size_t data_pointer = binary_pointer;
            std::size_t valid_pointer = data_pointer + GpuFunctions::getDataCapacity(&cudf_column);
            binary_pointer = valid_pointer + GpuFunctions::getValidCapacity(&cudf_column);

            tools::GdfColumnInHost<GpuFunctions> gdfColumn{
                binary_data, binary_pointer, object};

            RalColumn ral_column;
            if (!is_ipc) {
            	if(cudf_column.null_count == 0){
                    ral_column.create_gdf_column(cudf_column.dtype,
                                                 cudf_column.size,
                                                 (typename GpuFunctions::DataTypePointer)&binary_data[data_pointer],
                                                 (typename GpuFunctions::ValidTypePointer)nullptr,
                                                 dtype_size,
                                                 column_name);
            	}else{
                    ral_column.create_gdf_column(cudf_column.dtype,
                                                 cudf_column.size,
                                                 (typename GpuFunctions::DataTypePointer)&binary_data[data_pointer],
                                                 (typename GpuFunctions::ValidTypePointer)&binary_data[valid_pointer],
                                                 dtype_size,
                                                 column_name);
            	}

            }
            else {
                ral_column.create_gdf_column_for_ipc(cudf_column.dtype,
                                                     (typename GpuFunctions::DataTypePointer)&binary_data[data_pointer],
                                                     (typename GpuFunctions::ValidTypePointer)&binary_data[valid_pointer],
                                                     cudf_column.size,
                                                     cudf_column.null_count,
                                                     column_name);
            }

            ral_column.set_column_token(column_token);
            ral_column.get_gdf_column()->null_count = cudf_column.null_count;
            ral_column.get_gdf_column()->dtype_info = cudf_column.dtype_info;

            return ral_column;
        }
    };

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H
