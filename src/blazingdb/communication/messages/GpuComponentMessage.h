#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H

#include <cmath>

#include "blazingdb/communication/messages/BaseComponentMessage.h"

#include <blazingdb/uc/Context.hpp>
// TODO(uc): UCPool should not be part of blazingdb-uc API.
//           Move it to blazingdb-communication messages component.
#include <blazingdb/uc/UCPool.hpp>

#include <cuda_runtime_api.h>

namespace blazingdb {
namespace communication {
namespace messages {

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

                writer.Key("dtype_info");
                writer.Uint(column->dtype_info.time_unit);
            }
            writer.EndObject();
        }

        static void serializeRalColumn(BaseComponentMessage::Writer& writer, RalColumn& column) {
            writer.StartObject();
            {
                writer.Key("is_ipc");
                writer.Bool(column.is_ipc());

                const size_t buffer_descriptor_size = 104;
                writer.Key("buffer_descriptor_size");
                writer.Uint64(buffer_descriptor_size);

                writer.Key("column_token");
                writer.Uint64(column.get_column_token());

                writer.Key("column_name");
                auto column_name = column.name();
                writer.String(column_name.c_str(), column_name.length());

                writer.Key("cudf_column");
                serializeCudfColumn(writer, column.get_gdf_column());
            }
            writer.EndObject();
        }

        //todo: deprecate this
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
        static std::string serializeToBinary(std::vector<RalColumn>& columns, const blazingdb::uc::Agent* agent) {
            std::basic_string<uint8_t>result;
            for (const auto& column : columns) {
                auto* column_ptr =  column.get_gdf_column();

                auto data_buffer = agent->Register(column_ptr->data, GpuFunctions::getDataCapacity(column_ptr));
                auto valid_buffer = agent->Register(column_ptr->valid, GpuFunctions::getValidCapacity(column_ptr));

                auto serialized_data = data_buffer->SerializedRecord();
                auto serialized_valid = valid_buffer->SerializedRecord();

                result += std::basic_string<uint8_t> (serialized_data->Data(), serialized_data->Size());
                result += std::basic_string<uint8_t> (serialized_valid->Data(), serialized_valid->Size());

                UCPool::getInstance().push(data_buffer.release());
                UCPool::getInstance().push(valid_buffer.release());
            }
            return std::string((const char *)result.c_str());
        }

        static CudfColumn deserializeCudfColumn(rapidjson::Value::ConstObject&& object) {
            CudfColumn column;

            column.size = object["size"].GetUint64();

            column.dtype = (typename GpuFunctions::DType)object["dtype"].GetUint();

            column.null_count = object["null_count"].GetUint64();

            column.dtype_info = (typename GpuFunctions::DTypeInfo) { (typename GpuFunctions::TimeUnit)object["dtype_info"].GetUint() };

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

            return ral_column;
        }

        static RalColumn
        deserializeRalColumn(std::size_t&                    binary_pointer,
                             const std::string&              binary_data,
                             rapidjson::Value::ConstObject&& object,
                             blazingdb::uc::Agent &agent) {
          const auto& column_name_data = object["column_name"];
          std::string column_name(column_name_data.GetString(),
                                  column_name_data.GetStringLength());

          bool is_ipc = object["is_ipc"].GetBool();

          std::uint64_t column_token = object["column_token"].GetUint64();

          auto cudf_column =
              deserializeCudfColumn(object["cudf_column"].GetObject());

          // Calculate pointers and update binary_pointer
          std::size_t dtype_size =
              GpuFunctions::getDTypeSize(cudf_column.dtype);
          std::size_t data_pointer = binary_pointer;
          std::size_t valid_pointer =
              data_pointer + GpuFunctions::getDataCapacity(&cudf_column);
          binary_pointer =
              valid_pointer + GpuFunctions::getValidCapacity(&cudf_column);

          // reserve for local data and valid for gdf column
          cudaError_t cudaStatus;

          void* data     = nullptr;
          int   dataSize = 100;  // gdf_size_type

          cudaStatus = cudaMalloc(&data, dataSize);
          assert(cudaSuccess == cudaStatus);

          void*       valid     = nullptr;
          std::size_t validSize = std::ceil(dataSize);

          cudaStatus = cudaMalloc(&valid, validSize);
          assert(cudaSuccess == cudaStatus);

          // get remote data and valid
          auto dataBuffer = agent.Register(data, dataSize);

          auto dataRecordData =
              reinterpret_cast<const std::uint8_t*>(binary_data.data());

          auto dataTransport = dataBuffer->Link(dataRecordData);
          auto dataFuture    = dataTransport->Get();
          dataFuture.wait();

          auto validBuffer = agent.Register(valid, validSize);

          auto validRecordData =
              reinterpret_cast<const std::uint8_t*>(binary_data.data() + 104);

          auto validTransport = validBuffer->Link(validRecordData);
          auto validFuture    = validTransport->Get();
          validFuture.wait();

          // set gdf column
          RalColumn ral_column;
          ral_column.set_column_token(column_token);

          auto gdfColumn        = ral_column.get_gdf_column();
          gdfColumn->data       = data;
          gdfColumn->valid      = valid;
          gdfColumn->null_count = cudf_column.null_count;
          gdfColumn->dtype_info = cudf_column.dtype_info;

          return ral_column;
        }
    };

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H
