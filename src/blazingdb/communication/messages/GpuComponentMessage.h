#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H

#include <cmath>

#include "blazingdb/communication/messages/BaseComponentMessage.h"
#include <blazingdb/communication/Configuration.h>

#include <blazingdb/uc/Context.hpp>
#include <iostream>

#include "blazingdb/communication/messages/UCPool.h"
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



        static std::string  RegisterAndGetBufferDescriptor(const void *agent_ptr, const void* data, size_t data_size) {
            const blazingdb::uc::Agent* agent = static_cast<const blazingdb::uc::Agent*>(agent_ptr);
            auto data_buffer = agent->Register(data, data_size);

            auto serialized_data = data_buffer->SerializedRecord();

            std::basic_string<uint8_t> binary_buffer(serialized_data->Data(), serialized_data->Size());


            std::cout << "***GetBufferDescriptor-ipc-handler***\n";
            for (auto c : binary_buffer)
                std::cout << (int) c << ", ";
            std::cout << std::endl;

            UCPool::getInstance().push(data_buffer.release());

            std::string response;
            for(auto c: binary_buffer)
                response += char(c);
            return response;
        }

        static void LinkDataRecordAndWaitForGpuData(const void *agent_ptr, const std::uint8_t* dataRecordData, const void* data, size_t dataSize) {
            const blazingdb::uc::Agent* agent = static_cast<const blazingdb::uc::Agent*>(agent_ptr);
            auto dataBuffer = agent->Register(data, dataSize);

            auto dataTransport = dataBuffer->Link(dataRecordData);
            auto dataFuture    = dataTransport->Get();
            dataFuture.wait();
        }


        static std::string serializeToBinary(std::vector<RalColumn>& columns) {
            std::string result;

            std::unique_ptr<blazingdb::uc::Context> context;

            const blazingdb::communication::Configuration &configuration =
              blazingdb::communication::Configuration::Instance();

            if (configuration.WithGDR()) {
              context = blazingdb::uc::Context::GDR();
            } else {
              context = blazingdb::uc::Context::IPCView();
            }

            auto agent  = context->Agent();

            for (const auto& column : columns) {
                auto* column_ptr =  column.get_gdf_column();
                result += GpuComponentMessage::RegisterAndGetBufferDescriptor(agent.get(), column_ptr->data, GpuFunctions::getDataCapacity(column_ptr));
                result += GpuComponentMessage::RegisterAndGetBufferDescriptor(agent.get(), column_ptr->valid, GpuFunctions::getValidCapacity(column_ptr));
            }
            std::hash<std::string> hasher;
            auto hashed = hasher(result); 

            std::cout << "****message sent: " << hashed << std::endl; 

            UCPool::getInstance().push(agent.release());
            UCPool::getInstance().push(context.release());
            return result;
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
        // TODO: elimitar todo rastro de simple-distribution.
        static RalColumn
        deserializeRalColumn(std::size_t&                    binary_pointer,
                             const std::string&              binary_data,
                             rapidjson::Value::ConstObject&& object,
                             const void*                     agent) {
          const auto& column_name_data = object["column_name"];
          std::string column_name(column_name_data.GetString(),
                                  column_name_data.GetStringLength());
          bool is_ipc = object["is_ipc"].GetBool();

          std::uint64_t column_token = object["column_token"].GetUint64();

          auto cudf_column =
              deserializeCudfColumn(object["cudf_column"].GetObject());

          // reserve for local data and valid for gdf column
          cudaError_t cudaStatus;

          void* data     = nullptr;
          int   dataSize = 100;   //TODO: fix dataSize gdf_size_type, RalColumn::DataSize(cudf_column.size, cudf_column.dtype)

          cudaStatus = cudaMalloc(&data, dataSize);
          assert(cudaSuccess == cudaStatus);

          void*       valid     = nullptr;
          std::size_t validSize = std::ceil(dataSize);

          cudaStatus = cudaMalloc(&valid, validSize);
          assert(cudaSuccess == cudaStatus);

          auto dataRecordData =
              reinterpret_cast<const std::uint8_t*>(binary_data.data()) +
              binary_pointer;

          // TODO(issue): get magic number 104 from json data
          auto validRecordData =
              reinterpret_cast<const std::uint8_t*>(binary_data.data()) +
              binary_pointer + 104;

          GpuComponentMessage::LinkDataRecordAndWaitForGpuData(
              agent, dataRecordData, data, dataSize);
          GpuComponentMessage::LinkDataRecordAndWaitForGpuData(
              agent, validRecordData, valid, validSize);

          // set gdf column
          RalColumn ral_column; 
          //@todo: is ipc columnn?
          ral_column.create_gdf_column_for_ipc(cudf_column.dtype,
                                                    data,
                                                    (unsigned char*)valid,
                                                    cudf_column.size,
                                                    column_name);
          ral_column.set_column_token(column_token);
          ral_column.get_gdf_column()->null_count = cudf_column.null_count;
          ral_column.get_gdf_column()->dtype_info = cudf_column.dtype_info;
        
          binary_pointer += 208;

          return ral_column;
        }
    };

    }  // namespace messages
    }  // namespace communication
    }  // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H
