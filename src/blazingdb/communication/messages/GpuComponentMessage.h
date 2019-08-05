#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H
#define BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <vector>

#include <boost/iterator/zip_iterator.hpp>
#include <boost/range.hpp>

#include "blazingdb/communication/messages/BaseComponentMessage.h"

namespace blazingdb {
namespace communication {
namespace messages {

namespace {
template <class... Containers>
inline auto zip(Containers&&... containers)
    -> boost::iterator_range<boost::zip_iterator<
        decltype(boost::make_tuple(std::begin(containers)...))>> {
  return boost::make_iterator_range(
      boost::make_zip_iterator(boost::make_tuple(std::begin(containers)...)),
      boost::make_zip_iterator(boost::make_tuple(std::end(containers)...)));
}
}  // namespace

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

                writer.Key("has_valids");
                writer.Bool(column.null_count() > 0 || (column.null_count() == 0 && column.valid() != nullptr));

                writer.Key("cudf_column");
                serializeCudfColumn(writer, column.get_gdf_column());
            }
            writer.EndObject();
        }

        static std::string serializeToBinary(std::vector<RalColumn>& columns) {
            std::string result;

            std::size_t capacity = 0;
            for (const auto& column : columns) {
              if (!GpuFunctions::isGdfString(*column.get_gdf_column())) {
                capacity +=
                    GpuFunctions::getDataCapacity(column.get_gdf_column());
                capacity +=
                    GpuFunctions::getValidCapacity(column.get_gdf_column());
              }
              // WARNING!!! Here we are only getting the size for non-string columns. The size we need for string columns is determined inside the copyGpuToCpu where it is resized again.
              // THIS is a bad performance issue. This needs to be addressed
              // TODO!!
            }

            const typename GpuFunctions::StringsInfo* stringsInfo =
                GpuFunctions::createStringsInfo(columns);

            capacity += GpuFunctions::getStringsCapacity(stringsInfo);
            result.resize(capacity);

            std::vector<std::size_t> offsets;
            offsets.reserve(columns.size());

            offsets.push_back(0);

            for (const auto& column : columns) {
              std::size_t preOff = offsets.back();

              std::size_t dataSize = 0;
              std::size_t validSize = 0;

              if (GpuFunctions::isGdfString(*column.get_gdf_column())) {
                dataSize = GpuFunctions::getStringTotalSize(
                    stringsInfo, column.get_gdf_column());
              } else {
                dataSize =
                    GpuFunctions::getDataCapacity(column.get_gdf_column());
                validSize =
                    GpuFunctions::getValidCapacity(column.get_gdf_column());
              }

              offsets.push_back(preOff + dataSize + validSize);
            }

            std::vector<std::future<void>> futures;
            futures.reserve(columns.size());

            for (std::size_t i = 0; i < columns.size(); i++) {
              RalColumn& column = columns[i];
              const std::size_t offset = offsets[i];

              futures.push_back(std::async(
                  std::launch::async, GpuFunctions::copyGpuToCpu, offset,
                  std::ref(result), std::ref(column), stringsInfo));
            }

            for (std::future<void> &future : futures) {
              future.wait();
            }

            GpuFunctions::destroyStringsInfo(stringsInfo);

            return result;
        }

        static CudfColumn deserializeCudfColumn(rapidjson::Value::ConstObject&& object) {
            CudfColumn column;

            column.size = object["size"].GetUint64();

            column.dtype = (typename GpuFunctions::DType)object["dtype"].GetUint();

            column.null_count = object["null_count"].GetUint64();

            column.dtype_info.time_unit =
                static_cast<typename GpuFunctions::TimeUnit>(
                    object["dtype_info-time_unit"].GetUint());

            return column;
        }

        static RalColumn deserializeRalColumn(
            std::size_t                     binary_pointer,
            const std::string&              binary_data,
            rapidjson::Value::ConstObject&& object) {
          auto	start = std::chrono::high_resolution_clock::now();

            const auto& column_name_data = object["column_name"];
            std::string column_name(column_name_data.GetString(), column_name_data.GetStringLength());

            bool is_ipc = object["is_ipc"].GetBool();

            std::uint64_t column_token = object["column_token"].GetUint64();

            bool has_valids = object["has_valids"].GetBool();

            auto cudf_column = deserializeCudfColumn(object["cudf_column"].GetObject());

            RalColumn ral_column;

            std::size_t dtype_size = GpuFunctions::getDTypeSize(cudf_column.dtype);

            if (GpuFunctions::isGdfString(cudf_column)) {
              if (!cudf_column.size) {
                typename GpuFunctions::NvCategory* nvCategory =
                    GpuFunctions::NvCategory::create_from_array(nullptr, 0);
                ral_column.create_gdf_column(nvCategory, 0, column_name);
                return ral_column;
              }

              const std::size_t stringsSize =
                  *reinterpret_cast<const std::size_t*>(
                      &binary_data[binary_pointer]);
              const std::size_t offsetsSize =
                  *reinterpret_cast<const std::size_t*>(
                      &binary_data[binary_pointer + sizeof(const std::size_t)]);

              const std::size_t stringsIndex =
                  binary_pointer + 3 * sizeof(const std::size_t);
              const std::size_t offsetsIndex = stringsIndex + stringsSize;

              const void* stringsPointer =
                  reinterpret_cast<const typename GpuFunctions::NvStrings*>(
                      &binary_data[stringsIndex]);
              const void* offsetsPointer =
                  reinterpret_cast<const typename GpuFunctions::NvStrings*>(
                      &binary_data[offsetsIndex]);

              const std::size_t keysLength =
                  *reinterpret_cast<const std::size_t*>(
                      &binary_data[binary_pointer +
                                   2 * sizeof(const std::size_t)]);

              typename GpuFunctions::NvStrings* nvStrings =
                  GpuFunctions::CreateNvStrings(stringsPointer, offsetsPointer,
                                                keysLength);

              typename GpuFunctions::NvCategory* nvCategory =
                  GpuFunctions::NvCategory::create_from_strings(*nvStrings);

              binary_pointer +=
                  stringsSize + offsetsSize + 3 * sizeof(const std::size_t);

              ral_column.create_gdf_column(nvCategory, keysLength, column_name);
            } else {  // gdf is not string
              // Calculate pointers and update binary_pointer
              std::size_t data_pointer = binary_pointer;
              std::size_t valid_pointer =
                  data_pointer + GpuFunctions::getDataCapacity(&cudf_column);
              binary_pointer =
                  valid_pointer + GpuFunctions::getValidCapacity(&cudf_column);

              if (!is_ipc) {
                if(cudf_column.null_count > 0){
                    ral_column.create_gdf_column(cudf_column.dtype,
                                                 cudf_column.size,
                                                 (typename GpuFunctions::DataTypePointer)&binary_data[data_pointer],
                                                 (typename GpuFunctions::ValidTypePointer)&binary_data[valid_pointer],
                                                 dtype_size,
                                                 column_name);
            	} else if(has_valids) {
                    ral_column.create_gdf_column(cudf_column.dtype,
                                                 cudf_column.size,
                                                 (typename GpuFunctions::DataTypePointer)&binary_data[data_pointer],
                                                 dtype_size,
                                                 column_name);
                } else {
                    ral_column.create_gdf_column(cudf_column.dtype,
                                                 cudf_column.size,
                                                 (typename GpuFunctions::DataTypePointer)&binary_data[data_pointer],
                                                 (typename GpuFunctions::ValidTypePointer)nullptr,
                                                 dtype_size,
                                                 column_name);
            	}
              } else {
                ral_column.create_gdf_column_for_ipc(
                    cudf_column.dtype,
                    (typename GpuFunctions::DataTypePointer) &
                        binary_data[data_pointer],
                    (typename GpuFunctions::ValidTypePointer) &
                        binary_data[valid_pointer],
                    cudf_column.size, cudf_column.null_count, column_name);
              }

              ral_column.set_column_token(column_token);
              ral_column.get_gdf_column()->null_count = cudf_column.null_count;
              ral_column.get_gdf_column()->dtype_info = cudf_column.dtype_info;
            }
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end-start).count();

            GpuFunctions::log("-> deserializeRalColumn " + std::to_string(duration) + " ms" );
            return ral_column;
        }

        static std::vector<RalColumn> deserializeRalColumns(
            const std::string& binary,
            const rapidjson::GenericValue<rapidjson::UTF8<char>>::Array&
                gpu_data_array) {
          const std::size_t gpuDataSize =
              static_cast<const std::size_t>(gpu_data_array.Size());

          std::vector<std::size_t> offsets;
          offsets.reserve(gpuDataSize);

          std::size_t lastOffset = 0;

          auto last = gpu_data_array.end() - 1;
          for (const auto& gpu_data : gpu_data_array) {
            offsets.push_back(lastOffset);

            if (&gpu_data == last) {
              break;
            }

            CudfColumn cudfColumn = deserializeCudfColumn(
                gpu_data.GetObject()["cudf_column"].GetObject());

            if (GpuFunctions::isGdfString(cudfColumn)) {
              if (!cudfColumn.size) {
                continue;
              }

              const std::size_t stringsSize =
                  *reinterpret_cast<const std::size_t*>(&binary[lastOffset]);
              const std::size_t offsetsSize =
                  *reinterpret_cast<const std::size_t*>(
                      &binary[lastOffset + sizeof(const std::size_t)]);

              lastOffset +=
                  stringsSize + offsetsSize + 3 * sizeof(const std::size_t);
            } else {
              lastOffset += GpuFunctions::getDataCapacity(&cudfColumn) +
                            GpuFunctions::getValidCapacity(&cudfColumn);
            }
          }

          std::vector<std::future<RalColumn>> futures;
          futures.reserve(gpuDataSize);

          const auto& futurePairs = zip(offsets, gpu_data_array);
          std::transform(
              futurePairs.begin(),
              futurePairs.end(),
              std::back_inserter(futures),
              [&binary](const boost::tuples::cons<
                        unsigned long&,
                        boost::tuples::cons<rapidjson::GenericValue<
                                                rapidjson::UTF8<char>,
                                                rapidjson::MemoryPoolAllocator<
                                                    rapidjson::CrtAllocator>>&,
                                            boost::tuples::null_type>>& pair) {
                std::size_t offset = pair.get<0>();
                const rapidjson::GenericValue<
                    rapidjson::UTF8<char>,
                    rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator>>&
                    gpu_data = pair.get<1>();

                return std::async(std::launch::async,
                                  deserializeRalColumn,
                                  offset,
                                  std::ref(binary),
                                  std::move(gpu_data.GetObject()));
              });

          std::vector<RalColumn> columns;
          columns.reserve(gpuDataSize);
          std::transform(futures.begin(),
                         futures.end(),
                         std::back_inserter(columns),
                         [](std::future<RalColumn>& future) {
                           future.wait();
                           return future.get();
                         });

          return columns;
        }
    };

} // namespace messages
} // namespace communication
} // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_MESSAGES_COMPONENTMESSAGE_H
