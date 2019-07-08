#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_

/// Helpers to (de)serialize gdf columns to transport between RALs.
///
/// The function `DeliverFrom` converts a collection of gdf columns
/// to a transportable buffer.
///
/// The funtion `CollectFrom` converts a transported buffer to a collection of
/// gdf columns.

#include <cstring>
#include <iostream>
#include <vector>

#include "gdf_columns/interfaces.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

template <template <class gdf_column> class GdfColumnInfo, class gdf_column, class nvcategory, class nvstrings_transfer>
std::string
DeliverFrom(const std::vector<gdf_column> &gdfColumns,
            blazingdb::uc::Agent &         agent) {
  std::unique_ptr<GdfColumnCollector> collector =
      GdfColumnCollector::MakeInHost();

  std::vector<std::unique_ptr<Payload>> payloads;
  payloads.reserve(gdfColumns.size());

  std::vector<std::unique_ptr<Payload>> categoryPayloads;
  categoryPayloads.reserve(gdfColumns.size());

  std::vector<std::unique_ptr<Payload>> dtypeInfoPayloads;
  dtypeInfoPayloads.reserve(gdfColumns.size());

  std::vector<std::unique_ptr<const CudaBuffer>> dataBuffers;
  std::vector<std::unique_ptr<const CudaBuffer>> validBuffers;
  std::vector<std::unique_ptr<const HostBuffer>> columnNameBuffers;

  dataBuffers.reserve(gdfColumns.size());
  validBuffers.reserve(gdfColumns.size());
  columnNameBuffers.reserve(gdfColumns.size());

  std::vector<std::unique_ptr<const CudaBuffer>> strsBuffers;
  std::vector<std::unique_ptr<const CudaBuffer>> memBuffers;
  std::vector<std::unique_ptr<const CudaBuffer>> mapBuffers;

  strsBuffers.reserve(gdfColumns.size());
  memBuffers.reserve(gdfColumns.size());
  mapBuffers.reserve(gdfColumns.size());

  for (const auto &gdfColumn : gdfColumns) {
    // auto *column_ptr = gdfColumn.get_gdf_column();

    std::unique_ptr<GdfColumnBuilder> builder =
        GdfColumnBuilder::MakeInHost(agent);

    // TODO: Add other members y compute correct buffer size
    dataBuffers.emplace_back(CudaBuffer::Make(
        gdfColumn.data, GdfColumnInfo<gdf_column>::DataSize(gdfColumn)));

    validBuffers.emplace_back(CudaBuffer::Make(
        gdfColumn.valid, GdfColumnInfo<gdf_column>::ValidSize(gdfColumn)));

    columnNameBuffers.emplace_back(
        HostBuffer::Make(gdfColumn.col_name, std::strlen(gdfColumn.col_name)));

    const std::size_t       size      = gdfColumn.size;
    const std::int_fast32_t dtype     = gdfColumn.dtype;
    const std::size_t       nullCount = gdfColumn.null_count;

    nvstrings_transfer handles;
    static_cast<nvcategory *>(gdfColumn.dtype_info.category)->create_transfer(handles);

    strsBuffers.emplace_back(CudaBuffer::Make(handles.strs, handles.keys));

    memBuffers.emplace_back(CudaBuffer::Make(handles.mem, handles.size));

    mapBuffers.emplace_back(CudaBuffer::Make(handles.vals, handles.count));

    const std::size_t count = handles.count;
    const std::size_t keys = handles.keys;
    const std::size_t size2 = handles.size;
    const std::size_t base = reinterpret_cast<std::size_t>(handles.base_address);

    std::unique_ptr<CategoryBuilder> categoryBuilder =
        CategoryBuilder::MakeInHost(agent);

    auto categoryPayload = categoryBuilder->Strs(*strsBuffers.back())
                               .Mem(*memBuffers.back())
                               .Map(*mapBuffers.back())
                               .Count(count)
                               .Keys(keys)
                               .Size(size2)
                               .BaseAddress(base)
                               .Build();

    categoryPayloads.emplace_back(std::move(categoryPayload));

    std::unique_ptr<DTypeInfoBuilder> dtypeInfoBuilder =
        DTypeInfoBuilder::MakeInHost();

    std::unique_ptr<Payload> dtypeInfoPayload;

    if (dtype == GdfColumnInfo<gdf_column>::gdf_dtype::GDF_STRING_CATEGORY) {
      dtypeInfoPayload =
          dtypeInfoBuilder->TimeUnit(gdfColumn.dtype_info.time_unit)
              .Category(*categoryPayloads.back())
              .Build();
    } else {
      dtypeInfoPayload =
          dtypeInfoBuilder->TimeUnit(gdfColumn.dtype_info.time_unit).Build();
    }

    dtypeInfoPayloads.emplace_back(std::move(dtypeInfoPayload));

    // TODO(potential bug): optional setters
    if (nullCount > 0) {
      payloads.emplace_back(builder->Data(*dataBuffers.back())
                                .Valid(*validBuffers.back())
                                .Size(size)
                                .DType(dtype)
                                .NullCount(nullCount)
                                .ColumnName(*columnNameBuffers.back())
                                .DTypeInfo(*dtypeInfoPayloads.back())
                                .Build());
    } else {
      payloads.emplace_back(builder->Data(*dataBuffers.back())
                                .Size(size)
                                .DType(dtype)
                                .NullCount(nullCount)
                                .ColumnName(*columnNameBuffers.back())
                                .DTypeInfo(*dtypeInfoPayloads.back())
                                .Build());
    }

    collector->Add(payloads.back()->Deliver());
  }

  std::unique_ptr<Buffer> resultBuffer = collector->Collect();

  return std::string{
      static_cast<const std::string::value_type *>(resultBuffer->Data()),
      resultBuffer->Size()};
}

std::string
StringFrom(const Buffer &buffer) UC_DEPRECATED;

namespace {
class StringViewBuffer : public Buffer {
public:
  explicit StringViewBuffer(const std::string &content) : content_{content} {}

  const void *
  Data() const noexcept final {
    return content_.data();
  }

  std::size_t
  Size() const noexcept final {
    return content_.length();
  }

private:
  const std::string &content_;
};
}  // namespace

template <template <class gdf_column> class GdfColumnInfo, class gdf_column, class nvcategory, class nvstrings_transfer>
std::vector<gdf_column>
CollectFrom(const std::string &content, blazingdb::uc::Agent &agent) {
  const Buffer &buffer = StringViewBuffer{content};

  std::unique_ptr<GdfColumnDispatcher> dispatcher =
      GdfColumnDispatcher::MakeInHost(buffer);

  std::unique_ptr<Collector> collector = dispatcher->Dispatch();

  std::vector<gdf_column> gdfColumns;
  gdfColumns.reserve(collector->Length());

  for (const Buffer &buffer : *collector) {
    std::unique_ptr<Specialized> specialized =
        GdfColumnSpecialized::MakeInHost(buffer);

    const std::unique_ptr<const Payload> payload = specialized->Apply();

    std::unique_ptr<GdfColumnValue> gdfColumnValue = GdfColumnValue::Make(
        static_cast<const GdfColumnPayload &>(*payload), agent);

    if (gdfColumnValue->dtype() ==
        GdfColumnInfo<gdf_column>::gdf_dtype::GDF_STRING_CATEGORY) {
      const CategoryValue &categoryValue =
          gdfColumnValue->dtype_info().category();
      nvcategory *       category = 0;
      nvstrings_transfer handles;
      handles.base_address =
          reinterpret_cast<char *>(categoryValue.base_address());
      handles.mem   = const_cast<void *>(categoryValue.mem());
      handles.strs  = const_cast<void *>(categoryValue.strs());
      handles.vals  = const_cast<void *>(categoryValue.map());
      handles.size  = categoryValue.size();
      handles.count = categoryValue.count();
      handles.keys  = categoryValue.keys();

      category = nvcategory::create_from_transfer(handles);

      gdfColumns.emplace_back(gdf_column{
          const_cast<void *>(gdfColumnValue->data()),
          reinterpret_cast<unsigned char *>(
              const_cast<void *>(gdfColumnValue->valid())),
          static_cast<int>(gdfColumnValue->size()),
          static_cast<decltype(gdf_column::dtype)>(gdfColumnValue->dtype()),
          static_cast<int>(gdfColumnValue->null_count()),
          {static_cast<decltype(decltype(gdf_column::dtype_info)::time_unit)>(
               gdfColumnValue->dtype_info().time_unit()),
           static_cast<void *>(category)},
          const_cast<char *>(gdfColumnValue->column_name()),
      });
    } else {
      gdfColumns.emplace_back(gdf_column{
          const_cast<void *>(gdfColumnValue->data()),
          reinterpret_cast<unsigned char *>(
              const_cast<void *>(gdfColumnValue->valid())),
          static_cast<int>(gdfColumnValue->size()),
          static_cast<decltype(gdf_column::dtype)>(gdfColumnValue->dtype()),
          static_cast<int>(gdfColumnValue->null_count()),
          {static_cast<decltype(decltype(gdf_column::dtype_info)::time_unit)>(
              gdfColumnValue->dtype_info().time_unit())},
          const_cast<char *>(gdfColumnValue->column_name()),
      });
    }
  }

  return gdfColumns;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
