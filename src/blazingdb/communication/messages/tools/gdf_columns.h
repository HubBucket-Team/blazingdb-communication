#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_

#include <iostream>
#include <vector>

#include "gdf_columns/interfaces.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

template <class GdfColumnInfo, class gdf_column>
std::string
DeliverFrom(const std::vector<gdf_column> &gdfColumns,
            blazingdb::uc::Agent &         agent) {
  std::unique_ptr<GdfColumnCollector> collector =
      GdfColumnCollector::MakeInHost();

  std::vector<std::unique_ptr<Payload>> payloads;
  payloads.reserve(gdfColumns.size());

  for (const auto &gdfColumn : gdfColumns) {
    // auto *column_ptr = gdfColumn.get_gdf_column();

    std::unique_ptr<GdfColumnBuilder> builder =
        GdfColumnBuilder::MakeInHost(agent);

    // TODO: Add other members y compute correct buffer size
    const std::unique_ptr<const CudaBuffer> dataBuffer =
        CudaBuffer::Make(gdfColumn.data, GdfColumnInfo::DataSize(gdfColumn));
    const std::unique_ptr<const CudaBuffer> validBuffer =
        CudaBuffer::Make(gdfColumn.valid, GdfColumnInfo::ValidSize(gdfColumn));
    const std::size_t size = gdfColumn.size;

    // TODO: support different buffer sizes (of payloads) in
    // GdfColumnCollector

    payloads.emplace_back(
        builder->Data(*dataBuffer).Valid(*validBuffer).Size(size).Build());

    collector->Add(*payloads.back());
  }

  std::unique_ptr<Buffer> resultBuffer = collector->Collect();

  return std::string{
      static_cast<const std::string::value_type *const>(resultBuffer->Data()),
      resultBuffer->Size()};
}

std::string
StringFrom(const Buffer &buffer);

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

template <class gdf_column>
std::vector<gdf_column>
CollectFrom(const std::string &content, blazingdb::uc::Agent &agent) {
  const Buffer &buffer = StringViewBuffer{content};

  std::unique_ptr<GdfColumnDispatcher> dispatcher =
      GdfColumnDispatcher::MakeInHost(buffer);

  std::unique_ptr<Collector> collector = dispatcher->Dispatch();

  std::vector<gdf_column> gdfColumns;
  gdfColumns.reserve(collector->Length());

  for (Collector::iterator it = collector->begin(); it != collector->end();
       ++it) {
    GdfColumnPayload &payload = *it;
    gdf_column        col{.data  = payload.Data().Data(),
                   .valid = payload.Valid().Data(),
                   .size  = payload.Size()};
    gdfColumns.push_back(col);
  }

  return gdfColumns;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
