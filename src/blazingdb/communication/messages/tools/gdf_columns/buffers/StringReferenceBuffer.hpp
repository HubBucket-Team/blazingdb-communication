#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_STRINGREFERENCEBUFFER_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_STRINGREFERENCEBUFFER_HPP_

#include "../../gdf_columns.h"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT StringReferenceBuffer : public Buffer {
  UC_CONCRETE(StringReferenceBuffer);

public:
  explicit StringReferenceBuffer(const std::string& content);

  const void*
  Data() const noexcept final;

  std::size_t
  Size() const noexcept final;

private:
  const std::string& content_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif

