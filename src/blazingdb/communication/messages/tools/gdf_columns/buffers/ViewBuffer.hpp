#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_VIEWBUFFER_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_VIEWBUFFER_HPP_

#include "../../gdf_columns.h"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT ViewBuffer : public Buffer {
public:
  explicit ViewBuffer(const void *const data, const std::size_t size);

  const void *
  Data() const noexcept final;

  std::size_t
  Size() const noexcept final;

private:
  const void *const data_;
  const std::size_t size_;

  UC_CONCRETE(ViewBuffer);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
