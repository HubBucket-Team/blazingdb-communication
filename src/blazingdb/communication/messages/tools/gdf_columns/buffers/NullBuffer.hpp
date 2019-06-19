#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_NULLBUFFER_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_NULLBUFFER_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT NullBuffer : public NullableBuffer {
  UC_CONCRETE(NullBuffer);

public:
  explicit NullBuffer();

  const void*
  Data() const noexcept final;

  std::size_t
  Size() const noexcept final;

  bool
  IsNull() const noexcept final;

  static const NullBuffer&
  Buffer() noexcept;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
