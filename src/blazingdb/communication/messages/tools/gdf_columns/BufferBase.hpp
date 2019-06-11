#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERBASE_H_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERBASE_H_

#include "../gdf_columns.h"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT BufferBase : public NullableBuffer {
public:
  explicit BufferBase(const void* const data, const std::size_t size);

  const void*
  Data() const noexcept final;

  std::size_t
  Size() const noexcept final;

  bool
  IsNull() const noexcept final;

  static std::unique_ptr<BufferBase>
  MakeNull() noexcept;

private:
  const void* const data_;
  const std::size_t size_;

  UC_CONCRETE(BufferBase);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
