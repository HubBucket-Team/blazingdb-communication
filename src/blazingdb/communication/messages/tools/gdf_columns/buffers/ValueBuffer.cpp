#include "ValueBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

const void*
ValueBuffer::Data() const noexcept {
  return data_;
}

std::size_t
ValueBuffer::Size() const noexcept {
  return size_;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
