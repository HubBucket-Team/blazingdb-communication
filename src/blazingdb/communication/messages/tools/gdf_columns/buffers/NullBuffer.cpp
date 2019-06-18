#include "NullBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

NullBuffer::NullBuffer() = default;

const void*
NullBuffer::Data() const noexcept {
  return nullptr;
}

std::size_t
NullBuffer::Size() const noexcept {
  return 0;
}

bool
NullBuffer::IsNull() const noexcept {
  return true;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
