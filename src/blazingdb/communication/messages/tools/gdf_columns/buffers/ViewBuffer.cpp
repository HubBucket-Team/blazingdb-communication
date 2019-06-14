#include "ViewBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

ViewBuffer::ViewBuffer(const void *const data, const std::size_t size)
    : data_{data}, size_{size} {}

const void *
ViewBuffer::Data() const noexcept {
  return data_;
}

std::size_t
ViewBuffer::Size() const noexcept {
  return size_;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
