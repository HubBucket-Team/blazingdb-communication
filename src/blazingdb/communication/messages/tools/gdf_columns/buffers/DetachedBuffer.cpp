#include "DetachedBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

DetachedBuffer::DetachedBuffer(const std::string &&content)
    : content_{std::move(content)} {}

const void *
DetachedBuffer::Data() const noexcept {
  return content_.data();
}

std::size_t
DetachedBuffer::Size() const noexcept {
  return content_.size();
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
