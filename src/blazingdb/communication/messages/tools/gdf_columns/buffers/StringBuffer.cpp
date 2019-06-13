#include "StringBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

StringBuffer::StringBuffer(const std::string &&content)
    : content_{std::move(content)} {}

const void *
StringBuffer::Data() const noexcept {
  return content_.data();
}

std::size_t
StringBuffer::Size() const noexcept {
  return content_.length();
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
