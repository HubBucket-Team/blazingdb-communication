#include "StringReferenceBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

StringReferenceBuffer::StringReferenceBuffer(const std::string& content)
    : content_{content} {}

const void*
StringReferenceBuffer::Data() const noexcept {
  return content_.data();
}

std::size_t
StringReferenceBuffer::Size() const noexcept {
  return content_.length();
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
