#include "DetachedBufferBuilder.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

std::unique_ptr<Buffer>
DetachedBufferBuilder::Build() const noexcept {
  std::string content = ss_.str();
  return std::make_unique<DetachedBuffer>(std::move(content));
}

DetachedBufferBuilder &
DetachedBufferBuilder::Add(const Buffer &buffer) {
  ss_.write(static_cast<const std::ostringstream::char_type *>(buffer.Data()),
            buffer.Size());
  return *this;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
