#include "BlankPayload.hpp"

#include "../buffers/NullBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

BlankPayload::BlankPayload() = default;

const Buffer&
BlankPayload::Deliver() const noexcept {
  return NullBuffer::Buffer();
}

const Payload&
BlankPayload::Payload() noexcept {
  static BlankPayload payload;
  return payload;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
