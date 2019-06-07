#include "InHostGdfColumnPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostGdfColumnPayload::InHostGdfColumnPayload() : buffer_{nullptr, 0} {};

const Buffer&
InHostGdfColumnPayload::Deliver() const noexcept {
  return buffer_;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
