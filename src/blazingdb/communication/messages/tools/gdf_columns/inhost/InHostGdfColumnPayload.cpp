#include "InHostGdfColumnPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

std::unique_ptr<Buffer>
InHostGdfColumnPayload::Deliver() const noexcept {
  return nullptr;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
