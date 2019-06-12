#include "InHostGdfColumnSpecialized.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostGdfColumnSpecialized::InHostGdfColumnSpecialized(const Buffer &buffer)
    : buffer_{buffer} {}

std::unique_ptr<Payload>
InHostGdfColumnSpecialized::Apply() const {
  return nullptr;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
