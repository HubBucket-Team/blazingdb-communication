#include "InHostGdfColumnSpecialized.hpp"

#include "InHostGdfColumnPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostGdfColumnSpecialized::InHostGdfColumnSpecialized(const Buffer &buffer)
    : buffer_{buffer} {}

std::unique_ptr<Payload>
InHostGdfColumnSpecialized::Apply() const {
  return std::make_unique<GdfColumnPayloadInHostBase>(buffer_);
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
