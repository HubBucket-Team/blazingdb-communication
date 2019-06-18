#include "InHostDTypeInfoSpecialized.hpp"

#include "InHostDTypeInfoPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostDTypeInfoSpecialized::InHostDTypeInfoSpecialized(const Buffer &buffer)
    : buffer_{buffer} {}

std::unique_ptr<Payload>
InHostDTypeInfoSpecialized::Apply() const {
  return std::make_unique<DTypeInfoPayloadInHostBase>(buffer_);
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
