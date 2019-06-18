#include "InHostCategorySpecialized.hpp"

#include "InHostCategoryPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostCategorySpecialized::InHostCategorySpecialized(const Buffer &buffer)
    : buffer_{buffer} {}

std::unique_ptr<Payload>
InHostCategorySpecialized::Apply() const {
  return std::make_unique<CategoryPayloadInHostBase>(buffer_);
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
