#include "InHostCategoryPayload.hpp"

#include "../buffers/StringBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostCategoryPayload::InHostCategoryPayload(const std::string&& content)
    : ActualBufferMixIn{std::make_unique<StringBuffer>(std::move(content))},
      CategoryPayloadInHostBase{buffer()} {}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
