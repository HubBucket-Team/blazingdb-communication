#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTCATEGORYPAYLOAD_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTCATEGORYPAYLOAD_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

#include "../mixins/ActualBufferMixIn.hpp"
#include "CategoryPayloadInHostBase.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT InHostCategoryPayload : public ActualBufferMixIn,
                                           public CategoryPayloadInHostBase {
  UC_CONCRETE(InHostCategoryPayload);

public:
  explicit InHostCategoryPayload(const std::string&& content);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
