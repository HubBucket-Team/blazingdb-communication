#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTDTYPEINFOPAYLOAD_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTDTYPEINFOPAYLOAD_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

#include "../mixins/ActualBufferMixIn.hpp"
#include "DTypeInfoPayloadInHostBase.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT InHostDTypeInfoPayload : public ActualBufferMixIn,
                                           public DTypeInfoPayloadInHostBase {
  UC_CONCRETE(InHostDTypeInfoPayload);

public:
  explicit InHostDTypeInfoPayload(const std::string&& content);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
