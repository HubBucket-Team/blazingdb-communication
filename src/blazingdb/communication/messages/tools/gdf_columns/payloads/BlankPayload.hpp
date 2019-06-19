#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_PAYLOADS_BLANKPAYLOAD_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_PAYLOADS_BLANKPAYLOAD_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT BlankPayload : public Payload {
  UC_CONCRETE(BlankPayload);

public:
  explicit BlankPayload();

  const Buffer&
  Deliver() const noexcept final;

  static const Payload&
  Payload() noexcept;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
