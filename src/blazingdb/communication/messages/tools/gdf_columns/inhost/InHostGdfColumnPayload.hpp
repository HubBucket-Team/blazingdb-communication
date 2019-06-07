#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNPAYLOAD_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNPAYLOAD_HPP_

#include "../../gdf_columns.h"
#include "../BufferBase.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class InHostGdfColumnPayload : public GdfColumnPayload {
public:
  explicit InHostGdfColumnPayload();

  const Buffer&
  Deliver() const noexcept final;

private:
  BufferBase buffer_;

  UC_CONCRETE(InHostGdfColumnPayload);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
