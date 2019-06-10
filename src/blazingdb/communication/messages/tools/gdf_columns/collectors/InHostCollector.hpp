#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_COLLECTORS_INHOSTCOLLECTOR_H_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_COLLECTORS_INHOSTCOLLECTOR_H_

#include "../../gdf_columns.h"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class InHostCollector : public Specialized {
public:
  explicit InHostCollector(const Buffer& buffer);

  std::unique_ptr<Payload>
  Apply() const final;

private:
  const Buffer& buffer_;

  UC_CONCRETE(InHostCollector);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
