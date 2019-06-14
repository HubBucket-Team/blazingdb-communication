#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNDISPATCHER_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNDISPATCHER_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class InHostGdfColumnDispatcher : public GdfColumnDispatcher {
public:
  explicit InHostGdfColumnDispatcher(const Buffer &buffer);

  std::unique_ptr<Collector> Dispatch() const final;

private:
  const Buffer &buffer_;

  UC_CONCRETE(InHostGdfColumnDispatcher);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
