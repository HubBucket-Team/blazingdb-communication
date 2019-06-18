#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_CUDAMEMORYRUNTIME_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_CUDAMEMORYRUNTIME_HPP_

#include "interfaces.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT CudaMemoryRuntime : public MemoryRuntime {
  UC_CONCRETE(CudaMemoryRuntime);

public:
  explicit CudaMemoryRuntime();

  void*
  Allocate(const std::size_t size) final;

  void
  Synchronize() final;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
