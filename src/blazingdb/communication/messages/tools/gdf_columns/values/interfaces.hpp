#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_INTERFACES_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_INTERFACES_HPP_

#include <memory>

#include <blazingdb/uc/util/macros.hpp>

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT MemoryRuntime {
  UC_INTERFACE(MemoryRuntime);

public:
  virtual void*
  Allocate(const std::size_t size) = 0;

  virtual void
  Synchronize() = 0;

  static std::unique_ptr<MemoryRuntime>
  MakeCuda();
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
