#include "interfaces.hpp"

#include "CudaMemoryRuntime.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

std::unique_ptr<MemoryRuntime>
MemoryRuntime::MakeCuda() {
  return std::make_unique<CudaMemoryRuntime>();
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
