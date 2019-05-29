#include "ViewContext.hpp"

#include "agents/ViewAgent.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace blazingdb {
namespace uc {
namespace internal {


std::unique_ptr<uc::Agent> ViewContext::Agent() const {
 return std::make_unique<ViewAgent>();
}

std::size_t
ViewContext::serializedRecordSize() const noexcept {
  return sizeof(cudaIpcMemHandle_t);
}


}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
