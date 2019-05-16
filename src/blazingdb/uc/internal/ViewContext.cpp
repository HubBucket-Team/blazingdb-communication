#include "ViewContext.hpp"

#include "agents/ViewAgent.hpp"

namespace blazingdb {
namespace uc {
namespace internal {


std::unique_ptr<uc::Agent> ViewContext::Agent() const {
 return std::make_unique<ViewAgent>();
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
