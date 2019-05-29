#pragma once 

#include <blazingdb/uc/Context.hpp>

#include "blazingdb/uc/internal/ManagedContext.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NOEXPORT ViewContext : public Context {
public:
  ViewContext() 
    {}

  ~ViewContext() = default;

  std::unique_ptr<uc::Agent>
  OwnAgent() const final {
    return nullptr;
  }

  std::unique_ptr<uc::Agent>
  PeerAgent() const final {
    return nullptr;
  }

 std::size_t
  serializedRecordSize() const noexcept final;

  std::unique_ptr<uc::Agent> Agent() const final;
};

} // namespace internal
} // namespace uc
} // namespace blazingdb