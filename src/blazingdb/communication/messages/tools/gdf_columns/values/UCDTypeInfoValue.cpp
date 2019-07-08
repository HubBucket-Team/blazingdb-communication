#include "UCDTypeInfoValue.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

#include <cuda_runtime_api.h>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

UCDTypeInfoValue::UCDTypeInfoValue(std::unique_ptr<MemoryRuntime> memoryRuntime,
                                   const DTypeInfoPayload& dtypeInfoPayload,
                                   blazingdb::uc::Agent&   agent)
    : dtypeInfoPayload_{dtypeInfoPayload} {}

std::int_fast32_t
UCDTypeInfoValue::time_unit() const noexcept {
  return dtypeInfoPayload_.TimeUnit();
}

const CategoryValue&
UCDTypeInfoValue::category() const noexcept {
  static CategoryValue* categoryValue_;
  UC_ABORT("Not support");
  return *categoryValue_;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
