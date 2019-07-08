#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_UCDTYPEINFOVALUE_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_UCDTYPEINFOVALUE_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

#include "interfaces.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT UCDTypeInfoValue : public DTypeInfoValue {
  UC_CONCRETE(UCDTypeInfoValue);

public:
  explicit UCDTypeInfoValue(std::unique_ptr<MemoryRuntime> memoryRuntime,
                            const DTypeInfoPayload&        dtypeInfoPayload,
                            blazingdb::uc::Agent&          agent);

  std::int_fast32_t
  time_unit() const noexcept final;

  const CategoryValue&
  category() const noexcept final;

private:
  const DTypeInfoPayload& dtypeInfoPayload_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
