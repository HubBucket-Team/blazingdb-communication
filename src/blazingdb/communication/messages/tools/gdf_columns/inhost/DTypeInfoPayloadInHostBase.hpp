#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_DTYPEINFOINHOSTBASE_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_DTYPEINFOINHOSTBASE_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT DTypeInfoPayloadInHostBase : public DTypeInfoPayload {
  UC_CONCRETE(DTypeInfoPayloadInHostBase);

public:
  explicit DTypeInfoPayloadInHostBase(const Buffer& buffer);

  std::int_fast32_t
  TimeUnit() const noexcept final;

  const CategoryPayload&
  Category() const noexcept final;

  const Buffer&
  Deliver() const noexcept final;

private:
  const Buffer& buffer_;

  const std::int_fast32_t* timeUnit_;
  const CategoryPayload*   categoryPayload_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
