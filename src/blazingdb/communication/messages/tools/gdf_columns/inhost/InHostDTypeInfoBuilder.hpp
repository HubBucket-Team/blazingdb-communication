#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTDTYPEINFOBUILDER_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTDTYPEINFOBUILDER_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT InHostDTypeInfoBuilder : public DTypeInfoBuilder {
  UC_CONCRETE(InHostDTypeInfoBuilder);

public:
  // TODO(improve): remove unnecessary agent
  explicit InHostDTypeInfoBuilder(blazingdb::uc::Agent &agent);

  std::unique_ptr<Payload>
  Build() const noexcept final;

  DTypeInfoBuilder &
  TimeUnit(const std::int_fast32_t timeUnit) noexcept final;

  DTypeInfoBuilder &
  Category(const Payload &categoryPayload) noexcept final;

private:
  std::int_fast32_t timeUnit_;
  const Payload *   categoryPayload_;

  blazingdb::uc::Agent &agent_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
