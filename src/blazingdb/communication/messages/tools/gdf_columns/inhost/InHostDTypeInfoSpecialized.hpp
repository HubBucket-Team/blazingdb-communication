#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTDTYPEINFOSPECIALIZED_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTDTYPEINFOSPECIALIZED_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT InHostDTypeInfoSpecialized : public Specialized {
  UC_CONCRETE(InHostDTypeInfoSpecialized);

public:
  explicit InHostDTypeInfoSpecialized(const Buffer &buffer);

  std::unique_ptr<Payload>
  Apply() const final;

private:
  const Buffer &buffer_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
