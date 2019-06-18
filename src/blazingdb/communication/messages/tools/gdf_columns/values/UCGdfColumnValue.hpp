#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_UCGDFCOLUMNVALUE_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_UCGDFCOLUMNVALUE_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT UCGdfColumnValue : public GdfColumnValue {
  UC_CONCRETE(UCGdfColumnValue);

public:
  explicit UCGdfColumnValue(const GdfColumnPayload& gdfColumnPayload,
                            blazingdb::uc::Agent&   agent);

  const void*
  data() const noexcept final;

  const void*
  valid() const noexcept final;

  std::size_t
  size() const noexcept final;

  std::int_fast32_t
  dtype() const noexcept final;

  std::size_t
  null_count() const noexcept final;

  const DTypeInfoValue&
  dtype_info() const noexcept final;

  const char*
  column_name() const noexcept final;

private:
  const GdfColumnPayload& gdfColumnPayload_;

  const void* const data_;
  const void* const valid_;

  std::unique_ptr<blazingdb::uc::Buffer> dataUcBuffer_;
  std::unique_ptr<blazingdb::uc::Buffer> validUcBuffer_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
