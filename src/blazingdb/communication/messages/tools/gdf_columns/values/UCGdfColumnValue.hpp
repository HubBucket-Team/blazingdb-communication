#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_UCGDFCOLUMNVALUE_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_UCGDFCOLUMNVALUE_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

#include "interfaces.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT UCGdfColumnValue : public GdfColumnValue {
  UC_CONCRETE(UCGdfColumnValue);

public:
  explicit UCGdfColumnValue(std::unique_ptr<MemoryRuntime> memoryRuntime,
                            const GdfColumnPayload&        gdfColumnPayload,
                            blazingdb::uc::Agent&          agent);

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

  std::unique_ptr<blazingdb::uc::Buffer> dataUcBuffer_;
  std::unique_ptr<blazingdb::uc::Buffer> validUcBuffer_;
  
  std::unique_ptr<blazingdb::uc::Transport> dataUcTransport_;
  std::unique_ptr<blazingdb::uc::Transport> validUcTransport_;

  const void* data_;
  const void* valid_;

  blazingdb::uc::Agent&          agent_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
