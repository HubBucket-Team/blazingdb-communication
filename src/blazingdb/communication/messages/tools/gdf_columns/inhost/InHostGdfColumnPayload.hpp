#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNPAYLOAD_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNPAYLOAD_HPP_

#include "../../gdf_columns.h"

#include "../buffers/StringReferenceBuffer.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT InHostGdfColumnPayload : public GdfColumnPayload {
public:
  explicit InHostGdfColumnPayload(const std::string&& content);

  const UCBuffer&
  Data() const noexcept final;

  UCBuffer&
  Valid() const noexcept final;

  std::size_t
  Size() const noexcept final;

  std::int_fast32_t
  DType() noexcept final;

  std::size_t
  NullCount() const noexcept final;

  DTypeInfoPayload&
  DTypeInfo() const noexcept final;

  std::string
  ColumnName() const noexcept final;

  const Buffer&
  Deliver() const noexcept final;

private:
  const std::string           content_;
  const StringReferenceBuffer buffer_;

  UC_CONCRETE(InHostGdfColumnPayload);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
