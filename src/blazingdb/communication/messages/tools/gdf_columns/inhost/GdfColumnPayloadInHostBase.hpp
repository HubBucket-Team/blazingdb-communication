#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_GDFCOLUMNINHOSTBASE_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_GDFCOLUMNINHOSTBASE_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT GdfColumnPayloadInHostBase : public GdfColumnPayload {
  UC_CONCRETE(GdfColumnPayloadInHostBase);

public:
  explicit GdfColumnPayloadInHostBase(const Buffer& buffer);

  const UCBuffer&
  Data() const noexcept final;

  const UCBuffer&
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
  const Buffer& buffer_;

  std::unique_ptr<Buffer> dataBuffer_;
  std::unique_ptr<Buffer> validBuffer_;
  const std::size_t*      size_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
