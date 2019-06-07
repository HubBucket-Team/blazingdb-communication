#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNPAYLOAD_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNPAYLOAD_HPP_

#include "../../gdf_columns.h"
#include "../BufferBase.hpp"

#include <blazingdb/uc/API.hpp>
#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class InHostGdfColumnPayload : public GdfColumnPayload {
public:
  explicit InHostGdfColumnPayload(
      std::unique_ptr<blazingdb::uc::Context>&& context,
      const std::string&&                       payload);

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
  UCBuffer*         ucBuffer_;
  DTypeInfoPayload* dtypeInfoPayload_;

  BufferBase buffer_;

  std::unique_ptr<blazingdb::uc::Context> context_;

  const std::string payload_;

  UC_CONCRETE(InHostGdfColumnPayload);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
