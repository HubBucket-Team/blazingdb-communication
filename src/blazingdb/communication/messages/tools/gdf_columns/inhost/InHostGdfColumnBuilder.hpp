#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNBUILDER_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNBUILDER_HPP_

#include "../../gdf_columns.h"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT InHostGdfColumnBuilder : public GdfColumnBuilder {
  UC_CONCRETE(InHostGdfColumnBuilder);

public:
  explicit InHostGdfColumnBuilder(blazingdb::uc::Agent &agent);

  std::unique_ptr<Payload>
  Build() const noexcept final;

  GdfColumnBuilder &
  Data(const CudaBuffer &cudaBuffer) noexcept final;

  GdfColumnBuilder &
  Valid(const CudaBuffer &cudaBuffer) noexcept final;

  GdfColumnBuilder &
  Size(const std::size_t size) noexcept final;

  GdfColumnBuilder &
  DType(const std::int_fast32_t dtype) noexcept final;

  GdfColumnBuilder &
  NullCount(const std::size_t nullCount) noexcept final;

  GdfColumnBuilder &
  DTypeInfo(const DTypeInfoPayload &dtypeInfoPayload) noexcept final;

  GdfColumnBuilder &
  ColumnName(const HostBuffer &hostBuffer) noexcept final;

private:
  const CudaBuffer *dataCudaBuffer_;
  const CudaBuffer *validCudaBuffer_;
  std::size_t       size_;

  blazingdb::uc::Agent &agent_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
