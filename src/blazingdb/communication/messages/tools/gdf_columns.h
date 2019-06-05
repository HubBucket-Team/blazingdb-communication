#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_

#include <memory>

#include <blazingdb/uc/util/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class Buffer {};
class HostBuffer : public Buffer {};
class CudaBuffer : public Buffer {};

/// ----------------------------------------------------------------------
/// Payloads

/// gdf_column {
///   data       : buffer
///   valid      : buffer
///   size       : integer+
///   dtype      : enum
///   null_count : integer+
///   dtype_info : {
///     time_unit : enum
///     category  : buffer
///   }
///   col_name   : buffer
/// }
class GdfColumn {
  UC_INTERFACE(GdfColumn);
};

/// ----------------------------------------------------------------------
/// Builders

class Builder {
public:
  virtual std::unique_ptr<Buffer>
  Build() const noexcept = 0;

  /// Experimental
  /// template <class T> Buffer & BufferFrom(T &t) scales exceptions
  virtual Builder &
  LoadFrom(const Buffer &) {
    throw std::runtime_error("Not Implemented");
  }

  UC_INTERFACE(Builder);
};

class DTypeInfoBuilder : public Builder {
public:
  /// ----------------------------------------------------------------------
  /// Member serializers
  virtual DTypeInfoBuilder &
  TimeUnit(const std::int_fast32_t timeUnit) noexcept = 0;

  virtual DTypeInfoBuilder &
  Category(const CudaBuffer &cudaBuffer) noexcept = 0;

  /// ----------------------------------------------------------------------
  /// Builders
  static std::unique_ptr<DTypeInfoBuilder>
  Make();
};

class GdfColumnBuilder : public Builder {
public:
  /// ----------------------------------------------------------------------
  /// Member serializers
  virtual GdfColumnBuilder &
  Data(const CudaBuffer &cudaBuffer) noexcept = 0;

  virtual GdfColumnBuilder &
  Valid(const CudaBuffer &cudaBuffer) noexcept = 0;

  virtual GdfColumnBuilder &
  Size(const std::size_t size) noexcept = 0;

  virtual GdfColumnBuilder &
  DType(const std::int_fast32_t dtype) noexcept = 0;

  virtual GdfColumnBuilder &
  NullCount(const std::size_t size) noexcept = 0;

  virtual GdfColumnBuilder &
  ColumnName(const HostBuffer &hostBuffer) noexcept = 0;

  /// ----------------------------------------------------------------------
  /// Builders
  static std::unique_ptr<GdfColumnBuilder>
  Make();
};

class GdfColumnsBuilder {
public:
  static std::unique_ptr<GdfColumnsBuilder>
  Make();
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
