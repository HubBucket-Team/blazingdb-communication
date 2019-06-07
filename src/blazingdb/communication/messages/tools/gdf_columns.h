#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_

#include <memory>

#include <blazingdb/uc/util/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class Buffer {
public:
  virtual const void *
  Data() const noexcept = 0;

  virtual std::size_t
  Size() const noexcept = 0;

  UC_INTERFACE(Buffer);
};

class HostBuffer : public Buffer {};

class CudaBuffer : public Buffer {
public:
  static std::unique_ptr<CudaBuffer>
  Make(const void *const data, const std::size_t size);

  UC_INTERFACE(CudaBuffer);
};

class UCBuffer : public Buffer {};

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

class Payload {
public:
  virtual const Buffer &
  Deliver() const noexcept = 0;

  UC_INTERFACE(Payload);
};

class DTypeInfoPayload : public Payload {
  UC_INTERFACE(DTypeInfoPayload);
};

class GdfColumnPayload : public Payload {
public:
  virtual const UCBuffer &
  Data() const noexcept = 0;

  virtual UCBuffer &
  Valid() const noexcept = 0;

  virtual std::size_t
  Size() const noexcept = 0;

  virtual std::int_fast32_t
  DType() noexcept = 0;

  virtual std::size_t
  NullCount() const noexcept = 0;

  virtual DTypeInfoPayload &
  DTypeInfo() const noexcept = 0;

  virtual std::string
  ColumnName() const noexcept = 0;

  UC_INTERFACE(GdfColumnPayload);
};

/// ----------------------------------------------------------------------
/// Builders

class Builder {
public:
  virtual std::unique_ptr<Payload>
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

  UC_INTERFACE(DTypeInfoBuilder);
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
  DTypeInfo(const DTypeInfoPayload &dtypeInfoPayload) noexcept = 0;

  virtual GdfColumnBuilder &
  ColumnName(const HostBuffer &hostBuffer) noexcept = 0;

  /// ----------------------------------------------------------------------
  /// Builders
  static std::unique_ptr<GdfColumnBuilder>
  MakeWithHostAllocation();

  UC_INTERFACE(GdfColumnBuilder);
};

class GdfColumnsBuilder {
public:
  static std::unique_ptr<GdfColumnsBuilder>
  Make();

  UC_INTERFACE(GdfColumnsBuilder);
};

/// ----------------------------------------------------------------------
/// Collector

class Collector {
public:
  virtual std::unique_ptr<Payload>
  Apply() const = 0;

  UC_INTERFACE(Collector);
};

class GdfColumnCollector : public Collector {
public:
  static std::unique_ptr<Collector>
  Make(const Buffer &buffer);

  UC_INTERFACE(GdfColumnCollector);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
