#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_

#include <memory>
#include <vector>

#include <blazingdb/uc/util/macros.hpp>

#include <blazingdb/uc/Agent.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

/// ----------------------------------------------------------------------
/// Buffers

class Buffer {
public:
  virtual const void *
  Data() const noexcept = 0;

  virtual std::size_t
  Size() const noexcept = 0;

  UC_INTERFACE(Buffer);
};

class NullableBuffer : public Buffer {
public:
  virtual bool
  IsNull() const noexcept = 0;

  UC_INTERFACE(NullableBuffer);
};

class HostBuffer : public NullableBuffer {
  UC_INTERFACE(HostBuffer);
};

class CudaBuffer : public NullableBuffer {
public:
  static std::unique_ptr<CudaBuffer>
  Make(const void *const data, const std::size_t size);

  UC_INTERFACE(CudaBuffer);
};

class UCBuffer : public HostBuffer {
  UC_INTERFACE(UCBuffer);
};

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
  MakeWithHostAllocation(blazingdb::uc::Agent &);

  UC_INTERFACE(GdfColumnBuilder);
};

/// ----------------------------------------------------------------------
/// Collectors

class Collector {
public:
  // collection

  virtual std::unique_ptr<Buffer>
  Collect() const noexcept = 0;

  virtual Collector &
  Add(const Payload &payload) noexcept = 0;

  // collected

  virtual std::size_t
  Length() const noexcept = 0;

  virtual const Payload &
  Get(std::size_t index) const = 0;

  const Payload &operator[](const std::size_t index) const {
    return Get(index);
  }

  UC_INTERFACE(Collector);
};

class GdfColumnCollector : public Collector {
public:
  static std::unique_ptr<GdfColumnCollector>
  MakeInHost();

  UC_INTERFACE(GdfColumnCollector);
};

/// ----------------------------------------------------------------------
/// Dispatchers

class Dispatcher {
public:
  virtual std::unique_ptr<Collector>
  Dispatch() const = 0;

  UC_INTERFACE(Dispatcher);
};

class GdfColumnDispatcher : public Dispatcher {
public:
  static std::unique_ptr<GdfColumnDispatcher>
  MakeInHost(const Buffer &buffer);

  UC_INTERFACE(GdfColumnDispatcher);
};

/// ----------------------------------------------------------------------
/// Specializeds

class Specialized {
public:
  virtual std::unique_ptr<Payload>
  Apply() const = 0;

  UC_INTERFACE(Specialized);
};

class GdfColumnSpecialized : public Specialized {
public:
  static std::unique_ptr<Specialized>
  Make(const Buffer &buffer);

  UC_INTERFACE(GdfColumnSpecialized);
};

/// ----------------------------------------------------------------------
/// Utils

template <class Column>
std::unique_ptr<Collector>
CollectorFrom(const std::vector<Column> &columns, blazingdb::uc::Agent &agent) {
  std::unique_ptr<GdfColumnCollector> collector =
      GdfColumnCollector::MakeInHost();

  for (const auto &column : columns) {
    auto *column_ptr = column.get_gdf_column();

    std::unique_ptr<GdfColumnBuilder> builder =
        GdfColumnBuilder::MakeWithHostAllocation(agent);

    const std::unique_ptr<const CudaBuffer> dataBuffer =
        CudaBuffer::Make(column_ptr->data, 0);
    const std::unique_ptr<const CudaBuffer> validBuffer =
        CudaBuffer::Make(column_ptr->valid, 0);

    std::unique_ptr<Payload> columnPayload = builder->Data(*dataBuffer)
                                                 .Valid(*validBuffer)
                                                 .Size(column_ptr->size)
                                                 .Build();

    collector->Add(*columnPayload);
  }

  return collector;
}


std::string
StringFrom(const Buffer &buffer);

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
