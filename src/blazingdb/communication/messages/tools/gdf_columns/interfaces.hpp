#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INTERFACES_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INTERFACES_HPP_

#include <memory>

#include <blazingdb/uc/Agent.hpp>
#include <blazingdb/uc/util/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

/// ----------------------------------------------------------------------
/// Buffers

class Buffer {
  UC_INTERFACE(Buffer);

public:
  virtual const void *
  Data() const noexcept = 0;

  virtual std::size_t
  Size() const noexcept = 0;

  /// ----------------------------------------------------------------------
  /// Casters
  template <class T, class U>
  static UC_INLINE std::enable_if_t<!std::is_array<std::decay_t<T>>::value &&
                                        !std::is_array<std::decay_t<U>>::value,
                                    T>
                   StaticCast(const U &u) noexcept {
    return static_cast<const T>(u, static_cast<const T *>(u.get()));
  }
};

class NullableBuffer : public Buffer {
  UC_INTERFACE(NullableBuffer);

public:
  virtual bool
  IsNull() const noexcept = 0;
};

class HostBuffer : public NullableBuffer {
  UC_INTERFACE(HostBuffer);

public:
  static std::unique_ptr<HostBuffer>
  Make(const void *const data, const std::size_t size);
};

class CudaBuffer : public NullableBuffer {
  UC_INTERFACE(CudaBuffer);

public:
  static std::unique_ptr<CudaBuffer>
  Make(const void *const data, const std::size_t size);
};

class UCBuffer : public HostBuffer {
  UC_INTERFACE(UCBuffer);

public:
  static UC_INLINE const UCBuffer &
                         From(const Buffer &buffer) noexcept {
    return static_cast<const UCBuffer &>(buffer);
  }
};

/// ----------------------------------------------------------------------
/// Payloads

/// gdf_column {
///   data       : buffer
///   valid      : buffer
///   size       : positive integer
///   dtype      : enum
///   null_count : positive integer
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

class CategoryPayload : public Payload {
public:
  virtual const UCBuffer &
  Strs() const noexcept = 0;

  virtual const UCBuffer &
  Mem() const noexcept = 0;

  virtual const UCBuffer &
  Map() const noexcept = 0;

  virtual std::size_t
  Count() const noexcept = 0;

  virtual std::size_t
  Keys() const noexcept = 0;

  virtual std::size_t
  Size() const noexcept = 0;

  virtual std::size_t
  BaseAddress() const noexcept = 0;

  UC_INTERFACE(CategoryPayload);
};

class DTypeInfoPayload : public Payload {
public:
  virtual std::int_fast32_t
  TimeUnit() const noexcept = 0;

  virtual CategoryPayload &
  Category() const noexcept = 0;

  UC_INTERFACE(DTypeInfoPayload);
};

class GdfColumnPayload : public Payload {
  UC_INTERFACE(GdfColumnPayload);

public:
  virtual const UCBuffer &
  Data() const noexcept = 0;

  virtual const UCBuffer &
  Valid() const noexcept = 0;

  virtual std::size_t
  Size() const noexcept = 0;

  virtual std::int_fast32_t
  DType() const noexcept = 0;

  virtual std::size_t
  NullCount() const noexcept = 0;

  virtual const DTypeInfoPayload &
  DTypeInfo() const noexcept = 0;

  virtual const UCBuffer &
  ColumnName() const noexcept = 0;
};

/// ----------------------------------------------------------------------
/// Values

class Value {
  UC_INTERFACE(Value);
};

class CategoryValue : public Value {
  UC_INTERFACE(CategoryValue);

public:
  virtual std::size_t
  base_address() const noexcept = 0;

  virtual const void *
  mem() const noexcept = 0;

  virtual const void *
  strs() const noexcept = 0;

  virtual std::size_t
  size() const noexcept = 0;

  virtual std::size_t
  count() const noexcept = 0;

  virtual std::size_t
  keys() const noexcept = 0;

  virtual const void *
  map() const noexcept = 0;
};

class DTypeInfoValue : public Value {
  UC_INTERFACE(DTypeInfoValue);

public:
  virtual std::int_fast32_t
  time_unit() const noexcept = 0;

  virtual const CategoryValue &
  category() const noexcept = 0;
};

class GdfColumnValue : public Value {
  UC_INTERFACE(GdfColumnValue);

public:
  virtual const void *
  data() const noexcept = 0;

  virtual const void *
  valid() const noexcept = 0;

  virtual std::size_t
  size() const noexcept = 0;

  virtual std::int_fast32_t
  dtype() const noexcept = 0;

  virtual std::size_t
  null_count() const noexcept = 0;

  virtual const DTypeInfoValue &
  dtype_info() const noexcept = 0;

  virtual const char *
  column_name() const noexcept = 0;

  static std::unique_ptr<GdfColumnValue>
  Make(const GdfColumnPayload &gdfColumnPayload, blazingdb::uc::Agent &agent);
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

class CategoryBuilder : public Builder {
public:
  /// ----------------------------------------------------------------------
  /// Member serializers
  virtual CategoryBuilder &
  Strs(const CudaBuffer &cudaBuffer) noexcept = 0;

  virtual CategoryBuilder &
  Mem(const CudaBuffer &cudaBuffer) noexcept = 0;

  virtual CategoryBuilder &
  Map(const CudaBuffer &cudaBuffer) noexcept = 0;

  virtual CategoryBuilder &
  Count(const std::size_t count) noexcept = 0;

  virtual CategoryBuilder &
  Keys(const std::size_t keys) noexcept = 0;

  virtual CategoryBuilder &
  Size(const std::size_t size) noexcept = 0;

  virtual CategoryBuilder &
  BaseAddress(const std::size_t baseAddress) noexcept = 0;

  /// ----------------------------------------------------------------------
  /// Builders
  static std::unique_ptr<CategoryBuilder>
  MakeInHost(blazingdb::uc::Agent &agent);

  UC_INTERFACE(CategoryBuilder);
};

class DTypeInfoBuilder : public Builder {
public:
  /// ----------------------------------------------------------------------
  /// Member serializers
  virtual DTypeInfoBuilder &
  TimeUnit(const std::int_fast32_t timeUnit) noexcept = 0;

  virtual DTypeInfoBuilder &
  Category(const Payload &categoryPayload) noexcept = 0;

  /// ----------------------------------------------------------------------
  /// Builders
  static std::unique_ptr<DTypeInfoBuilder>
  MakeInHost(blazingdb::uc::Agent &agent);

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
  NullCount(const std::size_t nullCount) noexcept = 0;

  virtual GdfColumnBuilder &
  DTypeInfo(const Payload &dtypeInfoPayload) noexcept = 0;

  virtual GdfColumnBuilder &
  ColumnName(const HostBuffer &hostBuffer) noexcept = 0;

  /// ----------------------------------------------------------------------
  /// Builders
  static std::unique_ptr<GdfColumnBuilder>
  MakeInHost(blazingdb::uc::Agent &agent);

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

  class iterator {
    const Collector &collector_;
    size_t           index_;

  public:
    iterator(const Collector &c) : collector_(c), index_(0) {}
    iterator(const Collector &c, size_t size) : collector_(c), index_(size) {}

    void
    operator++() {
      index_++;
    }

    void
    operator--() {
      index_--;
    }

    bool
    operator!=(const iterator &other) {
      return index_ != other.index_;
    }

    const Payload &operator*() { return collector_.Get(index_); }
  };

  iterator
  begin() {
    iterator it(*this);
    return it;
  }

  iterator
  end() {
    iterator it(*this, this->Length());
    return it;
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
  MakeInHost(const Buffer &buffer);

  UC_INTERFACE(GdfColumnSpecialized);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
