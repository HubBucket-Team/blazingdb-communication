#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_H_

#include <memory>
#include <vector>
#include <iostream>

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
  NullCount(const std::size_t nullCount) noexcept = 0;

  virtual GdfColumnBuilder &
  DTypeInfo(const DTypeInfoPayload &dtypeInfoPayload) noexcept = 0;

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

/// ----------------------------------------------------------------------
/// Utils

template <class GdfColumnInfo, class gdf_column>
std::string
DeliverFrom(const std::vector<gdf_column> &gdfColumns,
            blazingdb::uc::Agent &        agent) {
  std::unique_ptr<GdfColumnCollector> collector =
      GdfColumnCollector::MakeInHost();

  std::vector<std::unique_ptr<Payload>> payloads;
  payloads.reserve(gdfColumns.size());

  for (const auto &gdfColumn : gdfColumns) {
    // auto *column_ptr = gdfColumn.get_gdf_column();

    std::unique_ptr<GdfColumnBuilder> builder =
        GdfColumnBuilder::MakeInHost(agent);

    // TODO: Add other members y compute correct buffer size
    const std::unique_ptr<const CudaBuffer> dataBuffer =
        CudaBuffer::Make(gdfColumn.data, GdfColumnInfo::DataSize(gdfColumn));
    const std::unique_ptr<const CudaBuffer> validBuffer =
        CudaBuffer::Make(gdfColumn.valid, GdfColumnInfo::ValidSize(gdfColumn));
    const std::size_t size = gdfColumn.size;

    payloads.emplace_back(builder->Data(*dataBuffer)
                                                 .Valid(*validBuffer)
                                                 .Size(size)
                                                 .Build());
    // TODO: support different buffer sizes (of payloads) in GdfColumnCollector

    collector->Add(*payloads.back());
  }

  std::unique_ptr<Buffer> resultBuffer = collector->Collect();

  //TODO usando el dispatcher chequear el primer elemento

  return std::string{
      static_cast<const std::string::value_type *const>(resultBuffer->Data()),
      resultBuffer->Size()};
}

std::string
StringFrom(const Buffer &buffer);

namespace {
class StringViewBuffer : public Buffer {
public:
  explicit StringViewBuffer(const std::string &content) : content_{content} {}

  const void *
  Data() const noexcept final {
    return content_.data();
  }

  std::size_t
  Size() const noexcept final {
    return content_.length();
  }

private:
  const std::string &content_;
};
}  // namespace

template <class gdf_column>
std::vector<gdf_column>
CollectFrom(const std::string &content, blazingdb::uc::Agent &agent) {
  const Buffer &buffer = StringViewBuffer{content};

  std::unique_ptr<GdfColumnDispatcher> dispatcher =
      GdfColumnDispatcher::MakeInHost(buffer);

  std::unique_ptr<Collector> collector = dispatcher->Dispatch();

  std::vector<gdf_column> gdfColumns;
  gdfColumns.reserve(collector->Length());

  for (Collector::iterator it = collector->begin(); it != collector->end();
       ++it) {
    GdfColumnPayload &payload = *it;
    gdf_column        col{.data  = payload.Data().Data(),
                   .valid = payload.Valid().Data(),
                   .size  = payload.Size()};
    gdfColumns.push_back(col);
  }

  return gdfColumns;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
