#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_DTYPEINFOINHOSTBASE_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_DTYPEINFOINHOSTBASE_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

#include "CategoryPayloadInHostBase.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT DTypeInfoPayloadInHostBase : public DTypeInfoPayload {
  UC_CONCRETE(DTypeInfoPayloadInHostBase);

public:
  explicit DTypeInfoPayloadInHostBase(const Buffer& buffer);

  std::int_fast32_t
  TimeUnit() const noexcept final;

  const PayloadableBuffer&
  Category() const noexcept final;

  const Buffer&
  Deliver() const noexcept final;

private:
  const Buffer& buffer_;

  const std::int_fast32_t*           timeUnit_;
  std::unique_ptr<PayloadableBuffer> categoryBuffer_;
};

class UC_NOEXPORT InHostDTypeInfoPayloadBuffer : public PayloadableBuffer {
  UC_CONCRETE(InHostDTypeInfoPayloadBuffer);

public:
  explicit InHostDTypeInfoPayloadBuffer(const void* const data,
                                       const std::size_t size)
      : data_{data}, size_{size} {}

  const void*
  Data() const noexcept final {
    return data_;
  }

  std::size_t
  Size() const noexcept final {
    return size_;
  }

  std::unique_ptr<Payload>
  ToPayload() const noexcept final {
    return std::make_unique<CategoryPayloadInHostBase>(*this);
  }

private:
  const void* const data_;
  const std::size_t size_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
