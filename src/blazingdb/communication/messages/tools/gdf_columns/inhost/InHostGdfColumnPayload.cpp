#include "InHostGdfColumnPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class ConcreteUCBuffer : public UCBuffer {
public:
  explicit ConcreteUCBuffer(const void* const data, const std::size_t size)
      : data_{data}, size_{size} {}

  const void*
  Data() const noexcept final {
    return data_;
  }

  std::size_t
  Size() const noexcept {
    return size_;
  }

private:
  const void* const data_;
  const std::size_t size_;
};

InHostGdfColumnPayload::InHostGdfColumnPayload(
    std::unique_ptr<blazingdb::uc::Context>&& context,
    const std::string&&                       payload)
    : buffer_{nullptr, 0},
      context_{std::move(context)},
      payload_{std::move(payload)} {};

const UCBuffer&
InHostGdfColumnPayload::Data() const noexcept {
  static ConcreteUCBuffer buffer(&payload_[0], 104);
  return buffer;
}

UCBuffer&
InHostGdfColumnPayload::Valid() const noexcept {
  return *ucBuffer_;
}

std::size_t
InHostGdfColumnPayload::Size() const noexcept {
  return -1;
}

std::int_fast32_t
InHostGdfColumnPayload::DType() noexcept {
  return -1;
}

std::size_t
InHostGdfColumnPayload::NullCount() const noexcept {
  return -1;
}

DTypeInfoPayload&
InHostGdfColumnPayload::DTypeInfo() const noexcept {
  return *dtypeInfoPayload_;
}

std::string
InHostGdfColumnPayload::ColumnName() const noexcept {
  return "";
}

const Buffer&
InHostGdfColumnPayload::Deliver() const noexcept {
  return buffer_;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
