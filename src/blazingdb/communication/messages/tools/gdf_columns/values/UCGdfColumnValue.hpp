#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_UCGDFCOLUMNVALUE_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_VALUES_UCGDFCOLUMNVALUE_HPP_

#include "../interfaces.hpp"

#include <cmath>

#include <blazingdb/uc/internal/macros.hpp>

#include <cuda_runtime_api.h>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

static UC_INLINE const void*
Malloc(const std::size_t size) {
  void* buffer;

  cudaError_t cudaStatus = cudaMalloc(&buffer, size);
  assert(cudaSuccess == cudaStatus);

  cudaStatus = cudaDeviceSynchronize();
  assert(cudaSuccess == cudaStatus);

  return buffer;
}

static UC_INLINE const void*
LinkThrough(blazingdb::uc::Agent&                   agent,
            const Buffer&                           buffer,
            const std::size_t                       size,
            std::unique_ptr<blazingdb::uc::Buffer>* outputUcBuffer) {
  if (nullptr == buffer.Data()) { return nullptr; }

  const void* data = Malloc(size);

  *outputUcBuffer = agent.Register(data, size);

  std::unique_ptr<blazingdb::uc::Transport> transport =
      (*outputUcBuffer)->Link(static_cast<const std::uint8_t*>(buffer.Data()));

  transport->Get().wait();

  return data;
}

class UCGdfColumnValue : public GdfColumnValue {
  UC_CONCRETE(UCGdfColumnValue);

public:
  explicit UCGdfColumnValue(const GdfColumnPayload& gdfColumnPayload,
                            blazingdb::uc::Agent&   agent)
      : gdfColumnPayload_{gdfColumnPayload},
        data_{LinkThrough(agent,
                          gdfColumnPayload.Data(),
                          gdfColumnPayload.Size() * 8,  // TODO: calculate
                          &dataBuffer_)},
        valid_{LinkThrough(
            agent,
            gdfColumnPayload.Valid(),
            std::ceil(gdfColumnPayload.NullCount() / 8),  // TODO: calculate
            &dataBuffer_)} {}

  const void*
  data() const noexcept final {
    return data_;
  }

  const void*
  valid() const noexcept final {
    return valid_;
  }

  std::size_t
  size() const noexcept final {
    return gdfColumnPayload_.Size();
  }

  std::int_fast32_t
  dtype() const noexcept final {
    return gdfColumnPayload_.DType();
  }

  std::size_t
  null_count() const noexcept final {
    return gdfColumnPayload_.NullCount();
  }

  const DTypeInfoValue&
  dtype_info() const noexcept final {
    static DTypeInfoValue* dtypeInfoValue_;
    UC_ABORT("Not support");
    return *dtypeInfoValue_;
  }

  const char*
  column_name() const noexcept final {
    UC_ABORT("Not support");
    return nullptr;
  }

private:
  const GdfColumnPayload& gdfColumnPayload_;

  const void* const data_;
  const void* const valid_;

  std::unique_ptr<blazingdb::uc::Buffer> dataBuffer_;
  std::unique_ptr<blazingdb::uc::Buffer> validBuffer_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
