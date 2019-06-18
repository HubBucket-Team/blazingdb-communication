#include "UCGdfColumnValue.hpp"

#include <cassert>
#include <cmath>

#include <cuda_runtime_api.h>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

namespace {

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

}  // namespace

UCGdfColumnValue::UCGdfColumnValue(const GdfColumnPayload& gdfColumnPayload,
                                   blazingdb::uc::Agent&   agent)
    : gdfColumnPayload_{gdfColumnPayload},
      data_{LinkThrough(agent,
                        gdfColumnPayload.Data(),
                        gdfColumnPayload.Size() * 8,  // TODO: calculate
                        &dataUcBuffer_)},
      valid_{LinkThrough(
          agent,
          gdfColumnPayload.Valid(),
          std::ceil(gdfColumnPayload.NullCount() / 8),  // TODO: calculate
          &validUcBuffer_)} {}

const void*
UCGdfColumnValue::data() const noexcept {
  return data_;
}

const void*
UCGdfColumnValue::valid() const noexcept {
  return valid_;
}

std::size_t
UCGdfColumnValue::size() const noexcept {
  return gdfColumnPayload_.Size();
}

std::int_fast32_t
UCGdfColumnValue::dtype() const noexcept {
  return gdfColumnPayload_.DType();
}

std::size_t
UCGdfColumnValue::null_count() const noexcept {
  return gdfColumnPayload_.NullCount();
}

const DTypeInfoValue&
UCGdfColumnValue::dtype_info() const noexcept {
  static DTypeInfoValue* dtypeInfoValue_;
  UC_ABORT("Not support");
  return *dtypeInfoValue_;
}

const char*
UCGdfColumnValue::column_name() const noexcept {
  UC_ABORT("Not support");
  return nullptr;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
