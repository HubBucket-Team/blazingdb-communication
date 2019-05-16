#include "ViewBuffer.hpp"

#include <cassert>
#include <cuda_runtime.h>

#include "records/RemotableRecord.hpp"
#include "transports/ZCopyTransport.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

ViewBuffer::ViewBuffer(const void *const data)
    : data_{data}
{

}
ViewBuffer::~ViewBuffer() {

}

std::unique_ptr<const Record::Serialized>
ViewBuffer::SerializedRecord() const noexcept {
  cudaIpcMemHandle_t ipc_memhandle;
  auto cudaError = cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &ipc_memhandle, (void *) this->data_);
  assert(cudaError != cudaSuccess);

  std::basic_string<uint8_t> bytes{};
  bytes.resize(104);
  memcpy((void*)bytes.data(), (int8_t*)(&ipc_memhandle), sizeof(cudaIpcMemHandle_t));
  return std::make_unique<IpcViewSerialized>(bytes);
}

std::unique_ptr<Transport>
ViewBuffer::Link(const std::uint8_t *recordData) {
  cudaIpcMemHandle_t ipc_memhandle;
  memcpy((int8_t*)&ipc_memhandle, recordData, sizeof(cudaIpcMemHandle_t));
  auto cudaError = cudaIpcOpenMemHandle((void **)(&this->data_), ipc_memhandle, cudaIpcMemLazyEnablePeerAccess);
  assert(cudaError != cudaSuccess);
  return std::make_unique<ViewTransport>();
} 

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
 
