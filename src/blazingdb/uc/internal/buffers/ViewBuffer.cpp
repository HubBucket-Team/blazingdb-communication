#include "ViewBuffer.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include <blazingdb/result/Result.h>
#include "records/RemotableRecord.hpp"
#include "transports/ZCopyTransport.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

ViewBuffer::ViewBuffer(const void* &data)
    : data_{data}
{
}

ViewBuffer::~ViewBuffer() {

}


#include <iostream>


#define CheckCudaErrors( call )                                      \
{                                                                    \
  cudaError_t cudaStatus = call;                                     \
  if (cudaSuccess != cudaStatus)                                     \
  {                                                                  \
    std::cerr << "ERROR: CUDA Runtime call " << #call                \
              << " in line " << __LINE__                             \
              << " of file " << __FILE__                             \
              << " failed with " << cudaGetErrorString(cudaStatus)   \
              << " (" << cudaStatus << ").\n";                       \
    /* Call cudaGetLastError to try to clear error if the cuda context is not corrupted */ \
    cudaGetLastError();                                              \
    throw std::runtime_error("In " + std::string(#call) + " function: CUDA Runtime call error " + cudaGetErrorName(cudaStatus));\
  }                                                                  \
}

std::unique_ptr<const Record::Serialized>
ViewBuffer::SerializedRecord() const noexcept {
  std::basic_string<uint8_t> bytes(sizeof(cudaIpcMemHandle_t), '\0');
  if (this->data_ != nullptr) {
    cudaIpcMemHandle_t ipc_memhandle;
    CheckCudaErrors(cudaIpcGetMemHandle(&ipc_memhandle, (void *) this->data_));
    memcpy((void*)bytes.data(), (uint8_t*)(&ipc_memhandle), sizeof(cudaIpcMemHandle_t));
  }

  std::cout << "send-serialized: ";
  for(auto c : bytes) {
    std::cout << (int)c << ",";
  }
  std::cout << "\n";
  return std::make_unique<IpcViewSerialized>(bytes);
}

static void* CudaIpcMemHandlerFrom (const std::basic_string<uint8_t>& handler) {
  std::cout << "buffer-received: ";
  for(auto c : handler) {
    std::cout << (int)c << ",";
  }
  std::cout << "\n";
  void * response = nullptr;
  if (handler.size() == sizeof(cudaIpcMemHandle_t)) {
    if (handler == std::basic_string<uint8_t>(sizeof(cudaIpcMemHandle_t), '\0')) {
      std::cerr << "buffer descriptor are all nulls\n";
      return response;
    }
    cudaIpcMemHandle_t ipc_memhandle;
    memcpy((uint8_t*)&ipc_memhandle, handler.data(), sizeof(ipc_memhandle));
    CheckCudaErrors(cudaIpcOpenMemHandle((void **)&response, ipc_memhandle, cudaIpcMemLazyEnablePeerAccess));
  } else {
    std::cerr << "error handler.size() != 64 at CudaIpcMemHandlerFrom\n";
  }
  return response;
}

std::unique_ptr<Transport>
ViewBuffer::Link(const std::uint8_t *recordData) {
  std::basic_string<uint8_t> bytes{recordData, sizeof(cudaIpcMemHandle_t)};
  this->data_ = CudaIpcMemHandlerFrom(bytes);
  return std::make_unique<ViewTransport>();
} 

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
 
