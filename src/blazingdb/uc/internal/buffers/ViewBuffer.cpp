#include "ViewBuffer.hpp"

#include <cassert>
#include <cuda_runtime.h>

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


void PrintRaw(const std::string &name, const void *&data, const std::size_t size) {
  std::uint8_t *host = new std::uint8_t[size];

  CheckCudaErrors(cudaMemcpy(host, data, size, cudaMemcpyDeviceToHost));
  
  std::stringstream ss;

  ss << ">>> [" <<  name << "]";
  for (std::size_t i = 0; i < size; i++) {
    ss << ' ' <<  static_cast<std::uint32_t>(host[i]);
  }
  ss << std::endl;
  std::cout << ss.str();
  delete[] host;
}


std::unique_ptr<const Record::Serialized>
ViewBuffer::SerializedRecord() const noexcept {
  cudaIpcMemHandle_t ipc_memhandle;
  if (this->data_ == nullptr) {
    std::cout << "***NULL - sender-ipc-handler***\n";
    std::basic_string<uint8_t> bytes;
    return std::make_unique<IpcViewSerialized>(bytes);
  }
  std::cout << "SerializedRecord: pointer address : " << this->data_ << std::endl;

  CheckCudaErrors(cudaIpcGetMemHandle(&ipc_memhandle, (void *) this->data_));
  // PrintRaw("SerializedRecord data:", this->data_, 128);

  std::basic_string<uint8_t> bytes;
  bytes.resize(104);
  memcpy((void*)bytes.data(), (uint8_t*)(&ipc_memhandle), sizeof(cudaIpcMemHandle_t));

  // std::cout << "***sender-ipc-handler***\n";
  // for (auto c : bytes)
  //   std::cout << (int) c << ", ";
  // std::cout << std::endl;

  return std::make_unique<IpcViewSerialized>(bytes);
}


static void* CudaIpcMemHandlerFrom (const std::basic_string<uint8_t>& handler) {
  void * response = nullptr;
  std::cout << "handler-content: " <<  handler.size() <<  std::endl;
  // if (handler.size() == sizeof(cudaIpcMemHandle_t)) {
    cudaIpcMemHandle_t ipc_memhandle;
    memcpy((int8_t*)&ipc_memhandle, handler.data(), sizeof(ipc_memhandle));
    CheckCudaErrors(cudaIpcOpenMemHandle((void **)&response, ipc_memhandle, cudaIpcMemLazyEnablePeerAccess));
  // }
  return response;
}

std::unique_ptr<Transport>
ViewBuffer::Link(const std::uint8_t *recordData) {
  std::basic_string<uint8_t> bytes{recordData, sizeof(cudaIpcMemHandle_t)};
  std::cout << "***link-ipc-handler***\n";
  for (auto c : bytes)
    std::cout << (int) c << ", ";
  std::cout << std::endl;

  this->data_ = CudaIpcMemHandlerFrom(bytes);
  // PrintRaw("Link data:", this->data_, 128);

  return std::make_unique<ViewTransport>();
} 

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
 
