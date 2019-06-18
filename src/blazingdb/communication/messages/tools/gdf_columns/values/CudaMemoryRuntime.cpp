#include "CudaMemoryRuntime.hpp"

#include <cuda_runtime_api.h>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

CudaMemoryRuntime::CudaMemoryRuntime() = default;

void*
CudaMemoryRuntime::Allocate(const std::size_t size) {
  void*       data;
  cudaError_t cudaStatus = cudaMalloc(&data, size);
  if (cudaSuccess != cudaStatus) {
    throw std::runtime_error("Allocation Error");
  }
  return data;
}

void
CudaMemoryRuntime::Synchronize() {
  cudaError_t cudaStatus = cudaDeviceSynchronize();
  if (cudaSuccess != cudaStatus) {
    throw std::runtime_error("Allocation Error");
  }
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
