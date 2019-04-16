#include "API.h"

#include <cstring>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

static UC_INLINE void
Print(const std::string &name, const void *data, const std::size_t size) {
  std::uint8_t *host = new std::uint8_t[size];

  cudaError_t cudaStatus = cudaMemcpy(host, data, size, cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStatus);

  std::stringstream ss;

  ss << ">>> [" << std::setw(9) << name << "]";
  for (std::size_t i = 0; i < size; i++) {
    ss << ' ' << std::setfill('0') << std::setw(3)
       << static_cast<std::uint32_t>(host[i]);
  }
  ss << std::endl;
  std::cout << ss.str();

  delete[] host;
}

static constexpr std::size_t    length     = 16;
static constexpr std::uint64_t  ownSeed    = 0x1111111111111111lu;
static constexpr std::uint64_t  peerSeed   = 0x2222222222222222lu;
static constexpr std::uint64_t  twinSeed   = 0x3333333333333333lu;
static constexpr std::ptrdiff_t ownOffset  = 1;
static constexpr std::ptrdiff_t peerOffset = 3;
static constexpr std::ptrdiff_t twinOffset = 4;

using namespace blazingdb::uc;

void *
CreateData(const std::size_t    size,
           std::uint64_t        seed,
           const std::ptrdiff_t offset);

void
Client(const std::string &name, const Trader &trader, const void *const data);
