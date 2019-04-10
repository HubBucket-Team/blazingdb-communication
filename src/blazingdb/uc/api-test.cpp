#include "API.h"

#include <cstring>

#include <cuda_runtime_api.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

static inline void *
CreateData(const std::size_t    size,
           std::uint64_t        seed,
           const std::ptrdiff_t offset) {
  static const std::uint64_t pn   = 1337;
  std::uint8_t *             data = new std::uint8_t[size];
  assert(nullptr != data);
  auto                it  = reinterpret_cast<std::uint64_t *>(data);
  const std::uint8_t *end = data + size;

  while (reinterpret_cast<std::uint8_t *>(it + 1) <= end) {
    *it  = seed;
    seed = (seed << 1) | (__builtin_parityl(seed & pn) & 1);
    ++it;
  }

  std::memcpy(
      it,
      &seed,
      static_cast<std::size_t>(end - reinterpret_cast<std::uint8_t *>(it)));

  void *      buffer;
  cudaError_t cudaStatus = cudaMalloc(&buffer, size);
  assert(cudaSuccess == cudaStatus);

  cudaStatus = cudaMemcpy(buffer, data + offset, size, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStatus);

  cudaStatus = cudaDeviceSynchronize();
  assert(cudaSuccess == cudaStatus);

  delete[] data;

  return buffer;
}

class MockManager : public blazingdb::uc::Manager {
public:
  MOCK_CONST_METHOD1(Get, void *(const std::size_t));
};

static constexpr std::size_t    length           = 16;
static constexpr std::uint16_t  port             = 8000;
static constexpr char           clienHostname[]  = "";
static constexpr char           serverHostname[] = "127.0.0.1";
static constexpr std::uint64_t  ownSeed          = 0x1111111111111111lu;
static constexpr std::uint64_t  peerSeed         = 0x2222222222222222lu;
static constexpr std::ptrdiff_t ownOffset        = 1;
static constexpr std::ptrdiff_t peerOffset       = 3;

TEST(API, Flow) {
  MockManager manager;

  auto context = blazingdb::uc::Context::CudaIPC(manager);

  auto own  = context->OwnAgent();
  auto peer = context->PeerAgent();

  void *ownData  = CreateData(length, ownSeed, ownOffset);
  void *peerData = CreateData(length, peerSeed, peerOffset);

  auto ownBuffer = own->Register(ownData, length);
  auto peerBuffer = own->Register(peerData, length);
}
