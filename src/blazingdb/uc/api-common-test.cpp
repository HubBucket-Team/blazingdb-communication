#include "api-common-test.hpp"

void
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

void *
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

void
Client(const std::string &name, const Trader &trader, const void *const data) {
  void *twinData = CreateData(length, twinSeed, twinOffset);

  auto context = Context::Copy(trader);

  auto own  = context->OwnAgent();
  auto peer = context->PeerAgent();

  auto ownBuffer  = own->Register(data, length);
  auto peerBuffer = peer->Register(twinData, length);

  Print(name, data, length);
  Print(name + " twin", twinData, length);

  auto transport = ownBuffer->Link(peerBuffer.get());

  // if ("own" == name) {
  transport->Get();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  Print(name, data, length);
  Print(name + " twin", twinData, length);
  //}
}

