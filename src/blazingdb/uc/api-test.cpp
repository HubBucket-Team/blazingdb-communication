#include "API.h"

#include <cstring>
#include <iomanip>

#include <cuda.h>
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

static void
Print(const std::string &name, const void *data, const std::size_t size) {
  std::uint8_t *host = new std::uint8_t[size];

  cudaError_t cudaStatus = cudaMemcpy(host, data, size, cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStatus);

  std::stringstream ss;

  ss << ">>> [" << std::setw(4) << name << "]";
  for (std::size_t i = 0; i < size; i++) {
    ss << ' ' << std::setfill('0') << std::setw(3)
       << static_cast<std::uint32_t>(host[i]);
  }
  ss << std::endl;
  std::cout << ss.str();

  delete[] host;
}

using namespace blazingdb::uc;

class StubTrader : public Trader {
public:
  inline StubTrader(const int (&pipedes)[2]) : pipedes_{pipedes} {}

  void
  OnRecording(Record *record) const noexcept {
    auto ownSerialized = record->GetOwn();
    write(pipedes_[1], ownSerialized->Data(), ownSerialized->Size());

    auto *data = new std::uint8_t[ownSerialized->Size()];
    read(pipedes_[0], data, ownSerialized->Size());

    record->SetPeer(data);
    delete[] data;
  }

private:
  const int (&pipedes_)[2];
};

class StubThreadTrader : public Trader {
public:
  inline StubThreadTrader(std::promise<const Record::Serialized *> &promise,
                          std::future<const Record::Serialized *> & future)
      : promise_{promise}, future_{future} {}

  void
  OnRecording(Record *record) const noexcept {
    auto ownSerialized = record->GetOwn();
    promise_.set_value(ownSerialized.get());

    future_.wait();
    auto peerSerialized = future_.get();

    record->SetPeer(peerSerialized->Data());

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

private:
  std::promise<const Record::Serialized *> &promise_;
  std::future<const Record::Serialized *> & future_;
};

static constexpr std::size_t    length     = 16;
static constexpr std::uint64_t  ownSeed    = 0x1111111111111111lu;
static constexpr std::uint64_t  peerSeed   = 0x2222222222222222lu;
static constexpr std::uint64_t  twinSeed   = 0x3333333333333333lu;
static constexpr std::ptrdiff_t ownOffset  = 1;
static constexpr std::ptrdiff_t peerOffset = 3;
static constexpr std::ptrdiff_t twinOffset = 4;

static void
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

static void
Exec(const std::string &  name,
     const std::uint64_t  seed,
     const std::ptrdiff_t offset,
     const int (&own_pipedes)[2],
     const int (&peer_pipedes)[2]) {
  cuInit(0);
  void *     data       = CreateData(length, seed, offset);
  int        pipedes[2] = {own_pipedes[0], peer_pipedes[1]};
  StubTrader trader{pipedes};
  Client(name, trader, data);
}

// TEST(ApiTest, Processes) {
// int own_pipedes[2];
// int peer_pipedes[2];

// pipe(own_pipedes);
// pipe(peer_pipedes);

// pid_t pid = fork();

// ASSERT_NE(-1, pid);

// if (pid) {
// Exec("own", ownSeed, ownOffset, own_pipedes, peer_pipedes);
// int stat_loc;
// pid = waitpid(pid, &stat_loc, 0);
// ASSERT_EQ(0, stat_loc);
// ASSERT_NE(-1, pid);
//} else {
// Exec("peer", peerSeed, peerOffset, peer_pipedes, own_pipedes);
// std::exit(EXIT_SUCCESS);
//}
//}

TEST(ApiTest, Threads) {
  cuInit(0);
  std::promise<const Record::Serialized *> ownPromise;
  std::promise<const Record::Serialized *> peerPromise;

  auto ownFuture  = ownPromise.get_future();
  auto peerFuture = peerPromise.get_future();

  StubThreadTrader ownTrader{ownPromise, peerFuture};
  StubThreadTrader peerTrader{peerPromise, ownFuture};

  auto ownData  = CreateData(length, ownSeed, ownOffset);
  auto peerData = CreateData(length, peerSeed, peerOffset);

  std::thread ownThread{Client, "own", std::ref(ownTrader), ownData};
  std::thread peerThread{Client, "peer", std::ref(peerTrader), peerData};

  ownThread.join();
  peerThread.join();
}
