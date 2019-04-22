#include "api-common-test.hpp"

namespace {
class StubTrader : public Trader {
public:
  inline StubTrader(std::promise<const Record::Serialized *> &promise,
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
}  // namespace

static std::thread
Run(std::promise<const Record::Serialized *> &ownPromise,
    std::future<const Record::Serialized *> & peerFuture,
    const std::string &                       name,
    const std::uint64_t                       seed,
    const std::ptrdiff_t                      offset) {
  auto &trader  = *new StubTrader{ownPromise, peerFuture};
  auto &context = *Context::Copy(trader).release();
  auto  data    = CreateData(length, seed, offset);
  return std::thread{Client, std::ref(name), std::ref(context), data};
}

TEST(ApiTest, ThreadsWithCopy) {
  cuInit(0);
  std::promise<const Record::Serialized *> ownPromise;
  std::promise<const Record::Serialized *> peerPromise;

  auto ownFuture  = ownPromise.get_future();
  auto peerFuture = peerPromise.get_future();

  std::thread ownThread{
      ::Run(ownPromise, peerFuture, "own", ownSeed, ownOffset)};
  std::thread peerThread{
      ::Run(peerPromise, ownFuture, "peer", peerSeed, peerOffset)};

  ownThread.join();
  peerThread.join();
  cudaDeviceReset();
}
