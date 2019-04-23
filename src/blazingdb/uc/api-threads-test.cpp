#include "api-common-test.hpp"

namespace {
class StubTrader : public Trader {
public:
  inline StubTrader(std::promise<const Record::Serialized *> &ownPromise,
                    std::future<const Record::Serialized *> & peerFuture,
                    std::promise<void> &                      resolvePromise,
                    std::future<void> &                       resolvedFuture)
      : ownPromise_{ownPromise},
        peerFuture_{peerFuture},
        resolvePromise_{resolvePromise},
        resolvedFuture_{resolvedFuture} {}

  void
  OnRecording(Record *record) const noexcept {
    auto ownSerialized = record->GetOwn();
    ownPromise_.set_value(ownSerialized.get());

    peerFuture_.wait();
    auto peerSerialized = peerFuture_.get();

    record->SetPeer(peerSerialized->Data());

    resolvePromise_.set_value();
    resolvedFuture_.wait();
  }

private:
  std::promise<const Record::Serialized *> &ownPromise_;
  std::future<const Record::Serialized *> & peerFuture_;
  std::promise<void> &                      resolvePromise_;
  std::future<void> &                       resolvedFuture_;
};
}  // namespace

static std::thread
Run(std::promise<const Record::Serialized *> &ownPromise,
    std::future<const Record::Serialized *> &&peerFuture,
    std::promise<void> &                      resolvePromise,
    std::future<void> &&                      resolvedFuture,
    const std::string &                       name,
    const std::uint64_t                       seed,
    const std::ptrdiff_t                      offset) {
  auto &trader =
      *new StubTrader{ownPromise, peerFuture, resolvePromise, resolvedFuture};
  auto &context = *Context::Copy(trader).release();
  auto  data    = CreateData(length, seed, offset);
  return std::thread{Client, std::ref(name), std::ref(context), data};
}

TEST(ApiOnThreadsTest, WithCopy) {
  cuInit(0);
  std::promise<const Record::Serialized *> ownPromise, peerPromise;
  std::promise<void>                       ownResolved, peerResolved;

  std::thread ownThread{::Run(ownPromise,
                              peerPromise.get_future(),
                              ownResolved,
                              peerResolved.get_future(),
                              "own",
                              ownSeed,
                              ownOffset)};
  std::thread peerThread{::Run(peerPromise,
                               ownPromise.get_future(),
                               peerResolved,
                               ownResolved.get_future(),
                               "peer",
                               peerSeed,
                               peerOffset)};

  ownThread.join();
  peerThread.join();
}
