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

TEST(ApiTest, Threads) {
  cuInit(0);
  std::promise<const Record::Serialized *> ownPromise;
  std::promise<const Record::Serialized *> peerPromise;

  auto ownFuture  = ownPromise.get_future();
  auto peerFuture = peerPromise.get_future();

  StubTrader ownTrader{ownPromise, peerFuture};
  StubTrader peerTrader{peerPromise, ownFuture};

  auto ownData  = CreateData(length, ownSeed, ownOffset);
  auto peerData = CreateData(length, peerSeed, peerOffset);

  std::thread ownThread{Client, "own", std::ref(ownTrader), ownData};
  std::thread peerThread{Client, "peer", std::ref(peerTrader), peerData};

  ownThread.join();
  peerThread.join();
}
