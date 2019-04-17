#include "api-common-test.hpp"

namespace {
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
}  // namespace

template <class Tp>
static UC_INLINE typename std::decay<Tp>::type
decaycopy(Tp &&_t) {
  return std::forward<Tp>(_t);
}

template <class ContextBuilder>
static void
Exec(ContextBuilder &&    builder,
     const std::string &  name,
     const std::uint64_t  seed,
     const std::ptrdiff_t offset,
     const int (&own_pipedes)[2],
     const int (&peer_pipedes)[2]) {
  cuInit(0);
  void *     data       = CreateData(length, seed, offset);
  int        pipedes[2] = {own_pipedes[0], peer_pipedes[1]};
  StubTrader trader{pipedes};
  auto       context = decaycopy(builder)(trader);
  Client(name, *context, data);
}

template <class ContextBuilder>
static void
Test(ContextBuilder &&builder) {
  int own_pipedes[2];
  int peer_pipedes[2];

  pipe(own_pipedes);
  pipe(peer_pipedes);

  pid_t pid = fork();

  ASSERT_NE(-1, pid);

  if (pid) {
    Exec(std::forward<ContextBuilder>(builder),
         "own",
         ownSeed,
         ownOffset,
         own_pipedes,
         peer_pipedes);
    int stat_loc;
    pid = waitpid(pid, &stat_loc, 0);
    ASSERT_EQ(0, stat_loc);
    ASSERT_NE(-1, pid);
  } else {
    Exec(std::forward<ContextBuilder>(builder),
         "peer",
         peerSeed,
         peerOffset,
         peer_pipedes,
         own_pipedes);
    std::exit(EXIT_SUCCESS);
  }
}

TEST(ApiTest, ProcessesWithIPC) { ::Test(Context::IPC); }

TEST(ApiTest, ProcessesWithGDR) { ::Test(Context::GDR); }
