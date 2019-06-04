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
     const int            device,
     const int (&own_pipedes)[2],
     const int (&peer_pipedes)[2]) {
  cudaSetDevice(device);
  const void *     data       = CreateData(length, seed, offset);
  int        pipedes[2] = {own_pipedes[0], peer_pipedes[1]};
  StubTrader trader{pipedes};
  auto       context = decaycopy(builder)(trader);
  Client(name, *context, data);
}

template <class ContextBuilder>
static void
Test(ContextBuilder &&builder, int ownDevice, int peerDevice) {
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
         ownDevice,
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
         peerDevice,
         peer_pipedes,
         own_pipedes);
    std::exit(EXIT_SUCCESS);
  }
}

class ApiOnProcessesTest
    : public testing::TestWithParam<
          testing::tuple<typename std::decay<Context::Builder>::type,
                         std::pair<int, int>>> {};

TEST_P(ApiOnProcessesTest, WithIPC) {
  int ownDevice, peerDevice;
  std::tie(ownDevice, peerDevice) = testing::get<1>(GetParam());
  ::Test(std::move(testing::get<0>(GetParam())), ownDevice, peerDevice);
}

#define Value(x, y, z)                                                         \
  testing::make_tuple<typename std::decay<Context::Builder>::type>(            \
      x, std::make_pair(y, z))

INSTANTIATE_TEST_SUITE_P(OneGPU,
                         ApiOnProcessesTest,
                         testing::Values(Value(Context::IPC, 0, 0)));

INSTANTIATE_TEST_SUITE_P(DISABLED_TwoGPUs,
                         ApiOnProcessesTest,
                         testing::Values(Value(Context::IPC, 0, 1),
                                         Value(Context::GDR, 0, 1)));

TEST(ApiOnProcessesTest, Direct) {
  using namespace blazingdb::uc;

  int pipedes[2];
  pipe(pipedes);

  pid_t pid = fork();

  ASSERT_NE(-1, pid);

  static constexpr std::size_t length = 20;

  if (pid) {
    close(pipedes[0]);
    const void *data = CreateData(length, ownSeed, ownOffset);
    Print("own", data, length);

    auto context = Context::IPC();
    auto agent   = context->Agent();
    auto buffer  = agent->Register(data, length);

    auto serializedRecord = buffer->SerializedRecord();

    write(pipedes[1], serializedRecord->Data(), serializedRecord->Size());
    int stat_loc;
    waitpid(pid, &stat_loc, WUNTRACED | WCONTINUED);
  } else {
    close(pipedes[1]);
    const void *data = CreateData(length, peerSeed, peerOffset);
    Print("peer", data, length);

    auto context = Context::IPC();
    auto agent   = context->Agent();
    auto buffer  = agent->Register(data, length);

    std::uint8_t recordData[104];
    read(pipedes[0], recordData, 104);
    auto transport = buffer->Link(recordData);

    auto future = transport->Get();
    future.wait();

    Print("peer", data, length);
    std::exit(EXIT_SUCCESS);
  }
}
