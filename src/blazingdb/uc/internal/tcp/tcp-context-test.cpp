#include "TCPContext.hpp"

#include <gtest/gtest.h>

#include "../../api-common-test.hpp"

TEST(TCPOnProcessesTest, Direct) {
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

    auto context = Context::TCP();
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

    auto context = Context::TCP();
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
