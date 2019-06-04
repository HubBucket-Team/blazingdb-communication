#include "TCPContext.hpp"

#include <gtest/gtest.h>

#include "../../api-common-test.hpp"

void
Sender(int (&pipedes)[2],
       const pid_t       pid,
       const std::size_t incrementalLength) {
  cuInit(0);
  close(pipedes[0]);

  const void *data = CreateData(incrementalLength, ownSeed, 0);

  auto context = Context::IPC();
  auto agent   = context->Agent();
  auto buffer  = agent->Register(data, incrementalLength);

  auto serializedRecord = buffer->SerializedRecord();

  write(pipedes[1], serializedRecord->Data(), serializedRecord->Size());

  int stat_loc;
  waitpid(pid, &stat_loc, WUNTRACED | WCONTINUED);
  EXPECT_EQ(0, stat_loc);
  close(pipedes[1]);
  std::exit(EXIT_SUCCESS);
}

void
Receiver(int (&pipedes)[2], const std::size_t incrementalLength) {
  cuInit(0);
  close(pipedes[1]);

  const void *data = CreateData(incrementalLength, peerSeed, 0);

  auto context = Context::IPC();
  auto agent   = context->Agent();
  auto buffer  = agent->Register(data, incrementalLength);

  std::uint8_t *recordData = new std::uint8_t[context->serializedRecordSize()];
  read(pipedes[0], recordData, context->serializedRecordSize());
  auto transport = buffer->Link(recordData);

  auto future = transport->Get();
  future.wait();

  const void *const expected = CreateHostData(incrementalLength, ownSeed, 0);

  std::uint8_t *result = new std::uint8_t[incrementalLength];

  cudaError_t cudaError;

  cudaError =
      cudaMemcpy(result, data, incrementalLength, cudaMemcpyDeviceToHost);
  EXPECT_EQ(cudaSuccess, cudaError);

  cudaError = cudaDeviceSynchronize();
  EXPECT_EQ(cudaSuccess, cudaError);

  EXPECT_EQ(0, std::memcmp(expected, result, incrementalLength));

  delete[] recordData;

  close(pipedes[0]);
  std::exit(EXIT_SUCCESS);
}

TEST(TCPContextTest, Sending) {
  using blazingdb::uc::Context;

  int pipedes[2];
  pipe(pipedes);

  pid_t pid = fork();
  ASSERT_NE(-1, pid);

  if (pid) {
    pid = fork();
    ASSERT_NE(-1, pid);
    if (pid) {
      int stat_loc;
      waitpid(pid, &stat_loc, WCONTINUED);
      EXPECT_EQ(0, stat_loc);
    } else {
      Sender(pipedes, pid, 1000);
    }
  } else {
    Receiver(pipedes, 1000);
  }
}
