#include "Context.hpp"

#include <gtest/gtest.h>

#include "api-common-test.hpp"

class TestAllocationParam {
  friend std::ostream &
  operator<<(std::ostream &os, const TestAllocationParam &param) {
    return os << "[size=" << param.size() << "]";
  }

public:
  explicit TestAllocationParam(const std::size_t size) : size_{size} {}

  std::size_t
  size() const noexcept {
    return size_;
  }

private:
  std::size_t size_;
};

class ContextMemoryStressTest
    : public ::testing::TestWithParam<TestAllocationParam> {
protected:
};

INSTANTIATE_TEST_SUITE_P(MemoryAllocations,
                         ContextMemoryStressTest,
                         ::testing::Values(200, 400, 80000, 320000000));

void
Sender(int (&pipedes)[2],
       const pid_t       pid,
       const std::size_t incrementalLength) {
  cuInit(0);
  close(pipedes[0]);

  const void *const data = CreateData(incrementalLength, ownSeed, ownOffset);

  auto context = Context::IPC();
  auto agent   = context->Agent();
  auto buffer  = agent->Register(data, incrementalLength);

  auto serializedRecord = buffer->SerializedRecord();

  write(pipedes[1], serializedRecord->Data(), serializedRecord->Size());

  int stat_loc;
  waitpid(pid, &stat_loc, WUNTRACED | WCONTINUED);
  EXPECT_EQ(0, stat_loc);
}

void
Receiver(int (&pipedes)[2], const std::size_t incrementalLength) {
  cuInit(0);
  close(pipedes[1]);

  const void *const data = CreateData(incrementalLength, peerSeed, peerOffset);

  auto context = Context::IPC();
  auto agent   = context->Agent();
  auto buffer  = agent->Register(data, incrementalLength);

  std::uint8_t *recordData = new std::uint8_t[context->serializedRecordSize()];
  read(pipedes[0], recordData, context->serializedRecordSize());
  auto transport = buffer->Link(recordData);

  auto future = transport->Get();
  future.wait();

  const void *const expectedData =
      CreateData(incrementalLength, ownSeed, ownOffset);

  std::uint8_t *expected = new std::uint8_t[incrementalLength];
  std::uint8_t *result   = new std::uint8_t[incrementalLength];

  cudaError_t cudaError;

  cudaError = cudaMemcpy(
      expected, expectedData, incrementalLength, cudaMemcpyDeviceToHost);
  EXPECT_EQ(cudaSuccess, cudaError);

  cudaError =
      cudaMemcpy(result, data, incrementalLength, cudaMemcpyDeviceToHost);
  EXPECT_EQ(cudaSuccess, cudaError);

  EXPECT_EQ(0, std::memcmp(expected, result, incrementalLength));

  delete[] recordData;

  std::exit(EXIT_SUCCESS);
}

TEST_P(ContextMemoryStressTest, HugeAllocations) {
  using blazingdb::uc::Context;

  int pipedes[2];
  pipe(pipedes);

  pid_t pid = fork();
  ASSERT_NE(-1, pid);

  if (pid) {
    Sender(pipedes, pid, GetParam().size());
  } else {
    Receiver(pipedes, GetParam().size());
  }
}
