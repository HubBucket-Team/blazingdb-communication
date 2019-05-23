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

//app 1, part of a 2-part IPC example                                                                                                                                                                                                                                                                                    
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define DSIZE 3


#include <cstring>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define cudaCheckErrors(msg) \
  do { \
  cudaError_t __err = cudaGetLastError(); \
  if (__err != cudaSuccess) { \
  fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
    msg, cudaGetErrorString(__err), \
          __FILE__, __LINE__); \
  fprintf(stderr, "*** FAILED - ABORTING\n"); \
  exit(1); \
  } \
  } while (0)


TEST(ApiOnProcessesTest, App1) {
  cuInit(0);
  using namespace blazingdb::uc;

  system("rm -f testfifo"); // remove any debris                                                                                                                                                                                                                                                                         
  int ret = mkfifo("testfifo", 0600); // create fifo                                                                                                                                                                                                                                                                     
  if (ret != 0) {printf("mkfifo error: %d\n",ret); return;}

  float h_nums[] = {1.1111, 2.2222, 3.141592654};
  float *data;
  cudaIpcMemHandle_t my_handle;
  cudaMalloc((void **)&data, DSIZE*sizeof(float));
  cudaCheckErrors("malloc fail");

  cudaMemcpy(data, h_nums, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("memcoy fail");
   
  auto context = Context::IPCView();
  auto agent   = context->Agent();
  const void* dptr = data; 
  auto buffer  = agent->Register(dptr, DSIZE*sizeof(float));
  auto serializedRecord = buffer->SerializedRecord();

  cudaCheckErrors("get IPC handle fail");
  FILE *fp;
  printf("waiting for app2\n");
  fp = fopen("testfifo", "w");
  if (fp == NULL) {printf("fifo open fail \n"); return;}
  for (int i=0; i < sizeof(my_handle); i++){
    ret = fprintf(fp,"%c", serializedRecord->Data()[i]);
    if (ret != 1) printf("ret = %d\n", ret);}
  
  fclose(fp);
  cudaFree(data);
  printf("App1 is finishing after cudaFree\n");
  // float *result = (float *)malloc(DSIZE*sizeof(float));
  // cudaMemcpy(result, data, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  // if (!(*result)) printf("Fail!\n");
  // else printf("Success!\n");
  system("rm testfifo");
}



TEST(ApiOnProcessesTest, App2) {
  using namespace blazingdb::uc;
  cuInit(0);

  float *data;
  float h_nums[DSIZE];

  cudaIpcMemHandle_t my_handle;
  unsigned char handle_buffer[sizeof(my_handle)+1];
  memset(handle_buffer, 0, sizeof(my_handle)+1);
  FILE *fp;
  fp = fopen("testfifo", "r");
  if (fp == NULL) {printf("fifo open fail \n"); return;}
  int ret;
  for (int i = 0; i < sizeof(my_handle); i++){
    ret = fscanf(fp,"%c", handle_buffer+i);
    if (ret == EOF) printf("received EOF\n");
    else if (ret != 1) printf("fscanf returned %d\n", ret);}
  memcpy((unsigned char *)(&my_handle), handle_buffer, sizeof(my_handle));
  
  auto context = Context::IPCView(); 
  auto agent   = context->Agent(); 
  const void *dptr = nullptr;

  auto buffer  = agent->Register(dptr, 0);// ViewBuffer { Regis
  auto transport = buffer->Link(handle_buffer); // Link() { cudaIpcGetMemHandle } 
  auto future = transport->Get();
  future.wait();

  data = (float*)dptr; 
  // cudaIpcOpenMemHandle((void **)&data, my_handle, cudaIpcMemLazyEnablePeerAccess);
  // cudaCheckErrors("IPC handle fail");
  //cudaMemset(data, 1, sizeof(float));             

  sleep(5); // wait for app 2 to modify data                                                                                                                                                                                                                                                                             
  cudaMemcpy(h_nums, data, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cudaCheckErrors("memset fail");                                                                                                            
  cudaCheckErrors("memcopy fail");
  printf("values read from GPU memory : %f %f %f\n", h_nums[0], h_nums[1], h_nums[2]);

  float * data_clone;
  cudaMalloc((void **)&data_clone, DSIZE*sizeof(float));
  cudaMemcpy(data_clone, data, DSIZE*sizeof(float), cudaMemcpyDeviceToDevice);
  cudaCheckErrors("cudaMemcpyDeviceToDevice fail");

  cudaIpcCloseMemHandle(&data);
  cudaFree(&data);
}

TEST(ApiOnProcessesTest, DirectView) {
  using namespace blazingdb::uc;
  cuInit(0);

  int pipedes[2];
  pipe(pipedes);

  pid_t pid = fork();

  ASSERT_NE(-1, pid);

  static constexpr std::size_t length = 32;

  if (pid) {
    std::cout << "*****Parent process\n";
    close(pipedes[0]);
    const void *data = CreateData(length, ownSeed, ownOffset);
    std::cout << "pointer address : " << data << std::endl;
    
    Print("own", data, length);

    auto context = Context::IPCView();
    auto agent   = context->Agent();
    auto buffer  = agent->Register(data, length);

    auto serializedRecord = buffer->SerializedRecord();
    std::cout << "--->>>>>>>\n";

    write(pipedes[1], serializedRecord->Data(), serializedRecord->Size());
    int stat_loc;
    waitpid(pid, &stat_loc, WUNTRACED | WCONTINUED);
    std::cout << "Parent process*****\n";
  } else {
    std::cout << "*****Child process\n";
    close(pipedes[1]);
    const void *data = nullptr;

    auto context = Context::IPCView(); // new builder: ViewContext : ManageContext ( Agent() -> ViewAgent ) 
    auto agent   = context->Agent(); // ViewAgent
    // caso 1: referencia al puntero
      // caso 2: auto buffer =  agent->View()
    // 
    auto buffer  = agent->Register(data, length);// ViewBuffer { Register()-> ViewBuffer }
 
    std::uint8_t recordData[sizeof(cudaIpcMemHandle_t)]; // CU_IPC_HANDLE_SIZE: 64
    read(pipedes[0], recordData, sizeof(cudaIpcMemHandle_t));
    std::cout << "<<<------------\n";

    sleep(5);
    auto transport = buffer->Link(recordData); // Link() { cudaIpcGetMemHandle } 
    auto future = transport->Get();
    future.wait();

    Print("peer", data, length);
    std::cout << "Child process*****\n";
    cudaIpcCloseMemHandle((void*)data);
    std::exit(EXIT_SUCCESS);
  }
}
