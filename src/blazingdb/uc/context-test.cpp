#include "Context.hpp"

#include <cuda.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using blazingdb::uc::Context;

static bool
FindIn(const std::vector<Context::Capability> &capabilities,
       const std::string &                     memoryModel) {
  return capabilities.cend() !=
         std::find_if(capabilities.cbegin(),
                      capabilities.cend(),
                      [&memoryModel](const Context::Capability &capability) {
                        return memoryModel == capability.memoryModel;
                      });
}

TEST(ContextTest, LookupCapabilities) {
  const std::vector<Context::Capability> capabilities =
      Context::LookupCapabilities();

  // expected default memory models for any machine
  EXPECT_TRUE(FindIn(capabilities, "self"));
  EXPECT_TRUE(FindIn(capabilities, "tcp"));
  EXPECT_TRUE(FindIn(capabilities, "cma"));
}

class MockCapabilities : public blazingdb::uc::Capabilities {
public:
  MOCK_CONST_METHOD0(resourceNames_, const std::vector<std::string> &());
  MOCK_CONST_METHOD0(AreNotThereResources_, bool());

  const std::vector<std::string> &
  resourceNames() const noexcept final {
    return resourceNames_();
  }

  bool
  AreNotThereResources() const noexcept final {
    return AreNotThereResources_();
  }
};

#include "internal/ManagedContext.hpp"
#include "internal/Resource.hpp"

TEST(ContextTest, BestContext) {
  MockCapabilities capabilities;
  cuInit(0);

  EXPECT_CALL(capabilities, AreNotThereResources_)
      .WillOnce(testing::Return(false));

  std::vector<std::string> resourceNames{"cuda_copy", "cuda_ipc"};
  EXPECT_CALL(capabilities, resourceNames_)
      .WillOnce(testing::ReturnRef(resourceNames));

  auto context = Context::BestContext(capabilities);

  auto managed =
      static_cast<blazingdb::uc::internal::ManagedContext *>(context.get());

  EXPECT_STREQ("cuda_ipc", managed->resource().md_name());
}
