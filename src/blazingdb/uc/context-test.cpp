#include "Context.hpp"

#include <gtest/gtest.h>

using blazingdb::uc::Context;

static bool FindIn(const std::vector<Context::Capability> &capabilities,
                   const std::string &memoryModel) {
  return capabilities.cend() !=
         std::find_if(capabilities.cbegin(), capabilities.cend(),
                      [&memoryModel](const Context::Capability &capability) {
                        return memoryModel == capability.memoryModel;
                      });
}

TEST(Context, LookupCapabilities) {
  const std::vector<Context::Capability> capabilities =
      Context::LookupCapabilities();

  // expected default memory models for any machine
  EXPECT_TRUE(FindIn(capabilities, "self"));
  EXPECT_TRUE(FindIn(capabilities, "tcp"));
  EXPECT_TRUE(FindIn(capabilities, "cma"));
}
