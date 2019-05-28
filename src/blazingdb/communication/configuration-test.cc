#include "Configuration.h"

#include <gtest/gtest.h>

TEST(ConfigurationTest, Initialization) {
  using blazingdb::communication::Configuration;

  Configuration::Set(true);

  const Configuration &configuration = Configuration::Instance();

  EXPECT_EQ(true, configuration.WithGDR());
}
