#include "Configuration.h"

#include <gtest/gtest.h>

TEST(ConfigurationTest, Initialization) {
  using blazgindb::communication::Configuration;

  Configuration::Set(true);

  const Configuration &configuration = Configuration::Instance();

  EXPECT_EQ(true, configuration.WithGDR());
}
