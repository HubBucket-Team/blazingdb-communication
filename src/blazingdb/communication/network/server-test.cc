#include "Server.h"

#include <gtest/gtest.h>

TEST(ServerTest, Ehlo) {
  std::unique_ptr<blazingdb::communication::network::Server> server =
      blazingdb::communication::network::Server::Make();

  server->Run();
}
