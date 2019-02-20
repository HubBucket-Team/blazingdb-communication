#include "Server.h"
#include "Client.h"

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

TEST(ServerTest, Ehlo) {
  std::unique_ptr<blazingdb::communication::network::Server> server =
      blazingdb::communication::network::Server::Make();

  std::thread serverThread([&server]() { server->Run(); });

  std::this_thread::sleep_for(std::chrono::seconds(1));

  server->Close();

  std::unique_ptr<blazingdb::communication::network::Client> client =
      blazingdb::communication::network::Client::Make();

  const std::uint8_t content[] = "HOLA";
  blazingdb::communication::Buffer buffer(content, sizeof(content));

  blazingdb::communication::NodeToken nodeToken{0};
  blazingdb::communication::Address address{"", 0};
  blazingdb::communication::Node node{nodeToken, address};

  client->Send(node, buffer);

  serverThread.join();
}
