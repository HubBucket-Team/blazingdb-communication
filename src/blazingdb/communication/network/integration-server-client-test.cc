#include "Client.h"
#include "Server.h"

#include <blazingdb/communication/MessageSerializer.h>

#include <chrono>
#include <memory>
#include <thread>

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

class MockNodeToken : public blazingdb::communication::NodeToken {
public:
  MOCK_CONST_METHOD1(SameValueAs, bool(const NodeToken &));
};

class MockAddress : public blazingdb::communication::Address {
public:
  MOCK_CONST_METHOD1(SameValueAs, bool(const Address &));
};

class FooMessage : public blazingdb::communication::Message {
public:
  using blazingdb::communication::Message::Message;

  const std::string code() const { return "qwerty"; }

  MOCK_CONST_METHOD1(SameIdentityAs, bool(const Message &));
  MOCK_CONST_METHOD0(Identity,
                     const blazingdb::communication::MessageToken &());
};

class FooMessageSerializer {
public:
  blazingdb::communication::Buffer serialize(const FooMessage &fooMessage) {
    rapidjson::StringBuffer stringBuffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(stringBuffer);

    writer.StartObject();
    writer.Key("code");
    writer.String(fooMessage.code().c_str());
    writer.EndObject();

    const std::string *json = new std::string(stringBuffer.GetString());

    return blazingdb::communication::Buffer(
        reinterpret_cast<const std::uint8_t *>(json->data()), json->size());
  }
};

FooMessage *CreateFooMessage() {
  std::unique_ptr<blazingdb::communication::MessageToken> messageToken =
      blazingdb::communication::MessageToken::Make();

  FooMessage *fooMessage = new FooMessage{std::move(messageToken)};

  return fooMessage;
}

TEST(ServerTest, Ehlo) {
  std::unique_ptr<blazingdb::communication::network::Server> server =
      blazingdb::communication::network::Server::Make();

  std::thread serverThread([&server]() { server->Run(); });

  std::this_thread::sleep_for(std::chrono::seconds(1));

  std::unique_ptr<blazingdb::communication::network::Client> client =
      blazingdb::communication::network::Client::Make();

  FooMessageSerializer serializer;
  blazingdb::communication::Buffer buffer =
      serializer.serialize(*CreateFooMessage());

  std::shared_ptr<blazingdb::communication::NodeToken> nodeToken =
      std::make_shared<MockNodeToken>();

  std::shared_ptr<blazingdb::communication::Address> address =
      blazingdb::communication::Address::Make("localhost", 8000);

  blazingdb::communication::Node node{std::move(nodeToken), std::move(address)};

  client->Send(node, buffer);

  server->Close();
  serverThread.join();
}
