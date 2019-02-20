#include <algorithm>
#include "gtest/gtest.h"
#include "blazingdb/communication/Message.h"
#include "blazingdb/communication/MessageToken.h"
#include "blazingdb/communication/Server.h"

namespace {

    using MessagePointer = std::shared_ptr<blazingdb::communication::Message>;

    struct ServerTest : public testing::Test {
        ServerTest() {
        }

        ~ServerTest() {
        }

        void SetUp() override {
        }

        void TearDown() override {
        }

        std::uint64_t GetToken(MessagePointer& pointer) {
            return pointer->getMessageToken().getToken();
        }

        MessagePointer MessageFactory(std::uint64_t value) {
            using blazingdb::communication::Message;
            using blazingdb::communication::MessageToken;
            return std::make_shared<Message>(MessageToken(value));
        }
    };


    TEST_F(ServerTest, Input) {
        blazingdb::communication::Server server;

        const int TOTAL = 10;
        std::vector<MessagePointer> input_messages;

        int x = 0;
        std::generate_n(std::back_inserter(input_messages),
                        TOTAL,
                        [&x, this]() {
                            return MessageFactory(x++);
                        });

        std::for_each(input_messages.begin(),
                      input_messages.end(),
                      [&server](MessagePointer& pointer) {
                          server.putMessage(pointer);
                      });

        std::vector<MessagePointer> output_messages;

        std::generate_n(std::back_inserter(output_messages),
                        TOTAL,
                        [&server]() {
                            return server.getMessage();
                        });

        for (int k = 0; k < TOTAL; ++k) {
            ASSERT_EQ(GetToken(input_messages[k]), k);
            ASSERT_EQ(GetToken(input_messages[k]), GetToken(output_messages[k]));
        }
    }
}
