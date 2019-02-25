#include <gtest/gtest.h>
#include "blazingdb/communication/messages/DataPivot.h"
#include "blazingdb/communication/messages/DataScatterMessage.h"
#include "blazingdb/communication/messages/PartitionPivotsMessage.h"
#include "blazingdb/communication/messages/SampleToNodeMasterMessage.h"
#include "blazingdb/communication/messages/Serializer.h"
#include "tests/utils/gdf_column.h"
#include "tests/utils/gdf_column_cpp.h"

struct QueryMessagesTest : public testing::Test {
    QueryMessagesTest() {
    }

    ~QueryMessagesTest() {
    }

    void SetUp() override {
    }

    void TearDown() override {
    }
};

/*
struct GpuFunctions {
    static void CopyGpuToCpu(std::string& result, blazingdb::test::gdf_column_cpp& column) {
        result.append(column.get_gdf_column()->data, column.get_gdf_column()->size);
        result.append((char*)column.get_gdf_column()->valid, column.get_gdf_column()->size);
    }
};


TEST_F(QueryMessagesTest, DataScatterMessage) {
    auto gdf_column_1 = blazingdb::test::build("data sample 1",
                                               "vali sample 1",
                                               blazingdb::test::GDF_INT64,
                                               12,
                                               blazingdb::test::TIME_UNIT_ms,
                                               "column name sample 1");

    auto gdf_column_2 = blazingdb::test::build("data sample 2",
                                               "vali sample 2",
                                               blazingdb::test::GDF_INT32,
                                               35,
                                               blazingdb::test::TIME_UNIT_s,
                                               "column name sample 2");

    auto gdf_column_cpp_1 = blazingdb::test::build(gdf_column_1.get(),
                                                   gdf_column_1->size,
                                                   gdf_column_1->size,
                                                   "column name another 1",
                                                   true,
                                                   123);

    auto gdf_column_cpp_2 = blazingdb::test::build(gdf_column_2.get(),
                                                   gdf_column_2->size,
                                                   gdf_column_2->size,
                                                   "column name another 2",
                                                   false,
                                                   3567);


    std::vector<blazingdb::test::gdf_column_cpp> columns;
    columns.emplace_back(gdf_column_cpp_1);
    columns.emplace_back(gdf_column_cpp_2);

    using DataScatterMessage = blazingdb::communication::messages::DataScatterMessage<blazingdb::test::gdf_column_cpp, GpuFunctions>;

    DataScatterMessage message(columns);

    auto ser = message.serializeToJson();
    std::cout << ser << std::endl;

    auto s = message.serializeToBinary();
    std::cout << "|" << s << "|" << std::endl;
}

TEST_F(QueryMessagesTest, SampleToNodeMasterMessage) {
    auto gdf_column_1 = blazingdb::test::build("data sample 1",
                                               "vali sample 1",
                                               blazingdb::test::GDF_INT64,
                                               12,
                                               blazingdb::test::TIME_UNIT_ms,
                                               "column name sample 1");

    auto gdf_column_2 = blazingdb::test::build("data sample 2",
                                               "vali sample 2",
                                               blazingdb::test::GDF_INT32,
                                               35,
                                               blazingdb::test::TIME_UNIT_s,
                                               "column name sample 2");

    auto gdf_column_cpp_1 = blazingdb::test::build(gdf_column_1.get(),
                                                   gdf_column_1->size,
                                                   gdf_column_1->size,
                                                   "column name another 1",
                                                   true,
                                                   123);

    auto gdf_column_cpp_2 = blazingdb::test::build(gdf_column_2.get(),
                                                   gdf_column_2->size,
                                                   gdf_column_2->size,
                                                   "column name another 2",
                                                   false,
                                                   3567);

    using Address = blazingdb::communication::Address;
    using NodeToken = blazingdb::communication::NodeToken;

    blazingdb::communication::Node node(NodeToken::Make("1.2.3.4", 1234),
                                        Address::Make("1.2.3.4", 1234));

    std::vector<blazingdb::test::gdf_column_cpp> samples;
    samples.emplace_back(gdf_column_cpp_1);
    samples.emplace_back(gdf_column_cpp_2);

    using SampleToNodeMasterMessage = blazingdb::communication::messages::SampleToNodeMasterMessage<blazingdb::test::gdf_column_cpp, GpuFunctions>;
    SampleToNodeMasterMessage message(node, samples);

    auto ser = message.serializeToJson();
    std::cout << ser << std::endl;

    auto s = message.serializeToBinary();
    std::cout << "|" << s << "|" << std::endl;
}
*/

TEST_F(QueryMessagesTest, PartitionPivotsMessage) {
    using Address = blazingdb::communication::Address;
    using NodeToken = blazingdb::communication::NodeToken;

    blazingdb::communication::Node node_1(NodeToken::Make("1.2.3.4", 1234),
                                          Address::Make("1.2.3.4", 1234));

//    blazingdb::communication::Node node_2(NodeToken::Make("5.6.7.8", 4564),
//                                          Address::Make("5.6.7.8", 4564));

    using blazingdb::communication::messages::DataPivot;

    std::string min_range_1("1111");
    std::string max_range_1("2222");
    DataPivot data_pivot_1(node_1, min_range_1, max_range_1);

//    std::string min_range_2("33");
//    std::string max_range_2("44");
//    DataPivot data_pivot_2(node_2, min_range_2, max_range_2);

    std::vector<DataPivot> nodes;
    nodes.emplace_back(data_pivot_1);
    //nodes.emplace_back(data_pivot_2);

//    {
//        rapidjson::StringBuffer string_buffer;
//        rapidjson::Writer<rapidjson::StringBuffer> writer(string_buffer);
//
//        node_1.serializeToJson(writer);
//
//        std::cout << std::string(string_buffer.GetString(), string_buffer.GetSize()) << std::endl;
//    }

    // Create message
    using PartitionPivotsMessage = blazingdb::communication::messages::PartitionPivotsMessage;
    PartitionPivotsMessage message(nodes);

    // Serialize message
    const std::string serialize_message = message.serializeToJson();
    //std::cout << serialize_message << std::endl;

    // Deserialize message
    std::shared_ptr<PartitionPivotsMessage> deserialize_message = PartitionPivotsMessage::make(serialize_message);

    // Tests
    const auto& pivots = deserialize_message->getDataPivots();
    ASSERT_EQ(pivots.size(), 1);
    ASSERT_EQ(pivots[0].getMinRange(), min_range_1);
    ASSERT_EQ(pivots[0].getMaxRange(), max_range_1);
}
