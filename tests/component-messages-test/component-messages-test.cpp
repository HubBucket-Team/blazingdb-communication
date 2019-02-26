#include <gtest/gtest.h>
#include "blazingdb/communication/messages/DataPivot.h"
#include "blazingdb/communication/messages/DataScatterMessage.h"
#include "blazingdb/communication/messages/PartitionPivotsMessage.h"
#include "blazingdb/communication/messages/SampleToNodeMasterMessage.h"
#include "blazingdb/communication/messages/NodeDataMessage.h"
#include "blazingdb/communication/messages/Serializer.h"
#include "tests/utils/gdf_column.h"
#include "tests/utils/gdf_column_cpp.h"

struct ComponentMessagesTest : public testing::Test {
    ComponentMessagesTest() {
    }

    ~ComponentMessagesTest() {
    }

    void SetUp() override {
    }

    void TearDown() override {
    }
};


struct GpuFunctions {
    using DType = blazingdb::test::gdf_dtype;
    using DTypeInfo = blazingdb::test::gdf_dtype_extra_info;
    using TimeUnit = blazingdb::test::gdf_time_unit;

    using DataTypePointer = void*;
    using ValidTypePointer = blazingdb::test::gdf_valid_type*;


    static void copyGpuToCpu(std::size_t& binary_pointer, std::string& result, blazingdb::test::gdf_column_cpp& column) {
        std::size_t data_size = getDataCapacity(column.get_gdf_column());
        std::memcpy(&result[binary_pointer], column.get_gdf_column()->data, data_size);
        binary_pointer += data_size;

        std::size_t valid_size = getValidCapacity(column.get_gdf_column());
        std::memcpy(&result[binary_pointer], column.get_gdf_column()->valid, valid_size);
        binary_pointer += valid_size;
    }

    static std::size_t getDataCapacity(blazingdb::test::gdf_column* column) {
        return column->size;
    }

    static std::size_t getValidCapacity(blazingdb::test::gdf_column* column) {
        return column->size * getDTypeSize(column->dtype);
    }

    static std::size_t getDTypeSize(blazingdb::test::gdf_dtype type) {
        return blazingdb::test::get_dtype_width(type);
    }
};


TEST_F(ComponentMessagesTest, DataScatterMessage) {
    auto gdf_column_1 = blazingdb::test::build(8,
                                               blazingdb::test::GDF_INT16,
                                               4,
                                               blazingdb::test::TIME_UNIT_ms,
                                               "column name 1");

    auto gdf_column_2 = blazingdb::test::build(16,
                                               blazingdb::test::GDF_INT32,
                                               8,
                                               blazingdb::test::TIME_UNIT_s,
                                               "column name 2");

    auto gdf_column_cpp_1 = blazingdb::test::build(gdf_column_1.get(),
                                                   gdf_column_1->size,
                                                   gdf_column_1->size * blazingdb::test::get_dtype_width(gdf_column_1->dtype),
                                                   "column another name 1",
                                                   false,
                                                   123);

    auto gdf_column_cpp_2 = blazingdb::test::build(gdf_column_2.get(),
                                                   gdf_column_2->size,
                                                   gdf_column_2->size * blazingdb::test::get_dtype_width(gdf_column_2->dtype),
                                                   "column name another 2",
                                                   false,
                                                   3567);

    std::vector<blazingdb::test::gdf_column_cpp> columns;
    columns.emplace_back(gdf_column_cpp_1);
    columns.emplace_back(gdf_column_cpp_2);

    using DataScatterMessage = blazingdb::communication::messages::DataScatterMessage<blazingdb::test::gdf_column_cpp,
                                                                                      blazingdb::test::gdf_column,
                                                                                      GpuFunctions>;

    std::string json_data;
    std::string binary_data;
    {
        DataScatterMessage message(columns);

        json_data = message.serializeToJson();
        binary_data = message.serializeToBinary();
    }
    {
        auto message = DataScatterMessage::make(json_data, binary_data);
        const auto& columns = message->getColumns();

        ASSERT_TRUE(columns[0] == gdf_column_cpp_1);
        ASSERT_TRUE(columns[1] == gdf_column_cpp_2);
    }
}


TEST_F(ComponentMessagesTest, SampleToNodeMasterMessage) {
    // Test data - create samples
    auto gdf_column_1 = blazingdb::test::build(16,
                                               blazingdb::test::GDF_INT64,
                                               12,
                                               blazingdb::test::TIME_UNIT_ms,
                                               "column name 1");

    auto gdf_column_2 = blazingdb::test::build(24,
                                               blazingdb::test::GDF_INT32,
                                               5,
                                               blazingdb::test::TIME_UNIT_s,
                                               "column name 2");

    auto gdf_column_cpp_1 = blazingdb::test::build(gdf_column_1.get(),
                                                   gdf_column_1->size,
                                                   gdf_column_1->size * blazingdb::test::get_dtype_width(gdf_column_1->dtype),
                                                   "column name another 1",
                                                   true,
                                                   123);

    auto gdf_column_cpp_2 = blazingdb::test::build(gdf_column_2.get(),
                                                   gdf_column_2->size,
                                                   gdf_column_2->size * blazingdb::test::get_dtype_width(gdf_column_2->dtype),
                                                   "column name another 2",
                                                   false,
                                                   3567);

    std::vector<blazingdb::test::gdf_column_cpp> samples;
    samples.emplace_back(gdf_column_cpp_1);
    samples.emplace_back(gdf_column_cpp_2);

    // Test data - create node
    using Address = blazingdb::communication::Address;
    blazingdb::communication::Node node(Address::Make("1.2.3.4", 1234));

    // Message alias
    using SampleToNodeMasterMessage = blazingdb::communication::messages::SampleToNodeMasterMessage<blazingdb::test::gdf_column_cpp,
                                                                                                    blazingdb::test::gdf_column,
                                                                                                    GpuFunctions>;

    // Serialize data
    std::string json_data;
    std::string binary_data;

    // Serialize message
    {
        SampleToNodeMasterMessage message(node, samples);

        json_data = message.serializeToJson();
        binary_data = message.serializeToBinary();
    }

    // Deserialize message & test
    {
        std::shared_ptr<SampleToNodeMasterMessage> message = SampleToNodeMasterMessage::make(json_data, binary_data);

        // Testing
        ASSERT_EQ(message->getNode(), node);
        ASSERT_EQ(message->getSamples().size(), samples.size());
        for (std::size_t k = 0; k < samples.size(); ++k) {
            ASSERT_TRUE(message->getSamples()[k] == samples[k]);
        }
    }
}


TEST_F(ComponentMessagesTest, PartitionPivotsMessage) {
    using Address = blazingdb::communication::Address;

    blazingdb::communication::Node node_1(Address::Make("1.2.3.4", 1234));

//    blazingdb::communication::Node node_2(Address::Make("5.6.7.8", 4564));

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


TEST_F(ComponentMessagesTest, NodeDataMessage) {
    using Address = blazingdb::communication::Address;

    blazingdb::communication::Node node(Address::Make("1.2.3.4", 1234));
    // Create message
    using blazingdb::communication::messages::NodeDataMessage;
    NodeDataMessage message(node);

    // Serialize message
    const std::string serialize_message = message.serializeToJson();
    std::cout << serialize_message << std::endl;

    // Deserialize message
    std::shared_ptr<NodeDataMessage> deserialize_message = NodeDataMessage::make(serialize_message, "");

    // Tests
    // ASSERT_EQ(message.node == deserialize_message->node, true);
}
