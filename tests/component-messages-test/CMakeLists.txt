set(test_name "component-messages-test")

project(${test_name} CXX)

set(test_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/component-messages-test.cpp
    ${CMAKE_SOURCE_DIR}/tests/utils/gdf_column.cpp
    ${CMAKE_SOURCE_DIR}/tests/utils/gdf_column_cpp.cpp
    ${CMAKE_SOURCE_DIR}/src/blazingdb/communication/Address.cc
    ${CMAKE_SOURCE_DIR}/src/blazingdb/communication/ContextToken.cc
    ${CMAKE_SOURCE_DIR}/src/blazingdb/communication/Node.cc
    ${CMAKE_SOURCE_DIR}/src/blazingdb/communication/messages/Message.cc
    ${CMAKE_SOURCE_DIR}/src/blazingdb/communication/messages/MessageToken.cc
    ${CMAKE_SOURCE_DIR}/src/blazingdb/communication/messages/DataPivot.cpp
    ${CMAKE_SOURCE_DIR}/src/blazingdb/communication/messages/PartitionPivotsMessage.cpp
    ${CMAKE_SOURCE_DIR}/src/blazingdb/communication/messages/NodeDataMessage.cc
)

add_executable(${test_name} ${test_sources})

target_include_directories(${test_name} PRIVATE ${RAPIDJSON_ROOT}/include)
target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)

message("DATA: ${RAPIDJSON_ROOT}")
target_link_libraries(${test_name} GTest::GTest GTest::Main)

set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

add_test(NAME ${test_name} COMMAND ${test_name})
