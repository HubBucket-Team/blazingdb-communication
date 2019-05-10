set(test_name "component-messages-test")

project(${test_name} CXX)

set (COMMUNICATION_DIR ${CMAKE_SOURCE_DIR}/src/blazingdb/communication)

set(test_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/component-messages-test.cpp
    ${CMAKE_SOURCE_DIR}/tests/utils/gdf_column.cpp
    ${CMAKE_SOURCE_DIR}/tests/utils/gdf_column_cpp.cpp
    ${COMMUNICATION_DIR}/Address.cc
    ${COMMUNICATION_DIR}/Address-Internal.cc
    ${COMMUNICATION_DIR}/ContextToken.cc
    ${COMMUNICATION_DIR}/Node.cc
    ${COMMUNICATION_DIR}/messages/Message.cc
    ${COMMUNICATION_DIR}/messages/MessageToken.cc
    ${COMMUNICATION_DIR}/messages/DataPivot.cpp
    ${COMMUNICATION_DIR}/messages/PartitionPivotsMessage.cpp
    ${COMMUNICATION_DIR}/messages/NodeDataMessage.cc
    ${COMMUNICATION_DIR}/internal/Trader.cpp
    ${COMMUNICATION_DIR}/internal/TraderLock.cpp
)

add_executable(${test_name} ${test_sources})

target_include_directories(${test_name} PRIVATE ${RAPIDJSON_ROOT}/include)
target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)

message("DATA: ${RAPIDJSON_ROOT}")
target_link_libraries(${test_name}
  GTest::GTest GTest::Main Boost::system blazingdb-uc
  cuda
  /usr/local/cuda/lib64/libcudart.so
  ${UCX_UCT_STATIC_LIB}
  ${GDR_STATIC_LIB}
  ${UCX_UCS_STATIC_LIB}
  ${UCX_UCM_STATIC_LIB})

set_target_properties(${test_name}
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

add_test(NAME ${test_name} COMMAND ${test_name})
