set(test_name "server_test")

project(${test_name} CXX)

set(test_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/server-test.cpp
    ${CMAKE_SOURCE_DIR}/src/blazingdb/communication/Message.cc
    ${CMAKE_SOURCE_DIR}/src/blazingdb/communication/MessageToken.cc
    ${CMAKE_SOURCE_DIR}/src/blazingdb/communication/Server.cpp
)

add_executable(${test_name} ${test_sources})

target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${test_name} GTest::GTest GTest::Main ${CMAKE_THREAD_LIBS_INIT})

set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

add_test(NAME ${test_name} COMMAND ${test_name})
