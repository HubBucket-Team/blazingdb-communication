add_library (blazingdb-communication
  Address.cc
  Buffer.cc
  ContextToken.cc
  Context.cc
  Cluster.cc
  Manager.cc
  Message.cc
  MessageToken.cc
  network/Client.cc
  network/Server.cc
  Node.cc
  NodeToken.cc
  Router.cc
  messages/DataPivot.cpp
  messages/PartitionPivotsMessage.cpp
  messages/NodeDataMessage.cc
)

target_include_directories (blazingdb-communication
  PUBLIC ${PROJECT_SOURCE_DIR}/src
)

find_package(Boost REQUIRED COMPONENTS system)

target_link_libraries(blazingdb-communication Boost::system)

install (TARGETS blazingdb-communication
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install (DIRECTORY "${PROJECT_SOURCE_DIR}/src/blazingdb" DESTINATION "include"
  FILES_MATCHING PATTERN "*.h" PATTERN "*Internal.h" EXCLUDE
)

if (GTEST_FOUND)
enable_testing ()
add_executable (blazingdb-communication-test EXCLUDE_FROM_ALL
  buffer-test.cc
  network/integration-server-client-test.cc
)
target_compile_options (blazingdb-communication
  PUBLIC -g -O0 -fprofile-arcs -ftest-coverage -fPIC)
target_include_directories (blazingdb-communication-test
  PUBLIC src
  PUBLIC ${GTEST_INCLUDE_DIRS}
  PUBLIC ${RAPIDJSON_INCLUDE_DIRS})
target_link_libraries (blazingdb-communication-test
  GTest::GTest gmock_main gmock blazingdb-communication gcov --coverage)
gtest_discover_tests (blazingdb-communication-test)
add_custom_target (coverage
  COMMAND
    ${CMAKE_BINARY_DIR}/src/blazingdb/communication/blazingdb-communication-test
  COMMAND lcov -c -d ${CMAKE_BINARY_DIR}/src -o coverage.info
  COMMAND genhtml coverage.info -o ${CMAKE_BINARY_DIR}/coverage-html
)
add_dependencies (coverage blazingdb-communication-test)
endif ()
