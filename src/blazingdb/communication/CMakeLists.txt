add_library (blazingdb-communication
  Address.cc
  ContextToken.cc
  Context.cc
  Cluster.cc
  Manager.cc
  network/Client.cc
  network/Server.cc
  Node.cc
  Router.cc
  messages/Message.cc
  messages/MessageToken.cc
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
set (TEST_FILES
  manager-test.cc
  network/integration-server-client-test.cc
)
enable_testing ()
add_executable (blazingdb-communication-test EXCLUDE_FROM_ALL ${TEST_FILES})
target_include_directories (blazingdb-communication-test
  PUBLIC src
  PUBLIC ${GTEST_INCLUDE_DIRS}
  PUBLIC ${RAPIDJSON_INCLUDE_DIRS})
if (BLAZINGDB_COVERAGE)
target_compile_options (blazingdb-communication
  PUBLIC -g -O0 -fprofile-arcs -ftest-coverage -fPIC)
target_link_libraries (blazingdb-communication-test
  GTest::GTest gmock_main gmock blazingdb-communication gcov --coverage)
add_custom_target (coverage
  COMMAND
    ${CMAKE_BINARY_DIR}/src/blazingdb/communication/blazingdb-communication-test
  COMMAND lcov -c -d ${CMAKE_BINARY_DIR}/src -o coverage.info
  COMMAND lcov -r coverage.info '/usr*' '*boost*' '*build*' -o coverage.info
  COMMAND genhtml coverage.info -o ${CMAKE_BINARY_DIR}/coverage-html
)
add_dependencies (coverage blazingdb-communication-test)
else ()
target_link_libraries (blazingdb-communication-test
  GTest::GTest gmock_main gmock blazingdb-communication)
endif ()
gtest_discover_tests (blazingdb-communication-test)
endif ()
