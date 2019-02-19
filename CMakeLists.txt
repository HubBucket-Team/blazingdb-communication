cmake_minimum_required (VERSION 3.11)

project (BlazingDB-Communication)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_STANDARD 14)

find_package (GTest)

add_subdirectory (src)

if (GTEST_FOUND)
  enable_testing ()
  add_executable (blazingdb-communication-test
    src/blazingdb/communication/buffer-test.cc
  )
  target_link_libraries (blazingdb-communication-test GTest::GTest GTest::Main)
  gtest_discover_tests(blazingdb-communication-test)
endif ()

