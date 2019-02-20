cmake_minimum_required (VERSION 3.11)

project (BlazingDB-Communication)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_STANDARD 14)

find_package (GTest)

add_subdirectory (src)

if (GTEST_FOUND)
  enable_testing ()
  add_executable (blazingdb-communication-test EXCLUDE_FROM_ALL
    src/blazingdb/communication/buffer-test.cc
    src/blazingdb/communication/router-builder-test.cc
  )
  target_include_directories (blazingdb-communication-test PUBLIC src)
  target_link_libraries (blazingdb-communication-test GTest::GTest GTest::Main
    blazingdb-communication)
  gtest_discover_tests(blazingdb-communication-test)
endif ()

