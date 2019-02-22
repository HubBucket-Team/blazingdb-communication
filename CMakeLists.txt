#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(blazingdb-communication C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQURIED ON)

message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

# TODO percy felipe alexander remove these lines when cudf stop using old C++ ABI
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
message(AUTHOR_WARNING "Using old C++ ABI: -D_GLIBCXX_USE_CXX11_ABI=0")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH:" "${CMAKE_MODULE_PATH}")

# Include CMake modules
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

# NOTE Do not change the inclusion order
# Configure dependencies
include(ConfigureBoost)
include(ConfigureRapidJSON)
include(ConfigureSimpleWebServer)

# BEGIN MAIN #
include_directories(
    ${LIB_BLAZINGIO_INCLUDEDIR}
    ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(src)

#target_link_libraries(blazingdb-communication
#    ${X_STATIC_LIB}
#)

# Tests
if(BUILD_TESTING)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        add_subdirectory(tests)

        enable_testing ()
        add_executable (blazingdb-communication-test EXCLUDE_FROM_ALL
          src/blazingdb/communication/buffer-test.cc
          src/blazingdb/communication/router-builder-test.cc
          src/blazingdb/communication/network/integration-server-client-test.cc
        )
        target_include_directories (blazingdb-communication-test
          PUBLIC src
          PUBLIC ${GTEST_INCLUDE_DIRS}
          PUBLIC ${RAPIDJSON_INCLUDE_DIRS})
        target_link_libraries (blazingdb-communication-test
          GTest::GTest gmock_main gmock blazingdb-communication)
        gtest_discover_tests(blazingdb-communication-test)
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif()
endif()

# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)

#END MAIN #
