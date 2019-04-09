if (NOT DEFINED UCX_INSTALL_DIR)
  message(FATAL_ERROR
    "- - - - - - - - - - - - - - - - -\n"
    "Don't forget -DUCX_INSTALL_DIR=/path/to/ucx\n"
    "- - - - - - - - - - - - - - - - -")
endif ()

add_library(UCX::UCX INTERFACE IMPORTED)
target_include_directories(UCX::UCX INTERFACE ${UCX_INSTALL_DIR}/include)
target_link_directories(UCX::UCX
  INTERFACE ${UCX_INSTALL_DIR}/lib INTERFACE /usr/local/cuda/lib64)
target_link_libraries(UCX::UCX INTERFACE uct ucs ucm cudart cuda)

add_library(blazingdb-uc
  Context.cpp
)
target_link_libraries(blazingdb-uc UCX::UCX)

set (BLAZINGBD_UC_PUBLIC_HEADERS
  Agent.hpp
  Buffer.hpp
  Context.hpp
  Manager.hpp
  Transport.hpp
)

find_package(GTest)

if (GTEST_FOUND)
  add_executable(blazingdb-uc-gtest
    context-test.cpp
  )
  target_link_libraries(blazingdb-uc-gtest blazingdb-uc GTest::Main)
endif ()
