if (NOT DEFINED UCX_INSTALL_DIR)
  message (FATAL_ERROR
    "- - - - - - - - - - - - - - - - -\n"
    "Don't forget -DUCX_INSTALL_DIR=/path/to/ucx\n"
    "- - - - - - - - - - - - - - - - -")
endif ()

add_library (UCX::UCX INTERFACE IMPORTED)
target_include_directories (UCX::UCX
  INTERFACE ${UCX_INSTALL_DIR}/include INTERFACE /usr/local/cuda/include)
target_link_directories (UCX::UCX
  INTERFACE ${UCX_INSTALL_DIR}/lib INTERFACE /usr/local/cuda/lib64)
target_link_libraries (UCX::UCX INTERFACE uct ucs ucm cudart cuda)

blazingdb_artifact (
  TARGET
    blazingdb-uc

  SOURCES
    Context.cpp
    internal/buffers/RemoteBuffer.cpp

  LIBRARIES
    UCX::UCX

  PUBLIC_HEADERS
    Agent.hpp
    Buffer.hpp
    Context.hpp
    Manager.hpp
    Transport.hpp

  TESTS
    api-common-test.cpp
    api-processes-test.cpp
    api-threads-test.cpp
    context-test.cpp
)
