if (NOT DEFINED UCX_INSTALL_DIR)
  message (FATAL_ERROR
    "- - - - - - - - - - - - - - - - -\n"
    "Don't forget -DUCX_INSTALL_DIR=/path/to/ucx\n"
    "- - - - - - - - - - - - - - - - -")
endif ()

add_library (UCX::UCX INTERFACE IMPORTED)

target_include_directories (UCX::UCX
  INTERFACE ${UCX_INSTALL_DIR}/include INTERFACE /usr/local/cuda/include)
set_target_properties(UCX::UCX PROPERTIES INTERFACE_LINK_LIBRARIES
  "${UCX_INSTALL_DIR}/lib;/usr/local/cuda/lib64")
target_link_libraries (UCX::UCX INTERFACE uct ucs ucm cudart cuda)

blazingdb_artifact (
  TARGET
    blazingdb-uc

  SOURCES
    Context.cpp
    internal/agents/AddressableAgent.cpp
    internal/agents/ManagedAgent.cpp
    internal/buffers/RemoteBuffer.cpp
    internal/buffers/records/RemotableRecord.cpp
    internal/buffers/transports/ZCopyTransport.cpp
    internal/ManagedContext.cpp

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
