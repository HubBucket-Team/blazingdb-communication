message(STATUS "================================================> GDR LIB: ${GDR_STATIC_LIB}")


blazingdb_artifact (
  VIRTUAL TARGET
    blazingdb-uc

  SOURCES
    Context.cpp
    ContextHelpers.cpp
    internal/agents/AddressableAgent.cpp
    internal/agents/ManagedAgent.cpp
    internal/buffers/LinkerBuffer.cpp
    internal/buffers/RemoteBuffer.cpp
    internal/buffers/records/RemotableRecord.cpp
    internal/buffers/transports/ZCopyTransport.cpp
    internal/ManagedContext.cpp

  INCLUDES
    PUBLIC /usr/local/cuda/include

  LIBRARIES
    cuda
    /usr/local/cuda/lib64/libcudart.so
    ${UCX_UCT_STATIC_LIB}
    ${GDR_STATIC_LIB}
    ${UCX_UCS_STATIC_LIB}
    ${UCX_UCM_STATIC_LIB}

  PREFIX
    blazingdb/uc

  PUBLIC_HEADERS
    Agent.hpp
    API.hpp
    Buffer.hpp
    Context.hpp
    Record.hpp
    Trader.hpp
    Transport.hpp
    util/macros.hpp

  TESTS
    api-common-test.cpp
    api-processes-test.cpp
    api-threads-test.cpp
    context-test.cpp
)
