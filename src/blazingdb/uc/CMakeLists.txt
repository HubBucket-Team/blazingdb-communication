message(STATUS
  "================================================> GDR LIB:"
  " ${GDR_STATIC_LIB}")

blazingdb_artifact (
  VIRTUAL TARGET
    blazingdb-uc

  SOURCES
    Capabilities.cpp
    Context.cpp
    ContextHelpers.cpp
    internal/agents/AddressableAgent.cpp
    internal/agents/ManagedAgent.cpp
    internal/agents/ViewAgent.cpp
    internal/buffers/LinkerBuffer.cpp
    internal/buffers/RemoteBuffer.cpp
    internal/buffers/ViewBuffer.cpp
    internal/buffers/records/RemotableRecord.cpp
    internal/buffers/transports/ZCopyTransport.cpp
    internal/ManagedContext.cpp
    internal/ViewContext.cpp

    internal/tcp/TCPAgent.cpp
    internal/tcp/TCPContext.cpp
    internal/tcp/TCPTransport.cpp

  INCLUDES
    PUBLIC /usr/local/cuda/include

  LIBRARIES
    /usr/local/cuda/lib64/libcudart.so
    cuda
    ${UCX_UCT_STATIC_LIB}
    ${GDR_STATIC_LIB}
    ${UCX_UCS_STATIC_LIB}
    ${UCX_UCM_STATIC_LIB}
    cuda
    /usr/local/cuda/lib64/libcudart.so

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
    Capabilities.hpp

  TESTS
    api-common-test.cpp
    api-processes-test.cpp
    api-threads-test.cpp
    context-test.cpp
    internal/tcp/tcp-context-test.cpp
    context-stress-test.cpp
)
