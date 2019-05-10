blazingdb_artifact (
  TARGET
    blazingdb-communication

  SOURCES
    Address.cc
    Address-Internal.cc
    ContextToken.cc
    Context.cc
    Cluster.cc
    internal/Trader.cpp
    internal/TraderLock.cpp
    Manager.cc
    network/Client.cc
    network/MessageQueue.cpp
    network/Server.cc
    Node.cc
    Router.cc
    messages/Message.cc
    messages/MessageToken.cc
    messages/DataPivot.cpp
    messages/PartitionPivotsMessage.cpp
    messages/NodeDataMessage.cc

  INCLUDES
    PUBLIC ${PROJECT_SOURCE_DIR}/src

  LIBRARIES
    Boost::system
    blazingdb-uc

  PREFIX
    blazingdb/communication

  PUBLIC_HEADERS
    messages/SampleToNodeMasterMessage.h
    messages/Message.h
    messages/GpuComponentMessage.h
    messages/DataScatterMessage.h
    messages/DataPivot.h
    messages/MessageToken.h
    messages/BaseComponentMessage.h
    messages/ColumnDataMessage.h
    messages/NodeDataMessage.h
    messages/PartitionPivotsMessage.h
    messages/UCPool.h
    ContextToken.h
    Cluster.h
    Address.h
    shared/Identity.h
    shared/macros.h
    shared/Entity.h
    shared/Builder.h
    shared/JsonSerializable.h
    Manager.h
    Context.h
    Buffer.h
    network/Client.h
    network/MessageQueue.h
    network/ClientExceptions.h
    network/Server.h
    network/Status.h
    Router.h
    Listener.h
    Node.h

  TESTS
    manager-test.cc
    messages/message-test.cpp
    node-test.cc
    network/integration-server-client-test.cc
    processes-test.cc
)
