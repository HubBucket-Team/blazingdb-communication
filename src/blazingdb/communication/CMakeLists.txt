find_package(Boost REQUIRED COMPONENTS system)

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

  TESTS
    manager-test.cc
    node-test.cc
    network/integration-server-client-test.cc
)

install (TARGETS blazingdb-communication
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install (DIRECTORY "${PROJECT_SOURCE_DIR}/src/blazingdb" DESTINATION "include"
  FILES_MATCHING PATTERN "*.h" PATTERN "*Internal.h" EXCLUDE
)
