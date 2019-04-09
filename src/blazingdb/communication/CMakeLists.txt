add_library (blazingdb-communication
  Address.cc
  ContextToken.cc
  Context.cc
  Cluster.cc
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
)

target_include_directories (blazingdb-communication
  PUBLIC ${PROJECT_SOURCE_DIR}/src
)

find_package(Boost REQUIRED COMPONENTS system)

target_link_libraries(blazingdb-communication Boost::system)

install (TARGETS blazingdb-communication
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install (DIRECTORY "${PROJECT_SOURCE_DIR}/src/blazingdb" DESTINATION "include"
  FILES_MATCHING PATTERN "*.h" PATTERN "*Internal.h" EXCLUDE
)

blazingdb_gtest(blazingdb-communication
  manager-test.cc
  node-test.cc
  network/integration-server-client-test.cc
)
