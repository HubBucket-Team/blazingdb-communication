add_library (blazingdb-communication
  Address.cc
  Buffer.cc
  ContextToken.cc
  Context.cc
  Cluster.cc
  Manager.cc
  Message.cc
  MessageToken.cc
  network/Client.cc
  network/Server.cc
  Node.cc
  NodeToken.cc
  Router.cc
  RouterBuilder.cc
  messages/DataPivot.cpp
  messages/PartitionPivotsMessage.cpp
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
