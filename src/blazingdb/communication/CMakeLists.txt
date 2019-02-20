add_library (blazingdb-communication
  Buffer.cc
  Message.cc
  MessageToken.cc
  Router.cc
  RouterBuilder.cc
  Server.cpp
)

set (PUBLIC_HEADER
  Buffer.h
  Message.h
  MessageToken.h
  Server.h
)

set_target_properties (blazingdb-communication
  PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADER}")

target_include_directories (blazingdb-communication
  PUBLIC ${PROJECT_SOURCE_DIR}/src
)

install (TARGETS blazingdb-communication
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)
