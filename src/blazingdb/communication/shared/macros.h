#ifndef BLAZINGDB_COMMUNICATION_SHARED_MACROS_H_
#define BLAZINGDB_COMMUNICATION_SHARED_MACROS_H_

#ifndef BLAZINGDB_TURN_COPYASSIGN_OFF
#define BLAZINGDB_TURN_COPYASSIGN_OFF(Kind) \
  Kind(const Kind&) = delete;    \
  void operator=(const Kind&) = delete
#endif

#endif
