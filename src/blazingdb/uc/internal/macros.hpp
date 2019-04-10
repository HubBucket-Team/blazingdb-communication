#ifndef BLAZINGDB_UC_INTERNAL_MACROS_HPP_
#define BLAZINGDB_UC_INTERNAL_MACROS_HPP_

#include <iostream>

#define DOMAIN_ABORT(_message)                                                 \
  do {                                                                         \
    std::cerr << __FILE__ << ": " << __LINE__ << std::endl;                    \
    std::exit(-1);                                                             \
  } while (0)

#define CHECK_UCS(_expr)                                                       \
  do {                                                                         \
    ucs_status_t _status = (_expr);                                            \
    if (UCS_OK != (_status)) {                                                 \
      DOMAIN_ABORT(ucs_status_string(_status));                                \
    }                                                                          \
  } while (0)

#undef UCT_MEM_HANDLE_NULL
#define UCT_MEM_HANDLE_NULL nullptr

#undef UCT_INVALID_RKEY
#define UCT_INVALID_RKEY static_cast<std::uintptr_t>(-1)

#define UC_NO_EXPORT __attribute__((visibility("hidden")))

#define UC_CONCRETE(Kind)                                                      \
private:                                                                       \
  Kind(const Kind &) = delete;                                                 \
  Kind(const Kind &&) = delete;                                                \
  void operator=(const Kind &) = delete;                                       \
  void operator=(const Kind &&) = delete

#define UC_STATIC_LOCAL(Kind, name) static const Kind &name = *new Kind

#define UC_DTO(Kind)                                                           \
  UC_CONCRETE(Kind);                                                           \
                                                                               \
public:                                                                        \
  inline explicit Kind() = default;                                            \
                                                                               \
public:                                                                        \
  inline ~Kind() = default

#endif
