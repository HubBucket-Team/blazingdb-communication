#ifndef BLAZINGDB_UC_INTERNAL_MACROS_HPP_
#define BLAZINGDB_UC_INTERNAL_MACROS_HPP_

#include <iostream>
#include <sstream>

namespace gsl {
template <class T, class = std::enable_if_t<std::is_pointer<T>::value>>
using owner = T;  // think about use a uniquer_ptr or shared_ptr when apply this
}  // namespace gsl

#define UC_ABORT(_message)                                                     \
  do {                                                                         \
    std::stringstream ss{std::ios_base::out | std::ios_base::in};              \
    ss << __FILE__ << ':' << __LINE__ << ": " << (_message) << std::endl;      \
    std::cerr << ss.str();                                                     \
    std::exit(-1);                                                             \
  } while (0)

#define CHECK_UCS(_expr)                                                       \
  do {                                                                         \
    ucs_status_t _status = (_expr);                                            \
    if (UCS_OK != (_status)) { UC_ABORT(ucs_status_string(_status)); }         \
  } while (0)

#define UC_MEM_HANDLE_NULL nullptr
#define UC_INVALID_RKEY static_cast<std::uintptr_t>(-1)
#define UC_BIT(i) (1UL << (i))

#define UC_CONST const __attribute__((__const__))
#ifndef UC_INLINE
#define UC_INLINE inline __attribute__((__always_inline__))
#endif
#define UC_NOEXPORT __attribute__((visibility("internal")))
#define UC_NORETURN __attribute__((__noreturn__))
#define UC_PURE __attribute__((__pure__))

#define uc_likely(x) __builtin_expect(x, 1)
#define uc_unlikely(x) __builtin_expect(x, 0)

#define UC_CONCRETE(Kind)                                                      \
private:                                                                       \
  Kind(const Kind &)  = delete;                                                \
  Kind(const Kind &&) = delete;                                                \
  void operator=(const Kind &) = delete;                                       \
  void operator=(const Kind &&) = delete

#define UC_STATIC_LOCAL(Kind, name) static const Kind &name = *new Kind

#define UC_DTO(Kind)                                                           \
  UC_CONCRETE(Kind);                                                           \
                                                                               \
public:                                                                        \
  inline explicit Kind() = default;                                            \
  inline ~Kind()         = default

#define UC_MIXIN(Kind)                                                         \
  UC_CONCRETE(Kind);                                                           \
                                                                               \
protected:                                                                     \
  inline explicit Kind() = default;                                            \
                                                                               \
public:                                                                        \
  inline ~Kind() = default

#endif
