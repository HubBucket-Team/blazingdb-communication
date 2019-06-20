#ifndef BLAZINGDB_UC_UTIL_MACROS_HPP_
#define BLAZINGDB_UC_UTIL_MACROS_HPP_

#define UC_DEPRECATED __attribute__((deprecated))
#define UC_INLINE inline __attribute__((always_inline))

#define UC_INTERFACE(Kind)                                                     \
public:                                                                        \
  virtual ~Kind() = default;                                                   \
                                                                               \
protected:                                                                     \
  explicit Kind() = default;                                                   \
                                                                               \
private:                                                                       \
  Kind(const Kind &)  = delete;                                                \
  Kind(const Kind &&) = delete;                                                \
  void operator=(const Kind &) = delete;                                       \
  void operator=(const Kind &&) = delete

#endif
