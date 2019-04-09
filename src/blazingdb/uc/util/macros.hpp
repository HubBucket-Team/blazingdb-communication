#ifndef BLAZINGDB_UC_UTIL_MACROS_HPP_
#define BLAZINGDB_UC_UTIL_MACROS_HPP_

#define UC_INTERFACE(Kind)                                                     \
public:                                                                        \
  virtual ~Kind() = default;                                                   \
                                                                               \
protected:                                                                     \
  explicit Kind() = default;                                                   \
                                                                               \
private:                                                                       \
  Kind(const Kind &) = delete;                                                 \
  Kind(const Kind &&) = delete;                                                \
  void operator=(const Kind &) = delete;                                       \
  void operator=(const Kind &&) = delete

#endif
