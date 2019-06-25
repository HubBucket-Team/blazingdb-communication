#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_COMMON_TESTHELPERS_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_COMMON_TESTHELPERS_HPP_

#include <gmock/gmock.h>

#define GMOCK_METHOD0_NE_(tn, constness, ct, Method, ...)                      \
  static_assert(                                                               \
      0 == ::testing::internal::Function<__VA_ARGS__>::ArgumentCount,          \
      "MOCK_METHOD<N> must match argument count.");                            \
  GMOCK_RESULT_(tn, __VA_ARGS__) ct Method() constness noexcept {              \
    GMOCK_MOCKER_(0, constness, Method).SetOwnerAndName(this, #Method);        \
    return GMOCK_MOCKER_(0, constness, Method).Invoke();                       \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method() constness {                \
    GMOCK_MOCKER_(0, constness, Method).RegisterOwner(this);                   \
    return GMOCK_MOCKER_(0, constness, Method).With();                         \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method(                             \
      const ::testing::internal::WithoutMatchers &,                            \
      constness ::testing::internal::Function<__VA_ARGS__> *) const {          \
    return ::testing::internal::AdjustConstness_##constness(this)              \
        ->gmock_##Method();                                                    \
  }                                                                            \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(                \
      0, constness, Method)

#define GMOCK_METHOD1_NE_(tn, constness, ct, Method, ...)                      \
  static_assert(                                                               \
      1 == ::testing::internal::Function<__VA_ARGS__>::ArgumentCount,          \
      "MOCK_METHOD<N> must match argument count.");                            \
  GMOCK_RESULT_(tn, __VA_ARGS__)                                               \
  ct Method(GMOCK_ARG_(tn, 1, __VA_ARGS__) gmock_a1) constness noexcept {      \
    GMOCK_MOCKER_(1, constness, Method).SetOwnerAndName(this, #Method);        \
    return GMOCK_MOCKER_(1, constness, Method)                                 \
        .Invoke(::std::forward<GMOCK_ARG_(tn, 1, __VA_ARGS__)>(gmock_a1));     \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method(                             \
      GMOCK_MATCHER_(tn, 1, __VA_ARGS__) gmock_a1) constness {                 \
    GMOCK_MOCKER_(1, constness, Method).RegisterOwner(this);                   \
    return GMOCK_MOCKER_(1, constness, Method).With(gmock_a1);                 \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method(                             \
      const ::testing::internal::WithoutMatchers &,                            \
      constness ::testing::internal::Function<__VA_ARGS__> *) const {          \
    return ::testing::internal::AdjustConstness_##constness(this)              \
        ->gmock_##Method(::testing::A<GMOCK_ARG_(tn, 1, __VA_ARGS__)>());      \
  }                                                                            \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(                \
      1, constness, Method)

#define GMOCK_METHOD2_NE_(tn, constness, ct, Method, ...)                      \
  static_assert(                                                               \
      2 == ::testing::internal::Function<__VA_ARGS__>::ArgumentCount,          \
      "MOCK_METHOD<N> must match argument count.");                            \
  GMOCK_RESULT_(tn, __VA_ARGS__)                                               \
  ct Method(GMOCK_ARG_(tn, 1, __VA_ARGS__) gmock_a1,                           \
            GMOCK_ARG_(tn, 2, __VA_ARGS__) gmock_a2) constness noexcept {      \
    GMOCK_MOCKER_(2, constness, Method).SetOwnerAndName(this, #Method);        \
    return GMOCK_MOCKER_(2, constness, Method)                                 \
        .Invoke(::std::forward<GMOCK_ARG_(tn, 1, __VA_ARGS__)>(gmock_a1),      \
                ::std::forward<GMOCK_ARG_(tn, 2, __VA_ARGS__)>(gmock_a2));     \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method(                             \
      GMOCK_MATCHER_(tn, 1, __VA_ARGS__) gmock_a1,                             \
      GMOCK_MATCHER_(tn, 2, __VA_ARGS__) gmock_a2) constness {                 \
    GMOCK_MOCKER_(2, constness, Method).RegisterOwner(this);                   \
    return GMOCK_MOCKER_(2, constness, Method).With(gmock_a1, gmock_a2);       \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method(                             \
      const ::testing::internal::WithoutMatchers &,                            \
      constness ::testing::internal::Function<__VA_ARGS__> *) const {          \
    return ::testing::internal::AdjustConstness_##constness(this)              \
        ->gmock_##Method(::testing::A<GMOCK_ARG_(tn, 1, __VA_ARGS__)>(),       \
                         ::testing::A<GMOCK_ARG_(tn, 2, __VA_ARGS__)>());      \
  }                                                                            \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(                \
      2, constness, Method)

#define GMOCK_METHOD3_NE_(tn, constness, ct, Method, ...)                      \
  static_assert(                                                               \
      3 == ::testing::internal::Function<__VA_ARGS__>::ArgumentCount,          \
      "MOCK_METHOD<N> must match argument count.");                            \
  GMOCK_RESULT_(tn, __VA_ARGS__)                                               \
  ct Method(GMOCK_ARG_(tn, 1, __VA_ARGS__) gmock_a1,                           \
            GMOCK_ARG_(tn, 2, __VA_ARGS__) gmock_a2,                           \
            GMOCK_ARG_(tn, 3, __VA_ARGS__) gmock_a3) constness noexcept {      \
    GMOCK_MOCKER_(3, constness, Method).SetOwnerAndName(this, #Method);        \
    return GMOCK_MOCKER_(3, constness, Method)                                 \
        .Invoke(::std::forward<GMOCK_ARG_(tn, 1, __VA_ARGS__)>(gmock_a1),      \
                ::std::forward<GMOCK_ARG_(tn, 2, __VA_ARGS__)>(gmock_a2),      \
                ::std::forward<GMOCK_ARG_(tn, 3, __VA_ARGS__)>(gmock_a3));     \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method(                             \
      GMOCK_MATCHER_(tn, 1, __VA_ARGS__) gmock_a1,                             \
      GMOCK_MATCHER_(tn, 2, __VA_ARGS__) gmock_a2,                             \
      GMOCK_MATCHER_(tn, 3, __VA_ARGS__) gmock_a3) constness {                 \
    GMOCK_MOCKER_(3, constness, Method).RegisterOwner(this);                   \
    return GMOCK_MOCKER_(3, constness, Method)                                 \
        .With(gmock_a1, gmock_a2, gmock_a3);                                   \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method(                             \
      const ::testing::internal::WithoutMatchers &,                            \
      constness ::testing::internal::Function<__VA_ARGS__> *) const {          \
    return ::testing::internal::AdjustConstness_##constness(this)              \
        ->gmock_##Method(::testing::A<GMOCK_ARG_(tn, 1, __VA_ARGS__)>(),       \
                         ::testing::A<GMOCK_ARG_(tn, 2, __VA_ARGS__)>(),       \
                         ::testing::A<GMOCK_ARG_(tn, 3, __VA_ARGS__)>());      \
  }                                                                            \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(                \
      3, constness, Method)

#define GMOCK_METHOD4_NE_(tn, constness, ct, Method, ...)                      \
  static_assert(                                                               \
      4 == ::testing::internal::Function<__VA_ARGS__>::ArgumentCount,          \
      "MOCK_METHOD<N> must match argument count.");                            \
  GMOCK_RESULT_(tn, __VA_ARGS__)                                               \
  ct Method(GMOCK_ARG_(tn, 1, __VA_ARGS__) gmock_a1,                           \
            GMOCK_ARG_(tn, 2, __VA_ARGS__) gmock_a2,                           \
            GMOCK_ARG_(tn, 3, __VA_ARGS__) gmock_a3,                           \
            GMOCK_ARG_(tn, 4, __VA_ARGS__) gmock_a4) constness noexcept {      \
    GMOCK_MOCKER_(4, constness, Method).SetOwnerAndName(this, #Method);        \
    return GMOCK_MOCKER_(4, constness, Method)                                 \
        .Invoke(::std::forward<GMOCK_ARG_(tn, 1, __VA_ARGS__)>(gmock_a1),      \
                ::std::forward<GMOCK_ARG_(tn, 2, __VA_ARGS__)>(gmock_a2),      \
                ::std::forward<GMOCK_ARG_(tn, 3, __VA_ARGS__)>(gmock_a3),      \
                ::std::forward<GMOCK_ARG_(tn, 4, __VA_ARGS__)>(gmock_a4));     \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method(                             \
      GMOCK_MATCHER_(tn, 1, __VA_ARGS__) gmock_a1,                             \
      GMOCK_MATCHER_(tn, 2, __VA_ARGS__) gmock_a2,                             \
      GMOCK_MATCHER_(tn, 3, __VA_ARGS__) gmock_a3,                             \
      GMOCK_MATCHER_(tn, 4, __VA_ARGS__) gmock_a4) constness {                 \
    GMOCK_MOCKER_(4, constness, Method).RegisterOwner(this);                   \
    return GMOCK_MOCKER_(4, constness, Method)                                 \
        .With(gmock_a1, gmock_a2, gmock_a3, gmock_a4);                         \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method(                             \
      const ::testing::internal::WithoutMatchers &,                            \
      constness ::testing::internal::Function<__VA_ARGS__> *) const {          \
    return ::testing::internal::AdjustConstness_##constness(this)              \
        ->gmock_##Method(::testing::A<GMOCK_ARG_(tn, 1, __VA_ARGS__)>(),       \
                         ::testing::A<GMOCK_ARG_(tn, 2, __VA_ARGS__)>(),       \
                         ::testing::A<GMOCK_ARG_(tn, 3, __VA_ARGS__)>(),       \
                         ::testing::A<GMOCK_ARG_(tn, 4, __VA_ARGS__)>());      \
  }                                                                            \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(                \
      4, constness, Method)

#define GMOCK_METHOD5_NE_(tn, constness, ct, Method, ...)                      \
  static_assert(                                                               \
      5 == ::testing::internal::Function<__VA_ARGS__>::ArgumentCount,          \
      "MOCK_METHOD<N> must match argument count.");                            \
  GMOCK_RESULT_(tn, __VA_ARGS__)                                               \
  ct Method(GMOCK_ARG_(tn, 1, __VA_ARGS__) gmock_a1,                           \
            GMOCK_ARG_(tn, 2, __VA_ARGS__) gmock_a2,                           \
            GMOCK_ARG_(tn, 3, __VA_ARGS__) gmock_a3,                           \
            GMOCK_ARG_(tn, 4, __VA_ARGS__) gmock_a4,                           \
            GMOCK_ARG_(tn, 5, __VA_ARGS__) gmock_a5) constness noexcept {      \
    GMOCK_MOCKER_(5, constness, Method).SetOwnerAndName(this, #Method);        \
    return GMOCK_MOCKER_(5, constness, Method)                                 \
        .Invoke(::std::forward<GMOCK_ARG_(tn, 1, __VA_ARGS__)>(gmock_a1),      \
                ::std::forward<GMOCK_ARG_(tn, 2, __VA_ARGS__)>(gmock_a2),      \
                ::std::forward<GMOCK_ARG_(tn, 3, __VA_ARGS__)>(gmock_a3),      \
                ::std::forward<GMOCK_ARG_(tn, 4, __VA_ARGS__)>(gmock_a4),      \
                ::std::forward<GMOCK_ARG_(tn, 5, __VA_ARGS__)>(gmock_a5));     \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method(                             \
      GMOCK_MATCHER_(tn, 1, __VA_ARGS__) gmock_a1,                             \
      GMOCK_MATCHER_(tn, 2, __VA_ARGS__) gmock_a2,                             \
      GMOCK_MATCHER_(tn, 3, __VA_ARGS__) gmock_a3,                             \
      GMOCK_MATCHER_(tn, 4, __VA_ARGS__) gmock_a4,                             \
      GMOCK_MATCHER_(tn, 5, __VA_ARGS__) gmock_a5) constness {                 \
    GMOCK_MOCKER_(5, constness, Method).RegisterOwner(this);                   \
    return GMOCK_MOCKER_(5, constness, Method)                                 \
        .With(gmock_a1, gmock_a2, gmock_a3, gmock_a4, gmock_a5);               \
  }                                                                            \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method(                             \
      const ::testing::internal::WithoutMatchers &,                            \
      constness ::testing::internal::Function<__VA_ARGS__> *) const {          \
    return ::testing::internal::AdjustConstness_##constness(this)              \
        ->gmock_##Method(::testing::A<GMOCK_ARG_(tn, 1, __VA_ARGS__)>(),       \
                         ::testing::A<GMOCK_ARG_(tn, 2, __VA_ARGS__)>(),       \
                         ::testing::A<GMOCK_ARG_(tn, 3, __VA_ARGS__)>(),       \
                         ::testing::A<GMOCK_ARG_(tn, 4, __VA_ARGS__)>(),       \
                         ::testing::A<GMOCK_ARG_(tn, 5, __VA_ARGS__)>());      \
  }                                                                            \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(                \
      5, constness, Method)

#define MOCK_CONST_METHOD0_NE(m, ...)                                          \
  GMOCK_METHOD0_NE_(, const, , m, __VA_ARGS__)
#define MOCK_CONST_METHOD1_NE(m, ...)                                          \
  GMOCK_METHOD1_NE_(, const, , m, __VA_ARGS__)
#define MOCK_CONST_METHOD2_NE(m, ...)                                          \
  GMOCK_METHOD2_NE_(, const, , m, __VA_ARGS__)
#define MOCK_CONST_METHOD3_NE(m, ...)                                          \
  GMOCK_METHOD3_NE_(, const, , m, __VA_ARGS__)
#define MOCK_CONST_METHOD4_NE(m, ...)                                          \
  GMOCK_METHOD4_NE_(, const, , m, __VA_ARGS__)
#define MOCK_CONST_METHOD5_NE(m, ...)                                          \
  GMOCK_METHOD5_NE_(, const, , m, __VA_ARGS__)

namespace testing {

ACTION_TEMPLATE(ReturnPointeeCast,
                HAS_1_TEMPLATE_PARAMS(class, T),
                AND_1_VALUE_PARAMS(pointer)) {
  return *std::shared_ptr<T>(pointer, reinterpret_cast<T *>(pointer.get()));
}

template <class Callable, class... Args>
std::result_of_t<Callable>
ReturnMake(Callable &&call, Args &&... args) {
  return std::forward<Callable>(call)(std::forward<Args>(args)...);
}

}  // namespace testing

#include <blazingdb/uc/API.hpp>

namespace uc {

class MockAgent : public blazingdb::uc::Agent {
public:
  using Buffer = blazingdb::uc::Buffer;

  MOCK_CONST_METHOD2_NE(Register,
                        std::unique_ptr<Buffer>(const void *&, std::size_t));
};

class MockBuffer : public blazingdb::uc::Buffer {
public:
  using Transport  = blazingdb::uc::Transport;
  using Serialized = blazingdb::uc::Record::Serialized;

  MOCK_CONST_METHOD1(Link, std::unique_ptr<Transport>(Buffer *));
  MOCK_CONST_METHOD0_NE(SerializedRecord, std::unique_ptr<const Serialized>());
  MOCK_METHOD1(Link, std::unique_ptr<Transport>(const std::uint8_t *));
};

class MockSerialized : public blazingdb::uc::Record::Serialized {
public:
  MOCK_CONST_METHOD0_NE(Data, const std::uint8_t *());
  MOCK_CONST_METHOD0_NE(Size, std::size_t());
};

class MockTransport : public blazingdb::uc::Transport {
public:
  MOCK_METHOD0(Get, std::future<void>());
};

}  // namespace uc

#endif
