find_package(GTest)

if (GTEST_FOUND)
  enable_testing ()
  add_custom_target (coverage
    COMMAND lcov -c -d ${CMAKE_BINARY_DIR}/src -o coverage.info
    COMMAND lcov -r coverage.info '/usr*' '*boost*' '*build*' -o coverage.info
    COMMAND genhtml coverage.info -o ${CMAKE_BINARY_DIR}/coverage-html)
endif ()

function (blazingdb_gtest LIBRARY)
  if (GTEST_FOUND)
    set (TARGET ${LIBRARY}-gtest)
    add_executable (${TARGET} EXCLUDE_FROM_ALL ${ARGN})
    target_include_directories (${TARGET} PUBLIC src)
    target_compile_options (${TARGET}
      PUBLIC -g -O0 -fprofile-arcs -ftest-coverage -fPIC)
    target_link_libraries (${TARGET} ${LIBRARY}
      GTest::GTest GTest::Main gmock gmock_main gcov --coverage)
    add_custom_target (${TARGET}-run COMMAND $<TARGET_FILE:${TARGET}>)
    add_dependencies (coverage ${TARGET}-run)
  endif ()
endfunction ()

add_subdirectory (uc)
add_subdirectory (communication)
