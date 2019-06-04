find_package(GTest)

if (GTEST_FOUND)
  enable_testing ()
  add_custom_target (coverage
    COMMAND lcov -c -d ${CMAKE_BINARY_DIR}/src -o coverage.info
    COMMAND lcov -r coverage.info '/usr*' '*boost*' '*build*' -o coverage.info
    COMMAND genhtml coverage.info -o ${CMAKE_BINARY_DIR}/coverage-html)
endif ()

function (blazingdb_artifact)
  set (OPTIONS OPTIONAL VIRTUAL)
  set (ONE_VALUE_ARGS PREFIX TARGET)
  set (MULTI_VALUE_ARGS SOURCES INCLUDES LIBRARIES PUBLIC_HEADERS TESTS
       LIBRARY_PATHS)
  cmake_parse_arguments (ARG "${OPTIONS}" "${ONE_VALUE_ARGS}"
    "${MULTI_VALUE_ARGS}" ${ARGN})
  if (ARG_UNPARSED_ARGUMENTS)
    message (SEND_ERROR
      "Error: unrecognized arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif ()

  if (${ARG_VIRTUAL})
    set (TYPE OBJECT)
  endif ()

  add_library (${ARG_TARGET} ${TYPE} ${ARG_SOURCES})

  if (ARG_LIBRARIES)
    target_link_libraries (${ARG_TARGET} ${ARG_LIBRARIES})
  endif ()

  if (ARG_INCLUDES)
    target_include_directories (${ARG_TARGET} ${ARG_INCLUDES})
  endif ()

  if (ARG_LIBRARY_PATHS)
    target_link_libraries (${ARG_TARGET} ${ARG_LIBRARY_PATHS})
  endif ()

  if (NOT ARG_PREFIX)
    set (ARG_PREFIX "")
  endif ()

  install (TARGETS ${ARG_TARGET}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

  if (ARG_PUBLIC_HEADERS)
    foreach (ARG_PUBLIC_HEADER ${ARG_PUBLIC_HEADERS})
      get_filename_component (ARG_BASE ${ARG_PUBLIC_HEADER} DIRECTORY)
      install (FILES ${ARG_PUBLIC_HEADER}
        DESTINATION include/${ARG_PREFIX}/${ARG_BASE})
    endforeach ()
  endif ()

  if (GTEST_FOUND AND ARG_TESTS)
    add_executable (${ARG_TARGET}-gtest EXCLUDE_FROM_ALL ${ARG_TESTS})
    target_include_directories (${ARG_TARGET} PUBLIC src)
    target_compile_options (${ARG_TARGET} PUBLIC -ggdb -O0)
    target_link_libraries (${ARG_TARGET}-gtest ${ARG_TARGET}
      GTest::GTest
      ${GTEST_ROOT}/lib/libgmock_main.a
      ${GTEST_ROOT}/lib/libgmock.a)
    add_custom_target (${ARG_TARGET}-gtest-run
      COMMAND $<TARGET_FILE:${ARG_TARGET}-gtest>)
    add_dependencies (coverage ${ARG_TARGET}-gtest-run)
  endif ()
endfunction ()

add_subdirectory (uc)
add_subdirectory (communication)
