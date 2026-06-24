file(GLOB_RECURSE TEST_SRCS ../tests/unit/*.c)
file(GLOB_RECURSE INTEGRATION_SRCS ../tests/integration/*.c)

foreach(test_src ${TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} arix_core arix_arch)
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/src/arch/include
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

foreach(test_src ${INTEGRATION_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} arix_core arix_arch)
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/src/arch/include
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
