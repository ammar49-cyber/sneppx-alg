file(GLOB_RECURSE TEST_SRCS ../tests/unit/*.c)
file(GLOB_RECURSE INTEGRATION_SRCS ../tests/integration/*.c)
file(GLOB_RECURSE SECURITY_SRCS ../tests/security/*.c)

macro(add_arix_test test_src link_libs)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} ${link_libs})
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/src/arch/include
        ${CMAKE_SOURCE_DIR}/src/security/c/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

foreach(test_src ${TEST_SRCS})
    add_arix_test(${test_src} "arix_core;arix_arch")
endforeach()

foreach(test_src ${INTEGRATION_SRCS})
    add_arix_test(${test_src} "arix_core;arix_arch")
endforeach()

foreach(test_src ${SECURITY_SRCS})
    add_arix_test(${test_src} "arix_security_c;arix_core;arix_arch")
endforeach()

file(GLOB_RECURSE SECURITY_CPP_SRCS ../tests/security/cpp/*.cpp)

macro(add_arix_cpp_test test_src link_libs)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} ${link_libs})
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/src/arch/include
        ${CMAKE_SOURCE_DIR}/src/security/c/include
        ${CMAKE_SOURCE_DIR}/src/security/cpp/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

foreach(test_src ${SECURITY_CPP_SRCS})
    add_arix_cpp_test(${test_src} "arix_security_cpp;arix_security_c;arix_core;arix_arch")
endforeach()

# --- Python tests ---
if(ARIX_BUILD_PYTHON)
    file(GLOB PYTHON_TEST_SRCS ../tests/python/*.py)
    foreach(test_src ${PYTHON_TEST_SRCS})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_test(NAME py_${test_name}
            COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=${CMAKE_BINARY_DIR}/src/python/$<CONFIG>
                ${Python3_EXECUTABLE} ${test_src}
        )
    endforeach()
endif()
