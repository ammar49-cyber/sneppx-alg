file(GLOB_RECURSE UNIT_TEST_SRCS ../tests/unit/*.c)
file(GLOB_RECURSE INTEGRATION_TEST_SRCS ../tests/integration/*.c)
file(GLOB_RECURSE SECURITY_TEST_SRCS ../tests/security/*.c)

# Tests excluded because they reference APIs that were never implemented
set(EXCLUDED_TESTS
)

macro(add_neural_test test_src link_libs)
    get_filename_component(test_name ${test_src} NAME_WE)
    list(FIND EXCLUDED_TESTS ${test_name} _excluded)
    if(NOT _excluded EQUAL -1)
        message(STATUS "Skipping excluded test: ${test_name}")
    else()
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} ${link_libs})
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
        ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
        ${CMAKE_SOURCE_DIR}/include/neural_core/security
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/lib/internal
        ${CMAKE_SOURCE_DIR}/drivers/cuda
        ${CMAKE_SOURCE_DIR}/drivers/tpu
        ${CMAKE_SOURCE_DIR}/drivers/rocm
        ${CMAKE_SOURCE_DIR}/fs/format
        ${CMAKE_SOURCE_DIR}/security/monitor
        ${CMAKE_SOURCE_DIR}/security/memory
        ${CMAKE_SOURCE_DIR}/kernel
        ${CMAKE_SOURCE_DIR}/security/crypto/c
        ${CMAKE_SOURCE_DIR}/security/network
        ${CMAKE_SOURCE_DIR}/security/updates
        ${CMAKE_SOURCE_DIR}/security/formal
        ${CMAKE_SOURCE_DIR}/security/pentest
        ${CMAKE_SOURCE_DIR}/security/ai
        ${CMAKE_SOURCE_DIR}/security/ui
        ${CMAKE_SOURCE_DIR}/drivers/vulkan
        ${CMAKE_SOURCE_DIR}/drivers/tpu
        ${CMAKE_SOURCE_DIR}/drivers/http
        ${CMAKE_SOURCE_DIR}/drivers/zk
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endmacro()

foreach(test_src ${UNIT_TEST_SRCS})
    add_neural_test(${test_src} "neural_core_kernel;neural_architecture_layer")
endforeach()

foreach(test_src ${INTEGRATION_TEST_SRCS})
    add_neural_test(${test_src} "neural_core_kernel;neural_architecture_layer")
endforeach()

foreach(test_src ${SECURITY_TEST_SRCS})
    add_neural_test(${test_src} "neural_security_c;neural_core_kernel;neural_architecture_layer")
endforeach()

file(GLOB_RECURSE SECURITY_CPP_TEST_SRCS ../tests/security/cpp/*.cpp)

macro(add_neural_cpp_test test_src link_libs)
    get_filename_component(test_name ${test_src} NAME_WE)
    list(FIND EXCLUDED_TESTS ${test_name} _excluded)
    if(NOT _excluded EQUAL -1)
        message(STATUS "Skipping excluded test: ${test_name}")
    else()
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} ${link_libs})
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
        ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
        ${CMAKE_SOURCE_DIR}/include/neural_core/security
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/security/obfuscation
        ${CMAKE_SOURCE_DIR}/security/cpp/include
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endmacro()

foreach(test_src ${SECURITY_CPP_TEST_SRCS})
    add_neural_cpp_test(${test_src} "neural_security_cpp;neural_security_c;neural_core_kernel;neural_architecture_layer")
endforeach()

if(SNEPPX_BUILD_PYTHON)
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
