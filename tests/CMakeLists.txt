file(GLOB_RECURSE UNIT_TEST_SRCS ../tests/unit/*.c)
file(GLOB_RECURSE INTEGRATION_TEST_SRCS ../tests/integration/*.c)
file(GLOB_RECURSE SECURITY_TEST_SRCS ../tests/security/*.c)

macro(add_neural_test test_src link_libs)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} ${link_libs})
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
        ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
        ${CMAKE_SOURCE_DIR}/include/neural_core/security
        ${CMAKE_SOURCE_DIR}/tests
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
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
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} ${link_libs})
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
        ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
        ${CMAKE_SOURCE_DIR}/include/neural_core/security
        ${CMAKE_SOURCE_DIR}/tests
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

foreach(test_src ${SECURITY_CPP_TEST_SRCS})
    add_neural_cpp_test(${test_src} "neural_security_cpp;neural_security_c;neural_core_kernel;neural_architecture_layer")
endforeach()

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
