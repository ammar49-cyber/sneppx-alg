if(ARIX_BUILD_PYTHON)
    find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 REQUIRED)

    pybind11_add_module(arix_algo_core bindings.cpp)
    target_include_directories(arix_algo_core PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/src/arch/include
        ${CMAKE_SOURCE_DIR}/src/security/c/include
        ${CMAKE_SOURCE_DIR}/src/security/cpp/include
    )
    target_link_libraries(arix_algo_core PRIVATE
        arix_core
        arix_arch
        pybind11::module
    )
    set_target_properties(arix_algo_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/python/arix_algo"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/python/arix_algo"
    )

    install(TARGETS arix_algo_core DESTINATION ${CMAKE_SOURCE_DIR}/src/python/arix_algo)
endif()
