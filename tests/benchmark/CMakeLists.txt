file(GLOB BENCH_SRCS *.c)

macro(add_arix_bench bench_src)
    get_filename_component(bench_name ${bench_src} NAME_WE)
    add_executable(${bench_name} ${bench_src})
    target_link_libraries(${bench_name} arix_core arix_arch)
    target_include_directories(${bench_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/src/arch/include
        ${CMAKE_SOURCE_DIR}/src/security/c/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    if(ARIX_USE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
        arix_enable_lto(${bench_name})
    endif()
endmacro()

foreach(bench_src ${BENCH_SRCS})
    add_arix_bench(${bench_src})
endforeach()
