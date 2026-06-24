add_executable(hss_demo hss_demo.c)
target_link_libraries(hss_demo arix_core arix_arch)
target_include_directories(hss_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/arch/include
)

add_executable(ser_demo ser_demo.c)
target_link_libraries(ser_demo arix_core arix_arch)
target_include_directories(ser_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/arch/include
)

add_executable(arc_demo arc_demo.c)
target_link_libraries(arc_demo arix_core arix_arch)
target_include_directories(arc_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/arch/include
)

add_executable(npe_demo npe_demo.c)
target_link_libraries(npe_demo arix_core arix_arch)
target_include_directories(npe_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/arch/include
)

add_executable(fm_demo fm_demo.c)
target_link_libraries(fm_demo arix_core arix_arch)
target_include_directories(fm_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/arch/include
)
