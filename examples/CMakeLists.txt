add_executable(hss_demo hss_demo.c)
target_link_libraries(hss_demo arix_core arix_arch)
target_include_directories(hss_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/arix/core
    ${CMAKE_SOURCE_DIR}/include/arix/arch
)

add_executable(ser_demo ser_demo.c)
target_link_libraries(ser_demo arix_core arix_arch)
target_include_directories(ser_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/arix/core
    ${CMAKE_SOURCE_DIR}/include/arix/arch
)

add_executable(arc_demo arc_demo.c)
target_link_libraries(arc_demo arix_core arix_arch)
target_include_directories(arc_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/arix/core
    ${CMAKE_SOURCE_DIR}/include/arix/arch
)

add_executable(npe_demo npe_demo.c)
target_link_libraries(npe_demo arix_core arix_arch)
target_include_directories(npe_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/arix/core
    ${CMAKE_SOURCE_DIR}/include/arix/arch
)

add_executable(fm_demo fm_demo.c)
target_link_libraries(fm_demo arix_core arix_arch)
target_include_directories(fm_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/arix/core
    ${CMAKE_SOURCE_DIR}/include/arix/arch
)

add_executable(security_stress security_stress.c)
target_link_libraries(security_stress arix_security_c arix_core arix_arch)
target_include_directories(security_stress PRIVATE
    ${CMAKE_SOURCE_DIR}/include/arix/security
    ${CMAKE_SOURCE_DIR}/include/arix/core
    ${CMAKE_SOURCE_DIR}/include/arix/arch
)

add_executable(obf_demo obf_demo.cpp)
target_link_libraries(obf_demo arix_security_cpp arix_security_c arix_core arix_arch)
target_include_directories(obf_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/arix/security
    ${CMAKE_SOURCE_DIR}/include/arix/security
    ${CMAKE_SOURCE_DIR}/include/arix/core
    ${CMAKE_SOURCE_DIR}/include/arix/arch
)
