cmake_minimum_required(VERSION 3.16)
project(cognitive_processing_system VERSION 0.7.8 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ARIX_BUILD_TESTS "Build test suite" ON)
option(ARIX_BUILD_BENCHMARKS "Build benchmarks" ON)
option(ARIX_BUILD_PYTHON "Build Python bindings" OFF)
option(ARIX_BUILD_CUDA "Build CUDA kernels" OFF)
option(ARIX_BUILD_TOOLS "Build CLI tools" OFF)
option(ARIX_USE_ASAN "Enable AddressSanitizer (GCC/Clang only)" OFF)
option(ARIX_USE_UBSAN "Enable UndefinedBehaviorSanitizer (GCC/Clang only)" OFF)
option(ARIX_USE_LTO "Enable Link-Time Optimization (Release only)" OFF)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    if(ARIX_USE_ASAN)
        set(CMAKE_C_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address")
    else()
        set(CMAKE_C_FLAGS_DEBUG "-g")
        set(CMAKE_CXX_FLAGS_DEBUG "-g")
    endif()
    if(ARIX_USE_UBSAN)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=undefined -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow")
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=undefined")
    endif()
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes")
    endif()
elseif(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4 /DARIX_CRYPTO_SECURE /D_CRT_SECURE_NO_WARNINGS /FS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /DARIX_CRYPTO_SECURE /D_CRT_SECURE_NO_WARNINGS /FS")
    set(CMAKE_C_FLAGS_DEBUG "/Zi /RTC1")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /RTC1")
    if(ARIX_USE_ASAN)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /fsanitize=address")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /fsanitize=address")
    endif()
    set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "/O2 /Zi")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Zi")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerOptions)
include(Sanitizers)
include(StandardSettings)
include(StaticAnalyzers)
include(CodeCoverage)

if(ARIX_USE_LTO)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -flto")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto")
    elseif(MSVC)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /GL")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL")
        set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "/LTCG")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG")
    endif()
endif()

file(GLOB_RECURSE CORE_KERNEL_SRC kernel/*.c)
file(GLOB_RECURSE CORE_SECURITY_C_SRC security/crypto/c/*.c)

add_library(neural_core_kernel STATIC ${CORE_KERNEL_SRC})
target_include_directories(neural_core_kernel PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/kernel>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/security>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/architecture>
    $<INSTALL_INTERFACE:include/neural_core/kernel>
    $<INSTALL_INTERFACE:include/neural_core/security>
    $<INSTALL_INTERFACE:include/neural_core/architecture>
)

file(GLOB_RECURSE NEURAL_ARCHITECTURE_SRC algorithms/*/core/*.c)

add_library(neural_architecture_layer STATIC ${NEURAL_ARCHITECTURE_SRC})
target_include_directories(neural_architecture_layer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/architecture>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/kernel>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/security>
    $<INSTALL_INTERFACE:include/neural_core/architecture>
    $<INSTALL_INTERFACE:include/neural_core/kernel>
    $<INSTALL_INTERFACE:include/neural_core/security>
)
target_link_libraries(neural_architecture_layer PUBLIC neural_core_kernel)

if(CORE_SECURITY_C_SRC)
    add_library(neural_security_c STATIC ${CORE_SECURITY_C_SRC})
    target_include_directories(neural_security_c PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/security>
        $<INSTALL_INTERFACE:include/neural_core/security>
    )
    target_compile_definitions(neural_security_c PRIVATE $<$<CONFIG:Release>:ARIX_S1_SECURE_MEMORY>)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
        file(GLOB ARIX_ASM_X86_64 security/crypto/asm/x86_64/*.asm security/crypto/asm/x86_64/*.S)
        if(ARIX_ASM_X86_64)
            set_source_files_properties(${ARIX_ASM_X86_64} PROPERTIES LANGUAGE ASM)
            target_sources(neural_security_c PRIVATE ${ARIX_ASM_X86_64})
        endif()
    endif()

    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_libraries(neural_security_c PRIVATE rt)
    endif()
endif()

file(GLOB_RECURSE NEURAL_SECURITY_CPP_SRC security/obfuscation/*.cpp)

if(NEURAL_SECURITY_CPP_SRC)
    add_library(neural_security_cpp STATIC ${NEURAL_SECURITY_CPP_SRC})
    target_include_directories(neural_security_cpp PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/security>
        $<INSTALL_INTERFACE:include/neural_core/security>
    )
    target_link_libraries(neural_security_cpp PUBLIC neural_security_c neural_core_kernel neural_architecture_layer)
    target_compile_definitions(neural_security_cpp PRIVATE ARIX_S2_OBFUSCATION)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_libraries(neural_security_cpp PRIVATE dl)
    endif()
endif()

if(ARIX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(examples)
endif()

if(ARIX_BUILD_BENCHMARKS)
    add_subdirectory(tests/benchmark)
endif()

if(ARIX_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()

if(ARIX_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)

    file(GLOB_RECURSE ARIX_CUDA_KERNEL_SRC algorithms/*/cuda/*.cu)
    if(ARIX_CUDA_KERNEL_SRC)
        add_library(neural_cuda_kernels STATIC ${ARIX_CUDA_KERNEL_SRC})
        target_include_directories(neural_cuda_kernels PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/architecture>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/kernel>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/security>
            $<INSTALL_INTERFACE:include/neural_core/architecture>
            $<INSTALL_INTERFACE:include/neural_core/kernel>
            $<INSTALL_INTERFACE:include/neural_core/security>
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_compile_definitions(neural_cuda_kernels PRIVATE ARIX_HAS_CUDA)
        target_link_libraries(neural_cuda_kernels PUBLIC CUDA::cudart CUDA::cublas)
    endif()

    target_compile_definitions(neural_core_kernel PRIVATE ARIX_HAS_CUDA)
    target_compile_definitions(neural_architecture_layer PRIVATE ARIX_HAS_CUDA)
    target_link_libraries(neural_core_kernel PUBLIC CUDA::cudart CUDA::cublas)
endif()

file(GLOB_RECURSE ARIX_MONITOR_SRC security/monitor/*.c)
if(ARIX_MONITOR_SRC)
    target_sources(neural_security_c PRIVATE ${ARIX_MONITOR_SRC})
endif()

file(GLOB_RECURSE ARIX_SECURITY_MEMORY_SRC security/memory/*.c)
if(ARIX_SECURITY_MEMORY_SRC)
    target_sources(neural_security_c PRIVATE ${ARIX_SECURITY_MEMORY_SRC})
    target_compile_definitions(neural_security_c PRIVATE ARIX_S1_SECURE_ALLOCATOR)
endif()

file(GLOB_RECURSE ARIX_C_BINDING_SRC security/c/src/*.c)
if(ARIX_C_BINDING_SRC)
    target_sources(neural_security_c PRIVATE ${ARIX_C_BINDING_SRC})
endif()

file(GLOB_RECURSE ARIX_S4_NETWORK_SRC security/network/*.c)
if(ARIX_S4_NETWORK_SRC)
    target_sources(neural_security_c PRIVATE ${ARIX_S4_NETWORK_SRC})
    target_compile_definitions(neural_security_c PRIVATE ARIX_S4_NETWORK_SECURITY)
endif()

file(GLOB_RECURSE ARIX_S5_AI_SRC security/ai/*.c)
if(ARIX_S5_AI_SRC)
    target_sources(neural_security_c PRIVATE ${ARIX_S5_AI_SRC})
    target_compile_definitions(neural_security_c PRIVATE ARIX_S5_AI_SANITIZER)
endif()

file(GLOB_RECURSE ARIX_S6_UI_SRC security/ui/*.c)
if(ARIX_S6_UI_SRC)
    target_sources(neural_security_c PRIVATE ${ARIX_S6_UI_SRC})
    target_compile_definitions(neural_security_c PRIVATE ARIX_S6_SECURITY_UI)
endif()

file(GLOB_RECURSE ARIX_S7_UPDATE_SRC security/updates/*.c)
if(ARIX_S7_UPDATE_SRC)
    target_sources(neural_security_c PRIVATE ${ARIX_S7_UPDATE_SRC})
    target_compile_definitions(neural_security_c PRIVATE ARIX_S7_SECURE_UPDATES)
endif()

file(GLOB_RECURSE ARIX_S8_FORMAL_SRC security/formal/*.c)
if(ARIX_S8_FORMAL_SRC)
    target_sources(neural_security_c PRIVATE ${ARIX_S8_FORMAL_SRC})
    target_compile_definitions(neural_security_c PRIVATE ARIX_S8_FORMAL_VERIFICATION)
endif()

file(GLOB_RECURSE ARIX_S9_PENTEST_SRC security/pentest/*.c)
if(ARIX_S9_PENTEST_SRC)
    target_sources(neural_security_c PRIVATE ${ARIX_S9_PENTEST_SRC})
    target_compile_definitions(neural_security_c PRIVATE ARIX_S9_PENETRATION_TESTING)
endif()

file(GLOB_RECURSE ARIX_CPP_BRIDGE_SRC security/cpp/src/*.cpp)
if(ARIX_CPP_BRIDGE_SRC)
    target_sources(neural_security_cpp PRIVATE ${ARIX_CPP_BRIDGE_SRC})
endif()

# ---- Tools ----
if(ARIX_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# ---- Install ----
include(GNUInstallDirs)

install(TARGETS neural_core_kernel neural_architecture_layer neural_security_c
    EXPORT ARIXTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/neural_core/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neural_core
    FILES_MATCHING PATTERN "*.h"
)
install(EXPORT ARIXTargets
    FILE ARIXTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ARIX
)

# ---- CPack ----
set(CPACK_PACKAGE_NAME "ARIX-Algo")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "Ammar [ARIX]")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Cognitive architecture for secure AI")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
    "/build/"
    "/build_py/"
    "/build_bench/"
    "/.git/"
    "/target/"
    ".*.tar.gz"
    ".*.zip"
)
include(CPack)
