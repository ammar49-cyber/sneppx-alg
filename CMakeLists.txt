cmake_minimum_required(VERSION 3.16)
project(cognitive_processing_system VERSION 0.9.7.890 LANGUAGES C CXX ASM_MASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SNEPPX_BUILD_TESTS "Build test suite" ON)
option(SNEPPX_BUILD_BENCHMARKS "Build benchmarks" ON)
option(SNEPPX_BUILD_PYTHON "Build Python bindings" OFF)
option(SNEPPX_BUILD_CUDA "Build CUDA kernels" OFF)
option(SNEPPX_BUILD_TOOLS "Build CLI tools" OFF)
# Opt-in accelerator backends (OFF by default so Linux/Windows CI stays green).
# These enable real reference-compute implementations of the Metal and
# oneAPI/SYCL drivers. When OFF, the drivers compile as UNSUPPORTED stubs.
option(SNEPPX_BUILD_METAL "Build the Apple Metal backend (requires Apple SDK)" OFF)
option(SNEPPX_BUILD_ONEAPI "Build the Intel oneAPI/SYCL backend (requires DPC++)" OFF)
# Additional opt-in subsystems (v0.9.7.x). All OFF by default.
option(SNEPPX_BUILD_VULKAN "Build the Vulkan backend (requires volk)" OFF)
option(SNEPPX_BUILD_TPU "Build the TPU/PjRt backend (requires PJRT C API)" OFF)
option(SNEPPX_BUILD_HTTP "Build the HTTP inference server" OFF)
option(SNEPPX_BUILD_ZK "Build ZK proof primitives" OFF)
option(SNEPPX_USE_ASAN "Enable AddressSanitizer (GCC/Clang only)" OFF)
option(SNEPPX_USE_UBSAN "Enable UndefinedBehaviorSanitizer (GCC/Clang only)" OFF)
option(SNEPPX_USE_LTO "Enable Link-Time Optimization (Release only)" OFF)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    if(SNEPPX_USE_ASAN)
        set(CMAKE_C_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address")
    else()
        set(CMAKE_C_FLAGS_DEBUG "-g")
        set(CMAKE_CXX_FLAGS_DEBUG "-g")
    endif()
    if(SNEPPX_USE_UBSAN)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=undefined -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow")
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=undefined")
    endif()
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    else()
        set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    endif()
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes")
    endif()
elseif(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4 /DSNEPPX_CRYPTO_SECURE /D_CRT_SECURE_NO_WARNINGS /FS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /DSNEPPX_CRYPTO_SECURE /D_CRT_SECURE_NO_WARNINGS /FS")
    set(CMAKE_C_FLAGS_DEBUG "/Zi /RTC1")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /RTC1")
    if(SNEPPX_USE_ASAN)
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

if(SNEPPX_USE_LTO)
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

file(GLOB_RECURSE CORE_KERNEL_SRC kernel/*.c drivers/*.c)
file(GLOB_RECURSE CORE_SECURITY_C_SRC security/crypto/c/*.c)
file(GLOB_RECURSE FS_SNAPSHOT_SRC fs/*.c)

add_library(neural_core_kernel STATIC ${CORE_KERNEL_SRC} ${FS_SNAPSHOT_SRC})
target_include_directories(neural_core_kernel PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/kernel>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/security>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/architecture>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/fs/format>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/neural_core/kernel>
    $<INSTALL_INTERFACE:include/neural_core/security>
    $<INSTALL_INTERFACE:include/neural_core/architecture>
)

if(SNEPPX_BUILD_METAL)
    target_compile_definitions(neural_core_kernel PRIVATE SNEPPX_BUILD_METAL)
endif()
if(SNEPPX_BUILD_ONEAPI)
    target_compile_definitions(neural_core_kernel PRIVATE SNEPPX_BUILD_ONEAPI)
endif()
if(SNEPPX_BUILD_VULKAN)
    target_compile_definitions(neural_core_kernel PRIVATE SNEPPX_BUILD_VULKAN)
endif()
if(SNEPPX_BUILD_TPU)
    target_compile_definitions(neural_core_kernel PRIVATE SNEPPX_BUILD_TPU)
endif()
if(SNEPPX_BUILD_HTTP)
    target_compile_definitions(neural_core_kernel PRIVATE SNEPPX_BUILD_HTTP)
endif()
if(SNEPPX_BUILD_ZK)
    target_compile_definitions(neural_core_kernel PRIVATE SNEPPX_BUILD_ZK)
endif()
if(SNEPPX_BUILD_HTTP)
    if(MSVC)
        target_link_libraries(neural_core_kernel PRIVATE ws2_32)
    endif()
endif()
if(NOT MSVC)
    target_link_libraries(neural_core_kernel PUBLIC m)
endif()

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
    target_compile_definitions(neural_security_c PRIVATE $<$<CONFIG:Release>:SNEPPX_S1_SECURE_MEMORY>)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        file(GLOB SNEPPX_ASM_X86_64_S security/crypto/asm/x86_64/*.S)
        if(SNEPPX_ASM_X86_64_S)
            target_sources(neural_security_c PRIVATE ${SNEPPX_ASM_X86_64_S})
        endif()
        if(MSVC)
            file(GLOB SNEPPX_ASM_X86_64_ASM security/crypto/asm/x86_64/*.asm)
            if(SNEPPX_ASM_X86_64_ASM)
                set_source_files_properties(${SNEPPX_ASM_X86_64_ASM} PROPERTIES LANGUAGE ASM_MASM)
                target_sources(neural_security_c PRIVATE ${SNEPPX_ASM_X86_64_ASM})
            endif()
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
    target_compile_definitions(neural_security_cpp PRIVATE SNEPPX_S2_OBFUSCATION)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_libraries(neural_security_cpp PRIVATE dl)
    endif()
endif()

if(SNEPPX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(examples)
endif()

if(SNEPPX_BUILD_BENCHMARKS)
    add_subdirectory(tests/benchmark)
endif()

if(SNEPPX_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()

if(SNEPPX_BUILD_CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED)
    enable_language(CUDA)

    # CUDA architecture targets
    set(CMAKE_CUDA_ARCHITECTURES "80;90")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # CUDA compile flags
    if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -O3 -DNDEBUG -use_fast_math")
        set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
        set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -use_fast_math")
        # Architecture-specific flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSNEPPX_CUDA_ARCH_80 -DSNEPPX_CUDA_ARCH_90")
    endif()

    # Find NCCL for distributed training
    find_package(NCCL QUIET)

    # Core CUDA kernel library
    add_subdirectory(kernel/cuda)

    # Algorithm CUDA extensions
    add_subdirectory(algorithms/hss/cuda)
    add_subdirectory(algorithms/ser/cuda)
    add_subdirectory(algorithms/arc/cuda)
    add_subdirectory(algorithms/npe/cuda)
    add_subdirectory(algorithms/fm/cuda)

    # Distributed communication
    add_subdirectory(net/distributed)

    # Link CUDA runtime to core libraries
    target_compile_definitions(neural_core_kernel PRIVATE SNEPPX_HAS_CUDA)
    target_compile_definitions(neural_architecture_layer PRIVATE SNEPPX_HAS_CUDA)
    target_link_libraries(neural_core_kernel PUBLIC CUDA::cudart CUDA::cublas)
    target_link_libraries(neural_architecture_layer PUBLIC CUDA::cudart)
endif()

file(GLOB_RECURSE SNEPPX_MONITOR_SRC security/monitor/*.c)
if(SNEPPX_MONITOR_SRC)
    target_sources(neural_security_c PRIVATE ${SNEPPX_MONITOR_SRC})
endif()

file(GLOB_RECURSE SNEPPX_SECURITY_MEMORY_SRC security/memory/*.c)
if(SNEPPX_SECURITY_MEMORY_SRC)
    target_sources(neural_security_c PRIVATE ${SNEPPX_SECURITY_MEMORY_SRC})
    target_compile_definitions(neural_security_c PRIVATE SNEPPX_S1_SECURE_ALLOCATOR)
endif()

file(GLOB_RECURSE SNEPPX_C_BINDING_SRC security/c/src/*.c)
if(SNEPPX_C_BINDING_SRC)
    target_sources(neural_security_c PRIVATE ${SNEPPX_C_BINDING_SRC})
endif()

file(GLOB_RECURSE SNEPPX_S4_NETWORK_SRC security/network/*.c)
if(SNEPPX_S4_NETWORK_SRC)
    target_sources(neural_security_c PRIVATE ${SNEPPX_S4_NETWORK_SRC})
    target_compile_definitions(neural_security_c PRIVATE SNEPPX_S4_NETWORK_SECURITY)
endif()

file(GLOB_RECURSE SNEPPX_S5_AI_SRC security/ai/*.c)
if(SNEPPX_S5_AI_SRC)
    target_sources(neural_security_c PRIVATE ${SNEPPX_S5_AI_SRC})
    target_compile_definitions(neural_security_c PRIVATE SNEPPX_S5_AI_SANITIZER)
endif()

file(GLOB_RECURSE SNEPPX_S6_UI_SRC security/ui/*.c)
if(SNEPPX_S6_UI_SRC)
    target_sources(neural_security_c PRIVATE ${SNEPPX_S6_UI_SRC})
    target_compile_definitions(neural_security_c PRIVATE SNEPPX_S6_SECURITY_UI)
endif()

file(GLOB_RECURSE SNEPPX_S7_UPDATE_SRC security/updates/*.c)
if(SNEPPX_S7_UPDATE_SRC)
    target_sources(neural_security_c PRIVATE ${SNEPPX_S7_UPDATE_SRC})
    target_compile_definitions(neural_security_c PRIVATE SNEPPX_S7_SECURE_UPDATES)
endif()

file(GLOB_RECURSE SNEPPX_S8_FORMAL_SRC security/formal/*.c)
if(SNEPPX_S8_FORMAL_SRC)
    target_sources(neural_security_c PRIVATE ${SNEPPX_S8_FORMAL_SRC})
    target_compile_definitions(neural_security_c PRIVATE SNEPPX_S8_FORMAL_VERIFICATION)
endif()

file(GLOB_RECURSE SNEPPX_S9_PENTEST_SRC security/pentest/*.c)
if(SNEPPX_S9_PENTEST_SRC)
    target_sources(neural_security_c PRIVATE ${SNEPPX_S9_PENTEST_SRC})
    target_compile_definitions(neural_security_c PRIVATE SNEPPX_S9_PENETRATION_TESTING)
endif()

file(GLOB_RECURSE SNEPPX_CPP_BRIDGE_SRC security/cpp/src/*.cpp)
if(SNEPPX_CPP_BRIDGE_SRC)
    target_sources(neural_security_cpp PRIVATE ${SNEPPX_CPP_BRIDGE_SRC})
endif()

if(ENABLE_CLANG_TIDY)
    enable_clang_tidy(neural_core_kernel)
    enable_clang_tidy(neural_architecture_layer)
    if(TARGET neural_security_c)
        enable_clang_tidy(neural_security_c)
    endif()
endif()

# ---- Tools ----
if(SNEPPX_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# ---- Install ----
include(GNUInstallDirs)

install(TARGETS neural_core_kernel neural_architecture_layer neural_security_c
    EXPORT SNEPPXTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/neural_core/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neural_core
    FILES_MATCHING PATTERN "*.h"
)
install(EXPORT SNEPPXTargets
    FILE SNEPPXTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SNEPPX
)

# ---- CPack ----
set(CPACK_PACKAGE_NAME "SNEPPX-Algo")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "Ammar [SNEPPX]")
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
