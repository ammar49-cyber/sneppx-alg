cmake_minimum_required(VERSION 3.16)
project(arix-algo VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ARIX_BUILD_TESTS "Build test suite" ON)
option(ARIX_BUILD_PYTHON "Build Python bindings" OFF)
option(ARIX_BUILD_CUDA "Build CUDA kernels" OFF)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
elseif(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_C_FLAGS_DEBUG "/Zi /fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /fsanitize=address")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "/O2 /Zi")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Zi")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

file(GLOB ARIX_CORE_SRC src/core/src/*.c)
file(GLOB ARIX_SECURITY_C_SRC src/security/c/src/*.c)

add_library(arix_core STATIC ${ARIX_CORE_SRC})
target_include_directories(arix_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/arch/include
    ${CMAKE_SOURCE_DIR}/src/security/c/include
    ${CMAKE_SOURCE_DIR}/src/security/cpp/include
)

if(ARIX_SECURITY_C_SRC)
    add_library(arix_security_c STATIC ${ARIX_SECURITY_C_SRC})
    target_include_directories(arix_security_c PUBLIC ${CMAKE_SOURCE_DIR}/src/security/c/include)
endif()

if(ARIX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ARIX_BUILD_PYTHON)
    add_subdirectory(src/python)
endif()
