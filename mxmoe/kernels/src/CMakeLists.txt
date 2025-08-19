cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CUDA_ARCHITECTURES 89) # A100: 80
add_compile_definitions(COMPUTE_ARCH=${CMAKE_CUDA_ARCHITECTURES})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -diag-suppress 128 -diag-suppress 2417, -diag-suppress 20050")

project(mxmoe LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#################################################################
# Config
#################################################################
set(cutlass_dir "${CMAKE_SOURCE_DIR}/3rdparty/cutlass")
set(cccl_dir "${CMAKE_SOURCE_DIR}/3rdparty/cccl")

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${cutlass_dir}/include
  ${cutlass_dir}/tools/util/include
)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${cccl_dir}/lib/cmake/cccl")

#################################################################
# Pure C++/CUDA Targets (No Python!)
#################################################################

set(GENERATED_DIR "${CMAKE_SOURCE_DIR}/generated")
include_directories(${GENERATED_DIR})

file(GLOB GENERATED_CU_FILES "${GENERATED_DIR}/*.cu")

add_library(gen_lib SHARED ${GENERATED_CU_FILES})
target_compile_options(gen_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic -lineinfo --use_fast_math>) 
target_link_libraries(gen_lib
  PRIVATE
  fmt::fmt
)

add_executable(test test.cu)
target_link_options(test PRIVATE "-Wl,--no-as-needed")
target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic -lineinfo --use_fast_math>) 
target_link_libraries(test
  PRIVATE
  fmt::fmt
  CCCL::Thrust
  nlohmann_json::nlohmann_json
  argparse
  gen_lib
)

# fmt
set(BUILD_SHARED_LIBS ON)
add_subdirectory(3rdparty/fmt)

# CCCL
find_package(CCCL REQUIRED CONFIG COMPONENTS Thrust CUB)

add_subdirectory(3rdparty/json)

add_subdirectory(3rdparty/argparse)


