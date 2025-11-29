cmake_minimum_required(VERSION 3.18)
project(MixedGemm LANGUAGES CXX CUDA)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _torch_found
)

if(NOT _torch_found EQUAL 0)
  message(FATAL_ERROR "PyTorch not found. Please make sure 'import torch' works in your Python environment.")
else()
  list(INSERT CMAKE_PREFIX_PATH 0 "${TORCH_CMAKE_PREFIX_PATH}")
  message(STATUS "Found PyTorch CMAKE_PREFIX_PATH: ${TORCH_CMAKE_PREFIX_PATH}")
endif()
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH}/../../)
find_package(Torch REQUIRED)

find_package(CUDAToolkit REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
find_package(pybind11 CONFIG REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_CMAKE_PREFIX_PATH}/../../lib/ NO_DEFAULT_PATH)

set(CUTLASS_ROOT ../third-party/cutlass)

include_directories(./include)
include_directories(/usr/local/cuda/include)
include_directories(${PYTHON_ROOT})
link_directories(${PYTHON_ROOT}/site-packages/torch/lib)
include_directories(${PYTHON_ROOT}/site-packages/torch/include)
include_directories(${PYTHON_ROOT}/site-packages/torch/include/torch/csrc/api/include)
include_directories(${CUTLASS_ROOT}/include)
include_directories(${CUTLASS_ROOT}/tools/util/include)
include_directories(${CUTLASS_ROOT}/examples/common)

set(CMAKE_CUDA_ARCHITECTURES 120a)

add_compile_options(-w)
add_compile_options(-fpermissive)

add_library(gemm_objs OBJECT src/nvfp4.cu src/reorder.cu src/rmsnorm.cu src/down.cu src/flashinfer.cu)


add_executable(bench_nvfp4 benchmark/bench_nvfp4.cu
$<TARGET_OBJECTS:gemm_objs>)


add_library(agemm MODULE src/bindings.cpp $<TARGET_OBJECTS:gemm_objs>)
set_source_files_properties(src/bindings.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(agemm PRIVATE pybind11::module "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" CUDA::cudart)

target_compile_options(bench_nvfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(gemm_objs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)
target_compile_options(agemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_compile_definitions(bench_nvfp4 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(agemm PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)

set_property(TARGET agemm PROPERTY CXX_STANDARD 17)
set_target_properties(agemm PROPERTIES PREFIX "")
add_definitions(-DTORCH_EXTENSION_NAME=agemm)

