cmake_minimum_required(VERSION 3.23)

project(turbobus LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(turbobus_core
  src/executor_cuda.cu
  src/planner.cpp
  src/profiler_cuda.cu
  src/runtime.cpp
  src/topology_cuda.cpp
)

target_include_directories(turbobus_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(turbobus_core
  PUBLIC
    CUDA::cudart
)

set_target_properties(turbobus_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

option(TURBOBUS_BUILD_PYTHON "Build the pybind11 Python extension" OFF)

if(TURBOBUS_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_turbobus src/pybind.cpp)
  target_link_libraries(_turbobus PRIVATE turbobus_core)
  target_include_directories(_turbobus PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(_turbobus PROPERTIES OUTPUT_NAME "_turbobus")
endif()
