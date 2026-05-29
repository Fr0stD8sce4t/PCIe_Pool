cmake_minimum_required(VERSION 3.23)

project(turbobus_tests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../cpp turbobus_cpp_build)

add_executable(test_planner test_planner.cpp)
target_link_libraries(test_planner PRIVATE turbobus_core)

add_executable(test_direct_h2d test_direct_h2d.cu)
target_link_libraries(test_direct_h2d PRIVATE turbobus_core)

add_executable(test_relay_h2d_p2p test_relay_h2d_p2p.cu)
target_link_libraries(test_relay_h2d_p2p PRIVATE turbobus_core)

add_executable(test_multi_relay_pool test_multi_relay_pool.cu)
target_link_libraries(test_multi_relay_pool PRIVATE turbobus_core)

add_executable(test_profiler test_profiler.cu)
target_link_libraries(test_profiler PRIVATE turbobus_core)

add_executable(test_p2p_matrix test_p2p_matrix.cu)
target_link_libraries(test_p2p_matrix PRIVATE turbobus_core)
