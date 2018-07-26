# Holy shit this works so far
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(TestProject LANGUAGES CXX)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)

add_subdirectory(external/glfw)
add_subdirectory(external/glm)

find_package(Vulkan REQUIRED)

add_executable(TestApp "${PROJECT_SOURCE_DIR}/src/main.cpp")
target_compile_features(TestApp PUBLIC cxx_std_14)
target_link_libraries(TestApp Vulkan::Vulkan glfw glm)