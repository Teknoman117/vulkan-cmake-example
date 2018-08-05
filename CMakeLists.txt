# Holy shit this works so far
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(TestProject LANGUAGES CXX)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)

find_package(Vulkan REQUIRED)

add_subdirectory(external/glfw)
add_subdirectory(external/glm)

# chapter01 executable
add_executable(chapter01
    "${PROJECT_SOURCE_DIR}/src/chapter01/main.cpp"
)
target_compile_features(chapter01 PUBLIC cxx_std_17)
target_link_libraries(chapter01 Vulkan::Vulkan glfw glm)

# chapter01 executable
add_executable(chapter02
    "${PROJECT_SOURCE_DIR}/src/chapter02/main.cpp"
)
target_compile_features(chapter02 PUBLIC cxx_std_17)
target_link_libraries(chapter02 Vulkan::Vulkan glfw glm)