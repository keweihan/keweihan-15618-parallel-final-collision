cmake_minimum_required(VERSION 3.16)
project(SimpleECS)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add global definition for DLL usage
add_definitions(-DSIMPLEECS_DLL)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Include Conan dependencies
find_package(Boost REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)
find_package(SDL2_mixer REQUIRED)
find_package(imgui REQUIRED CONFIG)

# Create interface library for headers
add_library(SimpleECS_Headers INTERFACE)
target_include_directories(SimpleECS_Headers INTERFACE 
    ${CMAKE_SOURCE_DIR}/include
)

get_target_property(IMGUI_INCLUDE_DIRS imgui::imgui INTERFACE_INCLUDE_DIRECTORIES)
message("ImGui include directories: ${IMGUI_INCLUDE_DIRS}")

# Subdirectories 
add_subdirectory(src)
add_subdirectory(demos)

message("Build type: ${CMAKE_BUILD_TYPE}")