# src/CMakeLists.txt

# Create a library from source files
file(GLOB_RECURSE SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.h
)

# Create SimpleECS DLL
add_library(SimpleECS SHARED ${SRC_FILES})

# Add DLL export definitions
target_compile_definitions(SimpleECS 
    PRIVATE SIMPLEECS_EXPORTS
    PUBLIC SIMPLEECS_DLL
)

# Configure RTTI
if(MSVC)
    target_compile_options(SimpleECS PRIVATE /GR)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wno-narrowing)
endif()

# Set properties
set_target_properties(SimpleECS PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Include Imgui SDL backend
get_filename_component(IMGUI_ROOT "${imgui_INCLUDE_DIRS}" DIRECTORY)
message("ImGui root: ${IMGUI_ROOT}")
message("Looking for backends in: ${IMGUI_ROOT}/res/bindings")

target_sources(SimpleECS
    PRIVATE
        "${IMGUI_ROOT}/res/bindings/imgui_impl_sdl2.cpp"
        "${IMGUI_ROOT}/res/bindings/imgui_impl_sdlrenderer2.cpp"
)

target_include_directories(SimpleECS
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        "${IMGUI_ROOT}/res/bindings" # Include imgui backends
)

get_target_property(IMGUI_INCLUDE_DIRS imgui::imgui INTERFACE_INCLUDE_DIRECTORIES)
message("ImGui include directories: ${IMGUI_INCLUDE_DIRS}")

target_link_libraries(SimpleECS
    PUBLIC
        SimpleECS_Headers
    PRIVATE
        SDL2::SDL2main
        SDL2::SDL2
        imgui::imgui
        sdl_ttf::sdl_ttf
        SDL2_mixer::SDL2_mixer
        # SDL2_image::SDL2_image // Not a dependency
        Boost::boost
        ${CMAKE_DL_LIBS}
)