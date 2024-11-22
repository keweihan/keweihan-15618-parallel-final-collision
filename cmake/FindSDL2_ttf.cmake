# FindSDL2_ttf.cmake
find_path(SDL2_TTF_INCLUDE_DIR SDL_ttf.h PATHS /usr/include/SDL2)
find_library(SDL2_TTF_LIBRARY SDL2_ttf PATHS /usr/lib /usr/lib/x86_64-linux-gnu)

set(SDL2_TTF_LIBRARIES ${SDL2_TTF_LIBRARY})
set(SDL2_TTF_INCLUDE_DIRS ${SDL2_TTF_INCLUDE_DIR})

mark_as_advanced(SDL2_TTF_INCLUDE_DIR SDL2_TTF_LIBRARY)

# Create an imported target for SDL2_ttf
if(SDL2_TTF_INCLUDE_DIR AND SDL2_TTF_LIBRARY)
    add_library(sdl_ttf::sdl_ttf INTERFACE IMPORTED)
    set_target_properties(sdl_ttf::sdl_ttf PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${SDL2_TTF_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${SDL2_TTF_LIBRARY}"
    )
else()
    message(FATAL_ERROR "SDL2_ttf library not found")
endif()