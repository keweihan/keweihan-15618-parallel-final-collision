# FindSDL2_mixer.cmake
if(NOT TARGET SDL2_mixer::SDL2_mixer)
    find_path(SDL2_MIXER_INCLUDE_DIR SDL_mixer.h PATHS /usr/include/SDL2)
    find_library(SDL2_MIXER_LIBRARY SDL2_mixer PATHS /usr/lib /usr/lib/x86_64-linux-gnu)

    set(SDL2_TTF_LIBRARIES ${SDL2_MIXER_LIBRARY})
    set(SDL2_MIXER_INCLUDE_DIRS ${SDL2_MIXER_INCLUDE_DIR})

    mark_as_advanced(SDL2_MIXER_INCLUDE_DIR SDL2_MIXER_LIBRARY)

    # Create an imported target for SDL2_ttf
    if(SDL2_MIXER_INCLUDE_DIR AND SDL2_MIXER_LIBRARY)
        add_library(SDL2_mixer::SDL2_mixer INTERFACE IMPORTED)
        set_target_properties(SDL2_mixer::SDL2_mixer PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${SDL2_MIXER_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${SDL2_MIXER_LIBRARY}"
        )
    else()
        message(FATAL_ERROR "SDL2_ttf library not found")
    endif()
endif()