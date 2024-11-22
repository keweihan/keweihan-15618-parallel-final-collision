#include "Input/Input.h"
#include <SDL.h>

bool SimpleECS::Input::getKeyDown(KeyCode key)
{
    const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
    return currentKeyStates[static_cast<int>(key)];
}
