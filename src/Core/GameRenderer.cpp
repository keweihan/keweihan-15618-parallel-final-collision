#include "GameRenderer.h"
#include <SDL.h>
#include <iostream>
#include <memory>
#include <utility>
#include <iostream>

using namespace UtilSimpleECS;

// ---------------- GameRenderer definitions ----------------//
void GameRenderer::initGameRenderer()
{
	// Initialize SDL
	if (SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_AUDIO))
	{
		throw std::runtime_error("Failure to initialize SDL. SDL_Error: " + std::string(SDL_GetError()));
	}

	// Create window
	window = SDL_CreateWindow(NULL, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
	if (window == NULL)
	{
		throw std::runtime_error("Failure to create window. SDL_Error: " + std::string(SDL_GetError()));
	}

	// Create renderer for window
	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	SDL_SetHintWithPriority(SDL_HINT_RENDER_VSYNC, "0", SDL_HINT_OVERRIDE);
	SDL_GL_SetSwapInterval(0);
	if (renderer == NULL) {
		std::cout << "Warning: Hardware acceleration not available. Falling back to software renderer." << std::endl;
		renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
	}
	//renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	gameTexture = SDL_CreateTexture(GameRenderer::renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, GameRenderer::SCREEN_WIDTH, GameRenderer::SCREEN_HEIGHT);
}

int GameRenderer::SCREEN_WIDTH  = 960;
int GameRenderer::SCREEN_HEIGHT = 540;

SDL_Window* GameRenderer::window			= nullptr;
SDL_Renderer* GameRenderer::renderer		= nullptr;
SDL_Texture* GameRenderer::gameTexture		= nullptr;
