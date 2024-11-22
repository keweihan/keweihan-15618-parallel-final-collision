#include "GameRenderer.h"
#include <SDL.h>
#include <iostream>
#include <memory>
#include <utility>

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
	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE); // SDL_RENDERER_ACCELERATED for GPU (not compatible with Mac)
	//renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	gameTexture = SDL_CreateTexture(GameRenderer::renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, GameRenderer::SCREEN_WIDTH, GameRenderer::SCREEN_HEIGHT);

	// Set white window surface
	screenSurface = SDL_GetWindowSurface(window);
	SDL_FillRect(screenSurface, NULL, SDL_MapRGB(screenSurface->format, 0xFF, 0xFF, 0xFF));
	SDL_UpdateWindowSurface(window);	

}

int GameRenderer::SCREEN_WIDTH  = 960;
int GameRenderer::SCREEN_HEIGHT = 540;

SDL_Window* GameRenderer::window			= nullptr;
SDL_Renderer* GameRenderer::renderer		= nullptr;
SDL_Surface* GameRenderer::screenSurface	= nullptr;
SDL_Texture* GameRenderer::gameTexture		= nullptr;
