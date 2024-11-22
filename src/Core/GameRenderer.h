#pragma once
#include <memory>
#include <SDL.h>

namespace UtilSimpleECS
{
	/**
	 * Globally available SDL window references for internal use.
	 */
	class GameRenderer {
	public:
		/**
		 * Initializes game window and renderer
		 */
		static void initGameRenderer();

		/**
		 * SDL Renderer access
		 */
		static int SCREEN_WIDTH;
		static int SCREEN_HEIGHT;

		static SDL_Window* window;
		static SDL_Renderer* renderer;
		static SDL_Texture* gameTexture;
		static SDL_Surface* screenSurface;
	};
}