#pragma once
#include "./SDLImGuiRenderer.h"
#include "Core/Game.h"
#include "Core/Scene.h"
#include "../GameRenderer.h"
#include "GUI/GuiManager.h"
#include "Utility/Color.h"
#include <SDL.h>

#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_sdlrenderer2.h>
#include "SDLImGuiRenderer.h"


UtilSimpleECS::SDLImGuiRenderer::~SDLImGuiRenderer()
{

}

void UtilSimpleECS::SDLImGuiRenderer::init(const SimpleECS::RenderConfig &config)
{
	GameRenderer::initGameRenderer();
	SDL_SetWindowTitle(GameRenderer::window, config.gameName.c_str());
	SDL_SetWindowResizable(GameRenderer::window, SDL_bool(true));
	GuiManager::getInstance().init();
}

void UtilSimpleECS::SDLImGuiRenderer::frameBegin(SimpleECS::Scene* scene)
{
	// Define engine GUI components
	GuiManager::getInstance().update();

	//Clear screen
	SDL_SetRenderTarget(GameRenderer::renderer, GameRenderer::gameTexture);
	SimpleECS::Color sceneColor = scene->backgroundColor;
	SDL_SetRenderDrawColor(GameRenderer::renderer, sceneColor.r, sceneColor.g, sceneColor.b, sceneColor.a);
	SDL_RenderClear(GameRenderer::renderer);
}

void UtilSimpleECS::SDLImGuiRenderer::frameEnd(SimpleECS::Scene* scene)
{
	// Render engine GUI components to window
	SDL_SetRenderTarget(GameRenderer::renderer, NULL);

	// Set window background coloring and clear render
	SimpleECS::Color sceneColor = scene->backgroundColor;
	SDL_SetRenderDrawColor(GameRenderer::renderer, sceneColor.r, sceneColor.g, sceneColor.b, sceneColor.a);
	SDL_RenderClear(GameRenderer::renderer);

	GuiManager::getInstance().render();
	SDL_RenderPresent(GameRenderer::renderer);
}