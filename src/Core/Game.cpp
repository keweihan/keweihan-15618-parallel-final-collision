#include "Core/Game.h"
#include "Core/Scene.h"
#include "GameRenderer.h"
#include "Core/Entity.h"
#include "../Collision/ColliderSystem.h"
#include "Core/ComponentPool.h"
#include "Core/Timer.h"
#include "Utility/Color.h"
#include "GUI/GuiManager.h"
#include <SDL.h>

#include "Core/RenderStrategy/IRenderStrategy.h"
#include "Core/RenderStrategy/SDLImGuiRenderer.h"
#include "Core/RenderStrategy/HeadlessRenderer.h"

#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_sdlrenderer2.h>

#include <iostream>

using namespace SimpleECS;
using namespace UtilSimpleECS;



Scene* Game::getCurrentScene()
{
	return sceneList[activeSceneIndex];
}

Game::Game()
{
	GameRenderer::SCREEN_WIDTH = 640;
	GameRenderer::SCREEN_HEIGHT = 480;
}

void Game::configure(const RenderConfig& config)
{
	GameRenderer::SCREEN_WIDTH = config.width;
	GameRenderer::SCREEN_HEIGHT = config.height;

	renderConfig = config;
	if(config.enableWindow)
		renderer = new SDLImGuiRenderer();
	else
		renderer = new HeadlessRenderer();
}

void Game::startGame()
{
	init();
	mainLoop();
}

int Game::addScene(Scene* scene)
{
	sceneList.push_back(scene);
	return static_cast<int>(sceneList.size() - 1);
}

void Game::init()
{
	renderer->init(renderConfig);
}

void Game::mainLoop()
{
	SDL_Event e;
	bool quit = false;
	bool movingWindow = false;

	if (sceneList.size() == 0) {
		throw std::runtime_error("No scenes in game! ");
	}

	// Run initialize of first scene components
	for (auto& pool : sceneList[0]->getComponentPools())
	{
		(*pool).invokeStart();
	}

	// Game loop
	while (!quit)
	{
		// Check for closing window
		while (SDL_PollEvent(&e))
		{
			// TODO: Move/abstract logic to GuiManager.
			ImGui_ImplSDL2_ProcessEvent(&e);

			if (e.type == SDL_QUIT)
			{
				quit = true;
			}
		}

		renderer->frameBegin(sceneList[0]);

		// Run update of first scene functions
		for (auto& pool : sceneList[0]->getComponentPools())
		{
			(*pool).invokeUpdate();
		}

		// Run collision functions
		ColliderSystem::getInstance().invokeCollisions();

		// Run late update
		for (auto& pool : sceneList[0]->getComponentPools())
		{
			(*pool).invokeLateUpdate();
		}

		// Delete objects
		sceneList[0]->destroyAllMarkedEntities();

		// Mark end of frame
		Timer::endFrame();
		
		renderer->frameEnd(sceneList[0]);
	}
}
