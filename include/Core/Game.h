#pragma once
#include "Core/Scene.h"
#include "Core/RenderConfig.h"
#include <vector>
#include <string>

#include "SimpleECSAPI.h"

namespace UtilSimpleECS {
	class IRenderStrategy;
}

namespace SimpleECS
{
	/**
	* A game instance represents the main game controller.
	* Sets up initialization and runs the main game loop. 
	*/
	class Game {
	public:
		// Singleton
		Game(const Game&) = delete;
		Game& operator=(const Game&) = delete;

		/**
		 * Singleton
		 */
		static SIMPLEECS_API Game& getInstance() {
			static Game instance;  
			return instance;
		}

		/**
		 * Configures window dimensions. Must be called before start game. 
		 *
		 */
		void SIMPLEECS_API configure(const RenderConfig& config);

		/**
		 * Start the main game loop. Game must have at least one scene.
		 *
		 * @throws Exception: If no scenes exist in game
		 */
		void SIMPLEECS_API startGame();

		/**
		 * Adds a scene containing entities to game instance.
		 *
		 * @param scene: Scene to be added.
		 */
		int SIMPLEECS_API addScene(Scene* scene);

		/**
		 * Sets name of window associated with this game.
		 */
		SIMPLEECS_API Scene* getCurrentScene();

	private:

		SIMPLEECS_API Game();
		
		std::vector<Scene*> sceneList;

		RenderConfig renderConfig;

		UtilSimpleECS::IRenderStrategy* renderer;

		int activeSceneIndex = 0;

		/**
		 * Call initialization functions
		 * 
		 * @returns 0 on failure, 1 on success.
		 * @throws Exception: On failure
		 */
		void init();

		/**
		 * @brief Main game loop that handles events, updates, rendering, and frame timing.
		 *
		 * This function is the core of the game engine, running continuously until the game is quit.
		 * Performs the following steps in each iteration of the loop:
		 *
		 * 1. **Event Handling**: Polls for SDL events and processes them. If an SDL_QUIT event is detected, the loop exits.
		 * 2. **GUI Update**: Updates the GUI components using the GuiManager.
		 * 3. **Screen Clearing**: Sets the render target to the game texture and clears the screen with the scene's background color.
		 * 4. **Scene Update**: Invokes the update functions for all component pools in the current scene.
		 * 5. **Collision Handling**: Invokes collision detection and handling using the ColliderSystem.
		 * 6. **Late Update**: Invokes the late update functions for all component pools in the current scene.
		 * 7. **Entity Deletion**: Deletes all entities marked for destruction in the current scene.
		 * 8. **Frame End**: Marks the end of the frame using the Timer.
		 * 9. **GUI Rendering**: Sets the render target to the window, clears the screen, and renders the GUI components.
		 * 10. **Present Frame**: Presents the rendered frame to the window.
		 *
		 * @throws std::runtime_error if no scenes are added to the game.
		 */
		void mainLoop();
	};
}