#pragma once
#include <imgui.h>
#include "Core/Entity.h"

namespace UtilSimpleECS
{
	/**
	 * Master Singleton class for GUI display.
	 * Provides component-like interface for use in main game loop.
	 */
	class GuiManager {

	public:
		GuiManager() {};

		static GuiManager& getInstance()
		{
			static GuiManager instance; // Guaranteed to be destroyed.
			// Instantiated on first use.
			return instance;
		}

		GuiManager(GuiManager const&) = delete;
		void operator=(GuiManager const&) = delete;

		/**
		 * Initialize GUI components and libraries
		 */
		void init();

		/**
		 * Initialize GUI components and libraries
		 */
		void update();

		/**
		 * Render calls. Not often changed
		 */
		void render();
	
	private:
		/**
		 * Configure theming
		 */
		void applyTheme();

		SimpleECS::Entity* selectedEntity = nullptr;
		ImFont* font = nullptr;
	};
}