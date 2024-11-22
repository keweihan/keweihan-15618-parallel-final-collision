#pragma once
#include "Core/Component.h"
#include "Utility/Color.h"

#include "SimpleECSAPI.h"

namespace SimpleECS
{
	/**
	Provides rectangle rendering to an entity. 
	*/
	class SIMPLEECS_API RectangleRenderer : public Component
	{
	public:
		RectangleRenderer() : width(40), height(40), renderColor(Color()) {}
		RectangleRenderer(int w, int h) : width(w), height(h) {}
		RectangleRenderer(int w, int h, Color color) : width(w), height(h), renderColor(color){}

		int width, height;
		Color renderColor;

		void update() override;
		void initialize() override; 
	};
}
