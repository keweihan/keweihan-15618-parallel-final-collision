#pragma once
#include "Core/Component.h"
#include "Utility/Color.h"
#include "Core/Vector.h"
#include <string>
#include <memory>


#include "SimpleECSAPI.h"

namespace SimpleECS
{
	/**
	Renders text when provided a .ttf font. Positioned at associated entity world position.
	*/
	class FontRenderer : public Component
	{
	public:
		SIMPLEECS_API FontRenderer(std::string text, std::string pathToFont, uint16_t size = 16, Color color = Color());
		//SIMPLEECS_API ~FontRenderer() { delete pImpl; }

		Color color		= Color(0,0,0,1);
		uint16_t size	= 24;
		std::string text = "FontRenderer Default Text";
		std::string path;

		/*
		* Returns(width, height) this font element occupies in world space.
		*/
		Vector SIMPLEECS_API getSize();
		
		void SIMPLEECS_API initialize() override;
		void SIMPLEECS_API update() override;
	private:
		class FontRendererImpl;
		FontRendererImpl* pImpl;
	};
}
