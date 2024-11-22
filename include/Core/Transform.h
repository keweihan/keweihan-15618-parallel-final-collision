#pragma once
#include "Core/Vector.h"
namespace SimpleECS
{
	class Transform : public Component
	{
	public:
		/**
		 * Position of this entity in world space
		 */
		Vector position = Vector(0, 0);

		/**
		 * TODO: Rotation of transform 
		 */
		//double rotation = 0;

		void update() override {};
		void initialize() override {};
	};
}