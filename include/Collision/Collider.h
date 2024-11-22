#pragma once

#include "SimpleECSAPI.h"

#include "Core/Component.h"
#include "Core/Vector.h"
#include <vector>
#include <iostream>

namespace SimpleECS
{
	// Collider class
	class SIMPLEECS_API Collider : public Component {

	public:
		struct AABB {
			double xMin, yMin;
			double xMax, yMax;
		};

		/**
		 * Register and deregister collider against ColliderSystem
		 * on construction/deconstruction
		 */
		Collider();
		~Collider();

		void update() override {}
		void initialize() override { std::cout << "wtf" << std::endl; }

		/**
		 * Returns whether this collider is colliding with another collider
		 */
		virtual bool isColliding(Collider* other) = 0;

		/**
		 * Gets AABB bounds of this collider
		 */
		virtual void getBounds(AABB& bounds) const = 0;
	};

	/**
	* Data container for collision information
	*/
	class SIMPLEECS_API Collision {
	public:
		// Collision object is not to be directly copied.
		Collision(const Collision&) = delete;
		Collision& operator=(const Collision&) = delete;

		Collider* a = nullptr;
		Collider* b = nullptr;
		double penetration = 0;
		Vector normal;
	};
}
