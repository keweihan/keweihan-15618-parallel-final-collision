#pragma once

#include "Collision/Collider.h"
#include "ColliderGrid.h"
#include "Physics/PhysicsBody.h"
#include "vector"

namespace SimpleECS
{
	// Library only class for resolving collider logic
	class ColliderSystem
	{
	public:
		/*
		TODO: Refactor plan
		- Give Collidergrid "configure(BoxCollider... CircleCollider ...)" function
			- Replace single std::vector<Collider*> colliderList; with multiple lists of...
			- vector<BoxCollide>*, vector<SphereCollide>* etc.

		- Remove register/deregister collider functions
		- Replace with initialize system function to be called at start of Game
			- Collidergrid constructor or in this initializer should obtain references to said lists. 
		
		*/
		static ColliderSystem& getInstance()
		{
			static ColliderSystem instance; // Guaranteed to be destroyed.
			// Instantiated on first use.
			return instance;
		}

		// Delete these methods to ensure that copies of the singleton can't be made.
		ColliderSystem(ColliderSystem const&) = delete;
		void operator=(ColliderSystem const&) = delete;
		
		/*
		* Checks for collisions between all active colliders and invoke
		* collided entities "onCollision" methods.
		*/
		void invokeCollisions();

		/**
		 * Retrieves collision information between this and another collider.
		 * Populates collide with collision information if there is a collision
		 * between collide.a and collide.b 
		 * 
		 * Normal is calculated with respect to a colliding with b. 
		 * 
		 * @returns false if no collision is present, true otherwise
		 */
		bool getCollisionInfo(Collision& collide);

	private:
		ColliderSystem() : colliderGrid(ColliderGrid(2, 2)) {}

		/*
		* Maintains list of all active colliders in scene. 
		*/

		/*
		*
 		*/
		ColliderGrid colliderGrid;


		/*
		* If collide contains two AABB box containers. Populate with collision data
		*/
		bool getCollisionBoxBox(Collision& collide, BoxCollider* a, BoxCollider* b);
	};
}