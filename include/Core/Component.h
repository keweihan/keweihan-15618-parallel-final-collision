#pragma once


#include "SimpleECSAPI.h"

namespace SimpleECS 
{
	class Entity;
	class Collider;
	class Collision;

	class SIMPLEECS_API Component {
		friend class Entity;

	public:
		virtual ~Component() {};

		/*
		* Called on every frame of rendering
		*/
		virtual void update() = 0;

		/*
		* Called during start of game loop.
		*/
		virtual void initialize() = 0;

		/*
		* Called after update after every frame of rendering
		*/
		virtual void lateUpdate() {}

		/*
		* Called on collision with another entity
		*/
		virtual void onCollide(const Collider& collide) {}

		/*
		* Called on collision with another entity. Gets collision information.
		*/
		virtual void onCollide(const Collision& collide) {}

		/*
		* Set component active status (if inactive, update no longer called)
		*/
		virtual void setActive(bool b) { isActive = b; }

		/*
		* Get component active status (if inactive, update no longer called)
		*/
		virtual bool getActive() { return isActive; }

		/*
		* The entity this component is attached to
		*/
		Entity* entity = nullptr;

		/*
		* Called on adding this component to an entity.
		*/
		void setEntity(Entity* entity);
	
	private:
		bool isActive = true;
	};
}
