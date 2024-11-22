#pragma once

#include "Core/CHandle.h"
#include "Core/Scene.h"
#include "Core/Transform.h"
#include <vector>
#include <string>
#include <utility>

#include "SimpleECSAPI.h"

namespace SimpleECS
{	
	template <typename T>
	class Handle;

	/**
	An object/actor inside scenes. Has a container of components which dictate entity behavior.
	and exists in world space.
	*/
	class Entity {
	private:		
		
		/**
		* Pointer to scene containing this entity.
		*/
		Scene* scene;

		/**
		* Constructs entity with respect to a scene. See Scene::createEntity() for
		* user creation of entity object.
		*/
		friend Scene;
		SIMPLEECS_API Entity(uint32_t id, Scene* s) : id(id), scene(s) {};
		SIMPLEECS_API ~Entity();
	
	public:
		/**
		* Internal identifier for this entity. Instantiated on construction.
		* TODO make private
		*/
		uint32_t id;

		/**
		* Optional string identifier for this entity
		*/
		std::string tag = "Default";

		/**
		* Entity position in world space.
		* TODO: redo...
		*/
		Handle<Transform> transform;

		/**
		* Add a component to this entity of type T
		* 
		* @returns Component* added to entity.
		*/
		template <typename T, typename... Args>
		Handle<T> addComponent(Args&&... args);

		/**
		* Retrieve a component attached to entity of type T.
		* 
		* @returns A single component of type T attached to entity.
		* nullptr if no component of such type is attached to entity.
		* 
		*/
		template <typename T>
		Handle<T> getComponent();

		std::vector<Component*> getComponents();
	};

	template <typename T, typename... Args>
	inline Handle<T> Entity::addComponent(Args&&... args)
	{
		return scene->addComponent<T>(id, std::forward<Args>(args)...);
	}

	template<typename T>
	inline Handle<T> Entity::getComponent()
	{
		return scene->getComponent<T>(id);
	}
}
