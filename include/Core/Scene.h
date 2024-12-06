#pragma once
#include "Core/Component.h"
#include "Utility/Color.h"
#include "Core/CHandle.h"
#include "Core/ComponentPool.h"
#include <unordered_set>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <string>
#include <unordered_map>

#include "SimpleECSAPI.h"

namespace SimpleECS
{
	class Entity;
	class Game;

	/*
	* Scene class represents a collection of Entities and components. A Game instance
	* can only display one scene at a time.
	*/
	class Scene {
	public:
		Scene() : backgroundColor(Color(255,255,255,255)) {};
		Scene(Color backgrundColor) : backgroundColor(backgrundColor) {};
		~Scene();

		// 15618 : get entity ptr based on eid
		SIMPLEECS_API Entity* getEntity(int eid) { return entities[eid]; }
		
		/**
		* Add Entity to this scene.
		* 
		* @returns false if entity is already contained by the scene and was not added. 
		* Otherwise returns true if successfully added.
		*/
		SIMPLEECS_API Entity* createEntity();
		SIMPLEECS_API Entity* createEntity(std::string tag);

		/*
		* Given an entity contained in this scene
		* create and attach component T to entity.
		* 
		* @throws if entity id is not contained in this scene. 
		* @returns Original component attached to entity
		*/
		template <typename T, typename... Args>
		Handle<T> addComponent(uint32_t e, Args&&... args);

		/*
		* Return list of components of type T.
		*
		* @throws if type T is not a component
		* @returns pointer original list of components
		*/
		template<typename T>
		std::vector<T>* getComponents();

		/*
		* Return component of type T attached to entity with id eid.
		*
		* @throws if type T is not a component
		* @returns pointer original list of components
		*/
		template<typename T>
		Handle<T> getComponent(uint32_t eid);

		/**
		* IMMEDIATELY Destroy entity contained by this scene. Proceed with caution, as
		* references can be broken 
		* 
		* @returns false if entity is not contained by the scene and was not deleted. 
		* Otherwise returns true if successfuly removed.
		* 
		*/
		SIMPLEECS_API bool destroyEntityImmediate(uint32_t eid);

		/**
		* Mark entity to be deleted at end of frame. Will call entity and component destructors
		* and entityToDelete will be deleted.
		*
		* @returns false if entity is not contained by the scene and was not deleted.
		* Otherwise returns true if successfuly removed.
		*
		*/
		SIMPLEECS_API bool destroyEntity(uint32_t eid);

		/**
		* Get a list of active entities
		*/
		SIMPLEECS_API std::vector<Entity*> getEntities();

		/**
		*  IMMEDIATELY destroys all entities marked for destruction (i.e. in toDestroyEntities) 
		*  Proceed with caution, as references can be broken.
		*/
		void destroyAllMarkedEntities();

		/*
		* Entities contained by the scene
		* TODO: move to private and create method to return active entities only.
		*/
		std::vector<Entity*> entities;

		/*
		* Main background render color.
		*/
		Color backgroundColor;

	private:
#pragma region Private_Members
		friend Entity;
		friend Game;

		/*
		* Return id unique to the type of component.
		*
		* @throws If T does not inherit from Component
		*/
		template<typename T>
		static std::size_t getComponentID();

		/*
		* Return the ID of the next entity to be created.
		*/
		std::uint32_t nextEntityID();

		/*
		* Component pool storage getter
		*/
		SIMPLEECS_API std::vector<std::shared_ptr<ComponentPoolBase>>& getComponentPools();

		/*
		* Pool of available entity ids. If empty, use max.
		*/
		std::unordered_set<uint32_t> availableEntityIds;

		/*
		* Entities marked for destruction. Cleared at end of every frame to be deleted.
		*/
		std::unordered_set<uint32_t> toDestroyEntities;

		/*
		* Component pool list for scene
		* Contiguous component pools storage.
		* Access component pool for type T with allComponents[getComponentID<T>()]
		*/
		std::vector<std::shared_ptr<ComponentPoolBase>> allComponents;

		/*
		* The next entity id to be created.
		*/
		int maxID = 0;
		SIMPLEECS_API static inline std::size_t counter = 0;
		SIMPLEECS_API static inline std::unordered_map<size_t, size_t> componentMap;

#pragma endregion Private_Members
	};

#pragma region Template_Implementation
	template <typename T, typename... Args>
	inline Handle<T> Scene::addComponent(uint32_t eid, Args&&... args)
	{
		// Check if T is of type component
		if (!std::is_base_of<Component, T>::value)
		{
			throw std::invalid_argument("Type called for addComponent is not a component.");
		}

		// Check if component pool exists
		size_t compId = getComponentID<T>();
		if (compId >= allComponents.size())
		{
			// Pool does not exist yet. Create component pool for type first
			allComponents.emplace_back(std::make_shared<ComponentPool<T>>());
		}

		// Assign component
		ComponentPool<T>* poolConv = dynamic_cast<ComponentPool<T>*>(&*allComponents[getComponentID<T>()]);
		if (!poolConv)
		{
			throw std::runtime_error("Failed to cast ComponentPoolBase to ComponentPool<T>.");
		}
		poolConv->createComponent(eid, std::forward<Args>(args)...);

		T* comp = poolConv->getComponent(eid);
		comp->setEntity(entities[eid]);

		Handle<T> handle(poolConv, eid);
		return handle;
	}

	template<typename T>
	std::vector<T>* SimpleECS::Scene::getComponents()
	{
		// Check that the component ID is within range
		std::size_t componentId = getComponentID<T>();
		if (componentId >= allComponents.size())
		{
			throw std::out_of_range("Component ID " + std::to_string(componentId) + " is out of range.");
		}

		// Cast ComponentPoolBase to concrete ComponentPool
		ComponentPool<T>* poolConv = dynamic_cast<ComponentPool<T>*>(&*allComponents[componentId]);
		if (!poolConv)
		{
			throw std::runtime_error("Failed to cast ComponentPoolBase to ComponentPool<T>.");
		}

		return poolConv->getComponents();
	}

	template<typename T>
	inline Handle<T> Scene::getComponent(uint32_t e)
	{
		// Check if T is of type component
		if (!std::is_base_of<Component, T>::value)
		{
			throw std::invalid_argument("Type called for addComponent is not a component.");
		}

		T* test = nullptr;	
		// Check that the component ID is within range
		std::size_t componentId = getComponentID<T>();
		if (componentId >= allComponents.size())
		{
			throw std::out_of_range("Component ID is out of range.");
		}

		// Cast ComponentPoolBase to concrete ComponentPool
		ComponentPool<T>* poolConv = dynamic_cast<ComponentPool<T>*>(&*allComponents[componentId]);
		if (!poolConv)
		{
			throw std::runtime_error("Failed to cast ComponentPoolBase to ComponentPool<T>.");
		}

		// Get the component and check if it's null
		T* component = poolConv->getComponent(e);
		if (!component)
		{
			throw std::runtime_error("Entity does not have a component of this type.");
		}

		return Handle<T>(poolConv, e);
	}

	template<typename T>
	inline std::size_t Scene::getComponentID()
	{
		size_t typeHash = typeid(T).hash_code();
		if (componentMap.find(typeHash) == componentMap.end())
		{
			componentMap[typeHash] = componentMap.size();
		}
		return componentMap[typeHash];
	}
#pragma endregion Template_Implementation
}
