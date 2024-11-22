#pragma once
#include "Core/Component.h"
#include <vector>
#include <memory>
#include <unordered_set>
#include <iterator>
#include <stdexcept>
#include <iostream>
#include <vector>

#include "SimpleECSAPI.h"

namespace SimpleECS {
	/*
	* Typeless base class representing component pools.
	*/
	class SIMPLEECS_API ComponentPoolBase {
	public:
		virtual ~ComponentPoolBase() {}

		/*
		* Return raw, UNSTABLE pointer to component owned by entityID, if it exists.
		* WARNING: pointer is NOT guaranteed to stay valid. createComponent
		* can invalidate this pointer.
		*
		* @return
		* raw pointer to component owned by entity id.
		* nullptr if entityID does not have component.
		*/
		virtual Component* getComponentRaw(uint32_t entityID) = 0; // note: Component* are not stable.
		
		/*
		* Delete component from entityID if it has component.
		* No error or behavior if entity does not have component.
		*/
		virtual void deleteComponent(uint32_t entityID) = 0;
		
		/*
		* Invoke start() of all components in pool.
		*/
		virtual void invokeStart() = 0;

		/*
		* Invoke update() of all components in pool.
		*/
		virtual void invokeUpdate() = 0;

		/*
		* Invoke lateUpdate() of all components in pool.
		*/
		virtual void invokeLateUpdate() = 0;
	};

	/*
	* Contiguous storage container for components of type T.
	*/
	template<typename T>
	class ComponentPool : public ComponentPoolBase {
	public:
		ComponentPool() {};

		/*
		* Create component of class type and assign it to the entity entityID.
		* @throws If entity already has component
		*/
		template <typename... Args>
		void createComponent(uint32_t entityID, Args&&... args);

		/*
		* Delete component from entityID if it has component.
		* No error or behavior if entity does not have component. 
		*/
		void deleteComponent(uint32_t entityID);

		/*
		* Return raw pointer to component owned by entityID, if it exists.
		* WARNING: pointer is NOT guaranteed to stay valid. createComponent
		* can invalidate this pointer.
		* 
		* @return 
		* raw pointer to component owned by entity id.
		* nullptr if entityID does not have component.
		*/
		T* getComponent(uint32_t entityID);

		/*
		* Override of base method. 
		*/
		Component* getComponentRaw(uint32_t entityID) override;

		/*
		* Invoke start() of all components in pool.
		*/
		void invokeStart();

		/*
		* Invoke update() of all components in pool.
		*/
		void invokeUpdate();

		/*
		* Invoke lateUpdate() of all components in pool.
		*/
		void invokeLateUpdate();

		/*
		Get list of components of this pool
		*/
		std::vector<T>* getComponents();
		
		/*
		* Sparse storage of indices. Index represents entityId and value the index of component
		* in componentList tha that belongs to entityId.
		* 
		* Value -1 at index 0 indicates entity 0 does not have component of this type.
		* Value 3 at index 2 indicates entity 2 has component of this type stored at componentList[3].
		* 
		* Note: Entity can only have one of each component type. 
		*/
		std::vector<int> sparseList; 

		/*
		* Dense storage of components owned by entities.
		*/ 
		std::vector<T> componentList;
	};

	template<typename T>
	template <typename... Args>
	void SimpleECS::ComponentPool<T>::createComponent(uint32_t entityID, Args&&... args)
	{
		if (entityID >= sparseList.size())
		{
			sparseList.resize(entityID + 1, -1);
		}
		else if (sparseList[entityID] != -1)
		{
			throw std::logic_error("Entity already has a component of this type.");
		}

		componentList.emplace_back(std::forward<Args>(args)...);
		sparseList[entityID] = static_cast<int>(componentList.size() - 1);

		//std::cout << "Created component list size: " << componentList.size() << std::endl;
		
	}

	template<typename T>
	void SimpleECS::ComponentPool<T>::deleteComponent(uint32_t entityID)
	{
		// Check if entity has this component
		if (entityID >= sparseList.size() || sparseList[entityID] == -1)
		{
			return;
			//throw std::logic_error("Entity does not have component to delete.");
		}

		// Remove the component from the componentList
		int index = sparseList[entityID];
		componentList[index] = componentList.back();
		componentList.pop_back();
		
		// Update the sparseList to reflect the removal
		int ent = componentList[index].entity->id;
		sparseList[ent] = index;
		sparseList[entityID] = -1;
	}

	template<typename T>
	T* SimpleECS::ComponentPool<T>::getComponent(uint32_t entityID)
	{
		if (entityID >= sparseList.size() || sparseList[entityID] == -1)
		{
			return nullptr;
		}
		else
		{
			return &componentList[sparseList[entityID]];
		}
	}

	template<typename T>
	void SimpleECS::ComponentPool<T>::invokeStart()
	{
		for(auto& component : componentList)
		{
			component.initialize();
		}
	}

	template<typename T>
	void SimpleECS::ComponentPool<T>::invokeUpdate()
	{
		for (auto& component : componentList)
		{
			if (component.getActive()) {
				component.update();
			}
		}
	}

	template<typename T>
	inline void ComponentPool<T>::invokeLateUpdate()
	{
		for (auto& component : componentList)
		{
			if (component.getActive()) {
				component.lateUpdate();
			}
		}
	}

	template<typename T>
	std::vector<T>* SimpleECS::ComponentPool<T>::getComponents()
	{
		return &componentList;
	}

	template<typename T>
	inline Component* ComponentPool<T>::getComponentRaw(uint32_t entityID)
	{
		return getComponent(entityID);
	}
}
