#include "Core/Scene.h"
#include "Core/Entity.h"
#include <vector>

using namespace SimpleECS;

SimpleECS::Entity::~Entity()
{
	// Free all components attached to entity
	//scene->destroyEntity(*this);
	std::cout << "DESTROYED ENTITY: " << tag << std::endl;
}

std::vector<Component*> SimpleECS::Entity::getComponents()
{
	std::vector<Component*> components;
	for (auto& pool : scene->getComponentPools())
	{
		Component* c = pool->getComponentRaw(id);
		if(c != nullptr)
			components.push_back(c);
	}
	return components;
}
