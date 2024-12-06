#include "Core/Entity.h"
#include "Core/Scene.h"
#include "Core/Component.h"
#include "Physics/PhysicsBody.h"
#include <stdexcept>

using namespace SimpleECS;

Scene::~Scene() = default;

Entity* Scene::createEntity()
{
	Entity* created = new Entity(nextEntityID(), this);
	if (created->id >= entities.size()) {
		entities.push_back(created);
	}
	else {
		entities[created->id] = created;
	}
	created->transform = addComponent<Transform>(created->id);
	created->phys = addComponent<PhysicsBody>(created->id);
	return created;
}

SIMPLEECS_API Entity* SimpleECS::Scene::createEntity(std::string tag)
{
	Entity* created = createEntity();
	created->tag = tag;
	return created;
}

bool Scene::destroyEntityImmediate(uint32_t eid)
{
	if (eid >= entities.size())
	{
		return false;
	}

	// Delete components
	for (auto& pool : allComponents)
	{
		(*pool).deleteComponent(eid);
	}	

	// Release id back into pool.
	availableEntityIds.insert(eid);
	delete entities[eid];
	entities[eid] = nullptr;

	return true;
}

bool Scene::destroyEntity(uint32_t eid)
{
	return toDestroyEntities.insert(eid).second;
}

std::vector<Entity*> SimpleECS::Scene::getEntities()
{
	std::vector<Entity*> active;
	active.reserve(entities.size());
	for (auto ent : entities) {
		if (ent) {
			active.push_back(ent);
		}
	}
	return active;
}

void Scene::destroyAllMarkedEntities()
{
	for (auto iter = toDestroyEntities.begin(); iter != toDestroyEntities.end();)
	{
		destroyEntityImmediate(*iter);
		iter = toDestroyEntities.erase(iter);
	}
}

std::vector<std::shared_ptr<ComponentPoolBase>>& Scene::getComponentPools()
{
	return allComponents;
}

std::uint32_t Scene::nextEntityID()
{
	if (availableEntityIds.empty())
	{
		return maxID++;
	}
	else
	{
		uint32_t id = *availableEntityIds.begin();
		availableEntityIds.erase(availableEntityIds.begin());
		return id;
	}
}




