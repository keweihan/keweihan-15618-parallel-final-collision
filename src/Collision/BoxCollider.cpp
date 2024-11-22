#include "Collision/BoxCollider.h"
#include "Physics/PhysicsBody.h"
#include "ColliderSystem.h"
#include "Core/Entity.h"

using namespace SimpleECS;

void SimpleECS::BoxCollider::update()
{
    Vector position = entity->transform->position;

    bound.xMin = position.x - width / 2;
    bound.xMax = position.x + width / 2;

    bound.yMin = position.y - height / 2;
    bound.yMax = position.y + height / 2;
}

bool BoxCollider::isColliding(Collider* other)
{
    // STUB
    //return ColliderSystem::getCollisionInfo(new Collide);
    return false;
}

void SimpleECS::BoxCollider::getBounds(Collider::AABB& bounds) const
{
    bounds = bound;
}
