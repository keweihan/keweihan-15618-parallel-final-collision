#include "ColliderSystem.h"
#include "Collision/Collider.h"
#include "Collision/BoxCollider.h"
#include "Core/Component.h"
#include "Core/Entity.h"
#include "Core/GameRenderer.h"
#include "Utility/TransformUtil.h"
#include "boost/functional/hash.hpp"
#include <vector>
#include <thread>
#include <iostream>
#include "CudaResolve.h"
#include <cuda_runtime.h>

using namespace SimpleECS;
using namespace UtilSimpleECS;

//------------------- Collision invocation ---------------------//

template<typename T1, typename T2>
struct PairHash {
	std::size_t operator()(const std::pair<T1, T2>& p) const {
		std::size_t seed = 0;
		boost::hash_combine(seed, boost::hash<T1>()(p.first));
		boost::hash_combine(seed, boost::hash<T2>()(p.second));
		return seed;
	}
};

inline void _invokeCollision(Collision& collision, Collider* a, Collider* b)
{
	collision.a = a;
	collision.b = b;

	// TODO: getComponents is an expensive operation. 
	if (ColliderSystem::getInstance().getCollisionInfo(collision)) {
		for (auto component : collision.a->entity->getComponents())
		{
			component->onCollide(*collision.b);
			component->onCollide(collision);
		}
	}
}

ColliderSystem::ColliderSystem()
    : quadtree(Collider::AABB{
          -GameRenderer::SCREEN_WIDTH / 2.0,
		  -GameRenderer::SCREEN_HEIGHT / 2.0,
          GameRenderer::SCREEN_WIDTH / 2.0,
          GameRenderer::SCREEN_HEIGHT / 2.0,
      }), colliderGrid(ColliderGrid(16, 16))
{}

void SimpleECS::ColliderSystem::invokeCollisions()
{
	quadtree.clear();
	Collision collision = {};

	// --------------------- QUADTREE SEQUENTIAL --------------------- //
	// // Get all colliders from the current scene
    // auto colliders = Game::getInstance().getCurrentScene()->getComponents<BoxCollider>();
	// for (auto& collider : *colliders) {
    //     quadtree.insert(&collider);
    // }

	// auto cells = quadtree.getCells();
	// for (auto& cell : *cells) {
	// 	for (auto iterA = cell.begin(); iterA != cell.end(); ++iterA)
	// 	{
	// 		for (auto iterB = iterA + 1; iterB != cell.end(); ++iterB)
	// 		{
	// 			collision.a = *iterA;
	// 			collision.b = *iterB;

	// 			_invokeCollision(collision, collision.a, collision.b);
	// 			_invokeCollision(collision, collision.b, collision.a);
	// 		}
	// 	}
	// }
	// --------------------- QUADTREE SEQUENTIAL --------------------- //

	// --------------------- STATIC GRID SEQUENTIAL --------------------- //
	// Set of potential collision pairs
	// std::unordered_set<std::pair<Collider*, Collider*>, PairHash<Collider*, Collider*>>
	// 	potentialPairs;

	// // Populate with potential pairs
	// try {
	
	// 	for (int i = 0; i < colliderGrid.size(); ++i)
	// 	{
	// 		auto cell = *colliderGrid.getCellContents(i);
	// 		for (auto iterA = cell.begin(); iterA != cell.end(); ++iterA)
	// 		{
	// 			for (auto iterB = iterA + 1; iterB != cell.end(); ++iterB)
	// 			{
	// 				//potentialPairs.insert({ *iterA, *iterB });

	// 				_invokeCollision(collision, (*iterA), (*iterB));
	// 				_invokeCollision(collision, (*iterB), (*iterA));
	// 			}
	// 		}
	// 	}
	// }
	// catch (const std::exception& e) {
	// 	std::cerr << "Exception occurred while populating potential pairs: " << e.what() << std::endl;
	// }
	// --------------------- STATIC GRID SEQUENTIAL --------------------- //

	// --------------------- STATIC GRID CUDA --------------------- //
	// CudaResolve resolver(colliderGrid.getRawGrid());
	// resolver.flattenCopyToDevice();
	// resolver.launchKernel(10);
	// ---------------------- END STATIC GRID CUDA ---------------------- //

	// --------------------- QUADTREE CUDA --------------------- //
	auto colliders = Game::getInstance().getCurrentScene()->getComponents<BoxCollider>();
	for (auto& collider : *colliders) {
        quadtree.insert(&collider);
    }

	CudaResolve resolver(quadtree.getCells());
	resolver.flattenCopyToDevice();
	resolver.launchKernel(10);
	// ---------------------- END QUADTREE CUDA ---------------------- //



}

bool SimpleECS::ColliderSystem::getCollisionBoxBox(Collision& collide, BoxCollider* a, BoxCollider* b)
{
	if (collide.a == nullptr || collide.b == nullptr) return false;

	Transform& aTransform = *(collide.a->entity->transform);
	Transform& bTransform = *(collide.b->entity->transform);

	// TODO: dynamic casts are expensive. Figure out a better way.
	BoxCollider* aBox = static_cast<BoxCollider*>(collide.a);
	BoxCollider* bBox = static_cast<BoxCollider*>(collide.b);

	if (bBox != nullptr && aBox != nullptr)
	{
		// AABB Collision 
		Collider::AABB aBounds;
		Collider::AABB bBounds;
		aBox->getBounds(aBounds);
		bBox->getBounds(bBounds);

		//If any of the sides from A are outside of B, no collision occuring.
		double epsilon = 0.01;  // Small margin value
		if (aBounds.yMin >= bBounds.yMax - epsilon || aBounds.yMax <= bBounds.yMin + epsilon
			|| aBounds.xMax <= bBounds.xMin + epsilon || aBounds.xMin >= bBounds.xMax - epsilon)
		{
			return false;
		}

		// Boxes are colliding. Find axis of least penetration
		double aExtentX = aBox->width / 2.0;
		double bExtentX = bBox->width / 2.0;
		double aExtentY = aBox->height / 2.0;
		double bExtentY = bBox->height / 2.0;

		double xDistance = std::abs(aTransform.position.x - bTransform.position.x);
		double xOverlap = (aExtentX + bExtentX) - xDistance;

		double yDistance = std::abs(aTransform.position.y - bTransform.position.y);
		double yOverlap = (aExtentY + bExtentY) - yDistance;

		//if ((yOverlap < xOverlap ? yOverlap : xOverlap) == 0) {
		//	return false;
		//}

		// Least penetration is on y-axis
		if (yOverlap < xOverlap)
		{
			collide.penetration = yOverlap;
			if (aTransform.position.y < bTransform.position.y)
			{
				collide.normal = Vector(0, -1);
			}
			else
			{
				collide.normal = Vector(0, 1);
			}
		}
		// Least penetration is on x-axis
		else
		{
			collide.penetration = xOverlap;
			if (aTransform.position.x < bTransform.position.x)
			{
				collide.normal = Vector(-1, 0);
			}
			else
			{
				collide.normal = Vector(1, 0);
			}
		}
	}


	return true;
}


bool SimpleECS::ColliderSystem::getCollisionInfo(Collision& collide)
{
	if (collide.a == nullptr || collide.b == nullptr) return false;

	// Breaking principles of polymorphism (likely) necessary. 
	// Different collider collisions (i.e. sphere-sphere, sphere-box, box-box) 
	// require different implementation.
	BoxCollider* boxA = dynamic_cast<BoxCollider*>(collide.a);
	BoxCollider* boxB = dynamic_cast<BoxCollider*>(collide.b);

	// AABB collision
    if (boxA != nullptr && boxB != nullptr)
    {
		return getCollisionBoxBox(collide, boxA, boxB);
    }
	// Other collider types here
	// else if(sphere-sphere...)

    return false;
}

