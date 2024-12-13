#include "ColliderSystem.h"
#include "Collision/Collider.h"
#include "Collision/BoxCollider.h"
#include "Core/Component.h"
#include "Core/Entity.h"
#include "Core/GameRenderer.h"
#include "Utility/TransformUtil.h"
#include "Utility/ThreadCount.h"
#include "boost/functional/hash.hpp"
#include <vector>
#include <thread>
#include <iostream>
#include "CudaResolve.h"
#include <cuda_runtime.h>
#include "SDL.h"

using namespace SimpleECS;
using namespace UtilSimpleECS;

const int QUAD_MAX_CELL =  20;
const int QUAD_MAX_LEVELS =  10;
 
const int GRID_WIDTH =  26;
const int GRID_HEIGHT =  26;

const int CELL_LOG_FREQUENCY = 5000;

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
      }, QUAD_MAX_CELL, QUAD_MAX_LEVELS), colliderGrid(ColliderGrid(GRID_WIDTH, GRID_HEIGHT))
{}

void logPrintThreads(int t) {
	UtilSimpleECS::ThreadCount::logThreads(t);
	if(UtilSimpleECS::ThreadCount::timeSinceGetAvg() > CELL_LOG_FREQUENCY)
	{
		std::cout << "Avg number of threads over past "; 
		std::cout <<  CELL_LOG_FREQUENCY/1000 << " seconds: ";
		std::cout << UtilSimpleECS::ThreadCount::getAvgThreads() << std::endl;
	}
}

void SimpleECS::ColliderSystem::invokeCollisions()
{
	quadtree.clear();
	Collision collision = {};
	switch (scheme) {
	case STATIC_GRID_SEQ: {
		colliderGrid.updateGrid();
		logPrintThreads(1);
		// Populate with potential pairs
		try {
			for (int i = 0; i < colliderGrid.size(); ++i)
			{
				auto cell = *colliderGrid.getCellContents(i);
				for (auto iterA = cell.begin(); iterA != cell.end(); ++iterA)
				{
					for (auto iterB = iterA + 1; iterB != cell.end(); ++iterB)
					{
						//potentialPairs.insert({ *iterA, *iterB });

						_invokeCollision(collision, (*iterA), (*iterB));
						_invokeCollision(collision, (*iterB), (*iterA));
					}
				}
			}
		}
		catch (const std::exception& e) {
			std::cerr << "Exception occurred while populating potential pairs: " << e.what() << std::endl;
		}
	}
		break;
	case STATIC_GRID_CUDA: {
		colliderGrid.updateGrid();
		logPrintThreads(colliderGrid.getRawGrid()->size());
		CudaResolve resolver(colliderGrid.getRawGrid());
		resolver.flattenCopyToDevice();
		resolver.launchKernel(10);
	}
		break;
	case QUADTREE_SEQ: {
		auto colliders = Game::getInstance().getCurrentScene()->getComponents<BoxCollider>();
		quadtree.reserve(10000000);
		for (auto& collider : *colliders) {
		    quadtree.insert(&collider);
		}
		logPrintThreads(1);

		auto cells = quadtree.getCells();
		for (auto& cell : *cells) {
			for (auto iterA = cell.begin(); iterA != cell.end(); ++iterA)
			{
				for (auto iterB = iterA + 1; iterB != cell.end(); ++iterB)
				{
					collision.a = *iterA;
					collision.b = *iterB;

					_invokeCollision(collision, collision.a, collision.b);
					_invokeCollision(collision, collision.b, collision.a);
				}
			}
		}
	}
		break;
	case QUADTREE_CUDA: {
		auto colliders = Game::getInstance().getCurrentScene()->getComponents<BoxCollider>();
		quadtree.reserve(10000000);
		for (auto& collider : *colliders) {
		    quadtree.insert(&collider);
		}
		logPrintThreads(quadtree.getCells()->size());
		CudaResolve resolver(quadtree.getCells());
		resolver.flattenCopyToDevice();
		resolver.launchKernel(10);
	}
		break;
	}
	visualizeCells();
}

void SimpleECS::ColliderSystem::visualizeCells() {
	std::vector<Collider::AABB>* grid;
	if(scheme == QUADTREE_CUDA || scheme == QUADTREE_SEQ) {
		grid = quadtree.getCellBounds();
	}
	else {
		grid = colliderGrid.getCellBounds();
	}
	
	for(const auto& bound : *grid)
	{
		Vector blCorner = TransformUtil::worldToScreenSpace(bound.xMin, bound.yMin);
		Vector trCorner = TransformUtil::worldToScreenSpace(bound.xMax, bound.yMax);

		// After transforming to screen space, let's ensure the corners are consistent
		float left   = std::min(blCorner.x, trCorner.x);
		float right  = std::max(blCorner.x, trCorner.x);
		float top    = std::min(blCorner.y, trCorner.y);
		float bottom = std::max(blCorner.y, trCorner.y);

		// Now define corners in a consistent top-left based coordinate system:
		// Remember SDL's origin is top-left, so 'top' is the smaller y-value and 'bottom' is the larger y-value
		SDL_FPoint points[] = {
			{left,  bottom}, // Bottom-left
			{left,  top},    // Top-left
			{right, top},    // Top-right
			{right, bottom}, // Bottom-right
			{left,  bottom}  // Close the loop back to Bottom-left
		};

		// Now draw with consistent ordering
		SDL_SetRenderDrawColor(GameRenderer::renderer, 102, 102, 102, 255);
		SDL_RenderDrawLinesF(GameRenderer::renderer, points, 5);
	}
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

