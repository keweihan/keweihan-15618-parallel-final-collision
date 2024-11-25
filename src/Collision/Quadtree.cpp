#include "Quadtree.h"

using namespace SimpleECS;

Quadtree::Quadtree(const Collider::AABB& bounds) {
    root = new QuadtreeNode(0, bounds);
}

Quadtree::~Quadtree() {
    delete root;
}

void Quadtree::clear() {
    root->clear();
}

void Quadtree::insert(Collider* collider) {
    root->insert(collider);
}

void Quadtree::retrievePotentialCollisions(Collider* collider, std::vector<Collider*>& potentialColliders) {
    root->retrieve(collider->getAABB(), potentialColliders);
}
