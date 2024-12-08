#include "Quadtree.h"

using namespace SimpleECS;

Quadtree::Quadtree(const Collider::AABB& bounds) {
    root = new QuadtreeNode(0, bounds);
}

Quadtree::~Quadtree() {
    clear();
    delete root;
}

void Quadtree::clear() {
    if (root) {
        root->clear();
    }
}

void Quadtree::insert(Collider* collider) {
    root->insert(collider);
}

void Quadtree::retrievePotentialCollisions(Collider* collider, std::vector<Collider*>& potentialColliders) {
    Collider::AABB colliderBounds;
    collider->getBounds(colliderBounds);

    root->retrieve(colliderBounds, potentialColliders);
}
