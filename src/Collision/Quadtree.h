#pragma once
#include "QuadtreeNode.h"
#include "ColliderCell.h"

namespace SimpleECS {

class Quadtree {
public:
    Quadtree(const Collider::AABB& bounds);
    ~Quadtree();

    void clear();
    void insert(Collider* collider);
    void retrievePotentialCollisions(Collider* collider, std::vector<Collider*>& potentialColliders);

private:
    QuadtreeNode* root;
};

}
