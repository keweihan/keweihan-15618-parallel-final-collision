#pragma once

#include <vector>
#include "Collision/Collider.h"
#include "ColliderCell.h" 

namespace SimpleECS {

class QuadtreeNode {
public:
    QuadtreeNode(int level, const Collider::AABB& bounds);
    ~QuadtreeNode();

    void clear();
    void insert(Collider* collider);
    void retrieve(const Collider::AABB& area, std::vector<Collider*>& returnedColliders);

private:
    void subdivide();
    int getIndex(const Collider::AABB& colliderBounds);

    static const int MAX_OBJECTS = 5;
    static const int MAX_LEVELS = 5;

    int level;
    Collider::AABB bounds;
    QuadtreeNode* children[4];
    ColliderCell cell;
};

}
