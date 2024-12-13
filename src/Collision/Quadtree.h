#pragma once
#include "QuadtreeNode.h"
#include "ColliderCell.h"

namespace SimpleECS {

class Quadtree {
public:
    Quadtree(const Collider::AABB& bounds, int maxObjects = 5, int maxLevels = 5);

    void clear();
    void insert(Collider* collider);
    void reserve(int count); // number of references

    std::vector<ColliderCell>* getCells() { return &allCells; }
    std::vector<Collider::AABB>* getCellBounds() { return &cellBounds; }
    
private:
    struct Node {
        int level;
        Collider::AABB bounds;
        int children[4];     // -1 if no child
        int cellIndex;       // Index into allCells
    };

    int createNode(int level, const Collider::AABB& bounds);
    void clearNode(int nodeIndex);
    void subdivide(int nodeIndex);
    void insertAtNode(int nodeIndex, Collider* collider);
    bool isIntersecting(const Collider::AABB& a, const Collider::AABB& b) {
        return (a.xMin <= b.xMax && a.xMax >= b.xMin &&
                a.yMin <= b.yMax && a.yMax >= b.yMin);
    }
    

    // Quadtree parameters
    int maxObjects;
    int maxLevels;

    // Storage
    std::vector<Node> nodes;
    std::vector<ColliderCell> allCells; // Global collider cells storage
    std::vector<Collider::AABB> cellBounds;

    // The root node index
    int rootIndex;
    Collider::AABB rootBounds;
};

}
