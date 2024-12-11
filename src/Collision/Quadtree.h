#pragma once

#include "Collider.h"
#include "ColliderCell.h"
#include <vector>

namespace SimpleECS {

class Quadtree {
public:
    Quadtree(const Collider::AABB& bounds, int maxObjects = 5, int maxLevels = 5);

    void clear();
    void insert(Collider* collider);

    // Optionally, provide direct access to allCells if needed
    std::vector<ColliderCell>* getCells() { return &allCells; }

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

    // Determines if two AABBs intersect
    bool isIntersecting(const Collider::AABB& a, const Collider::AABB& b) {
        return (a.xMin <= b.xMax && a.xMax >= b.xMin &&
                a.yMin <= b.yMax && a.yMax >= b.yMin);
    }

    // Determines which child quadrant a collider's bounds fit into
    // Returns the child index [0-3], or -1 if it doesn't fit into a single quadrant.
    int getQuadrant(const Collider::AABB& cBounds, const Collider::AABB& nodeBounds) const;

    // Quadtree parameters
    int maxObjects;
    int maxLevels;

    // Storage for nodes and cells
    std::vector<Node> nodes;
    std::vector<ColliderCell> allCells; // Stores collider lists for each node

    // The root node index and bounds
    int rootIndex;
    Collider::AABB rootBounds;
};

}
