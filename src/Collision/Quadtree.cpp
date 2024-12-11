#include "Quadtree.h"

using namespace SimpleECS;

Quadtree::Quadtree(const Collider::AABB& bounds, int maxObjects, int maxLevels)
    : maxObjects(maxObjects), maxLevels(maxLevels)
{
    rootBounds = bounds;
    rootIndex = createNode(0, rootBounds);

    nodes.reserve(100000);
    allCells.reserve(100000);
}

int Quadtree::createNode(int level, const Collider::AABB& bounds) {
    Node node;
    node.level = level;
    node.bounds = bounds;
    for (int i = 0; i < 4; i++) {
        node.children[i] = -1;
    }

    // Add a new cell for this node
    allCells.emplace_back();
    node.cellIndex = static_cast<int>(allCells.size()) - 1;

    nodes.push_back(node);
    return static_cast<int>(nodes.size()) - 1;
}

void Quadtree::clear() {
    // Reset all data
    nodes.clear();
    allCells.clear();

    // Re-create root
    rootIndex = createNode(0, rootBounds);
}

void Quadtree::clearNode(int nodeIndex) {
    if (nodeIndex < 0) return;
    Node& node = nodes[nodeIndex];

    // Reset the cell at this node
    allCells[node.cellIndex] = ColliderCell();

    // Clear children
    for (int i = 0; i < 4; ++i) {
        int c = node.children[i];
        if (c != -1) {
            clearNode(c);
            node.children[i] = -1;
        }
    }
}

void Quadtree::subdivide(int nodeIndex) {
    Node& node = nodes[nodeIndex];

    double subWidth = (node.bounds.xMax - node.bounds.xMin) / 2.0;
    double subHeight = (node.bounds.yMax - node.bounds.yMin) / 2.0;
    double x = node.bounds.xMin;
    double y = node.bounds.yMin;

    // top-right
    node.children[0] = createNode(node.level + 1, { x + subWidth, y + subHeight, x + 2 * subWidth, y + 2 * subHeight });
    // top-left
    node.children[1] = createNode(node.level + 1, { x, y + subHeight, x + subWidth, y + 2 * subHeight });
    // bottom-left
    node.children[2] = createNode(node.level + 1, { x, y, x + subWidth, y + subHeight });
    // bottom-right
    node.children[3] = createNode(node.level + 1, { x + subWidth, y, x + 2 * subWidth, y + subHeight });
}

void Quadtree::insert(Collider* collider) {
    insertAtNode(rootIndex, collider);
}

void Quadtree::insertAtNode(int nodeIndex, Collider* collider) {
    Node& node = nodes[nodeIndex];

    Collider::AABB cb;
    collider->getBounds(cb);

    // If children exist, try to put collider into any that intersect
    if (node.children[0] != -1) {
        double verticalMidpoint = node.bounds.xMin + (node.bounds.xMax - node.bounds.xMin) / 2.0;
        double horizontalMidpoint = node.bounds.yMin + (node.bounds.yMax - node.bounds.yMin) / 2.0;

        Collider::AABB topRight { verticalMidpoint, horizontalMidpoint, node.bounds.xMax, node.bounds.yMax };
        Collider::AABB topLeft  { node.bounds.xMin, horizontalMidpoint, verticalMidpoint, node.bounds.yMax };
        Collider::AABB bottomLeft { node.bounds.xMin, node.bounds.yMin, verticalMidpoint, horizontalMidpoint };
        Collider::AABB bottomRight { verticalMidpoint, node.bounds.yMin, node.bounds.xMax, horizontalMidpoint };

        // Insert collider into intersecting child nodes
        if (isIntersecting(cb, topRight)) {
            insertAtNode(node.children[0], collider);
        }
        if (isIntersecting(cb, topLeft)) {
            insertAtNode(node.children[1], collider);
        }
        if (isIntersecting(cb, bottomLeft)) {
            insertAtNode(node.children[2], collider);
        }
        if (isIntersecting(cb, bottomRight)) {
            insertAtNode(node.children[3], collider);
        }

        // Do not store collider in this node
        return;
    }

    // Store collider in this node's cell if it is a leaf
    ColliderCell& cell = allCells[node.cellIndex];
    cell.insert(collider);

    // If we exceed max objects and we are not at max level, subdivide and redistribute
    if (cell.size() > static_cast<size_t>(maxObjects) && node.level < maxLevels) {
        if (node.children[0] == -1) {
            subdivide(nodeIndex);
        }

        double verticalMidpoint = node.bounds.xMin + (node.bounds.xMax - node.bounds.xMin) / 2.0;
        double horizontalMidpoint = node.bounds.yMin + (node.bounds.yMax - node.bounds.yMin) / 2.0;

        Collider::AABB topRight { verticalMidpoint, horizontalMidpoint, node.bounds.xMax, node.bounds.yMax };
        Collider::AABB topLeft  { node.bounds.xMin, horizontalMidpoint, verticalMidpoint, node.bounds.yMax };
        Collider::AABB bottomLeft { node.bounds.xMin, node.bounds.yMin, verticalMidpoint, horizontalMidpoint };
        Collider::AABB bottomRight { verticalMidpoint, node.bounds.yMin, node.bounds.xMax, horizontalMidpoint };

        for (auto it = cell.begin(); it != cell.end();) {
            Collider* c = *it;
            Collider::AABB cBounds;
            c->getBounds(cBounds);

            bool moved = false;
            if (isIntersecting(cBounds, topRight)) {
                insertAtNode(node.children[0], c);
                moved = true;
            }
            if (isIntersecting(cBounds, topLeft)) {
                insertAtNode(node.children[1], c);
                moved = true;
            }
            if (isIntersecting(cBounds, bottomLeft)) {
                insertAtNode(node.children[2], c);
                moved = true;
            }
            if (isIntersecting(cBounds, bottomRight)) {
                insertAtNode(node.children[3], c);
                moved = true;
            }

            it = cell.erase(it);
        }
    }
}