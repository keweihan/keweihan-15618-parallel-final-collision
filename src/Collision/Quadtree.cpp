#include "Quadtree.h"

using namespace SimpleECS;

Quadtree::Quadtree(const Collider::AABB& bounds, int maxObjects, int maxLevels)
    : maxObjects(maxObjects), maxLevels(maxLevels), rootBounds(bounds)
{
    nodes.reserve(100000);
    allCells.reserve(100000);

    rootIndex = createNode(0, rootBounds);
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
    nodes.clear();
    allCells.clear();
    rootIndex = createNode(0, rootBounds);
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

int Quadtree::getQuadrant(const Collider::AABB& cBounds, const Collider::AABB& nodeBounds) const {
    double verticalMidpoint = nodeBounds.xMin + (nodeBounds.xMax - nodeBounds.xMin) / 2.0;
    double horizontalMidpoint = nodeBounds.yMin + (nodeBounds.yMax - nodeBounds.yMin) / 2.0;

    bool fitsTop = (cBounds.yMin >= horizontalMidpoint);
    bool fitsBottom = (cBounds.yMax <= horizontalMidpoint);
    bool fitsLeft = (cBounds.xMax <= verticalMidpoint);
    bool fitsRight = (cBounds.xMin >= verticalMidpoint);

    // Quadrants:
    // 0: top-right, 1: top-left, 2: bottom-left, 3: bottom-right
    if (fitsRight && fitsTop) {
        return 0; // top-right
    } else if (fitsLeft && fitsTop) {
        return 1; // top-left
    } else if (fitsLeft && fitsBottom) {
        return 2; // bottom-left
    } else if (fitsRight && fitsBottom) {
        return 3; // bottom-right
    }

    // Doesn't fit exclusively in one quadrant
    return -1;
}

void Quadtree::insertAtNode(int nodeIndex, Collider* collider) {
    Node& node = nodes[nodeIndex];
    ColliderCell& cell = allCells[node.cellIndex];

    Collider::AABB cb;
    collider->getBounds(cb);

    // If children exist, try placing collider in a child
    if (node.children[0] != -1) {
        int quadrant = getQuadrant(cb, node.bounds);
        if (quadrant != -1) {
            insertAtNode(node.children[quadrant], collider);
            return;
        }
    }

    // Otherwise, store collider in this node's cell
    cell.insert(collider);

    // If we exceed max objects and we are not at max level, subdivide and redistribute
    if (cell.size() > static_cast<size_t>(maxObjects) && node.level < maxLevels) {
        if (node.children[0] == -1) {
            subdivide(nodeIndex);
        }

        for (auto it = cell.begin(); it != cell.end();) {
            Collider* c = *it;
            Collider::AABB cBounds;
            c->getBounds(cBounds);

            int quadrant = getQuadrant(cBounds, node.bounds);
            if (quadrant != -1) {
                insertAtNode(node.children[quadrant], c);
                it = cell.erase(it);
            } else {
                ++it;
            }
        }
    }
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
