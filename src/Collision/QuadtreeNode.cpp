#include "QuadtreeNode.h"

using namespace SimpleECS;

QuadtreeNode::QuadtreeNode(int level, const Collider::AABB& bounds)
    : level(level), bounds(bounds) {
    for (int i = 0; i < 4; ++i) {
        children[i] = nullptr;
    }
}

QuadtreeNode::~QuadtreeNode() {
    clear();
}

void QuadtreeNode::clear() {
    colliders.clear();
    for (int i = 0; i < 4; ++i) {
        if (children[i]) {
            children[i]->clear();
            delete children[i];
            children[i] = nullptr;
        }
    }
}

void QuadtreeNode::subdivide() {
    double subWidth = (bounds.xMax - bounds.xMin) / 2.0;
    double subHeight = (bounds.yMax - bounds.yMin) / 2.0;
    double x = bounds.xMin;
    double y = bounds.yMin;

    children[0] = new QuadtreeNode(level + 1, { x + subWidth, x + 2 * subWidth, y + subHeight, y + 2 * subHeight });
    children[1] = new QuadtreeNode(level + 1, { x, x + subWidth, y + subHeight, y + 2 * subHeight });
    children[2] = new QuadtreeNode(level + 1, { x, x + subWidth, y, y + subHeight });
    children[3] = new QuadtreeNode(level + 1, { x + subWidth, x + 2 * subWidth, y, y + subHeight });
}

int QuadtreeNode::getIndex(const Collider::AABB& colliderBounds) {
    int index = -1;
    double verticalMidpoint = bounds.xMin + (bounds.xMax - bounds.xMin) / 2.0;
    double horizontalMidpoint = bounds.yMin + (bounds.yMax - bounds.yMin) / 2.0;

    bool topQuadrant = (colliderBounds.yMin >= horizontalMidpoint);
    bool bottomQuadrant = (colliderBounds.yMax <= horizontalMidpoint);

    if (colliderBounds.xMax <= verticalMidpoint) {
        if (topQuadrant) {
            index = 1;
        } else if (bottomQuadrant) {
            index = 2;
        }
    } else if (colliderBounds.xMin >= verticalMidpoint) {
        if (topQuadrant) {
            index = 0;
        } else if (bottomQuadrant) {
            index = 3;
        }
    }

    return index;
}

void QuadtreeNode::insert(Collider* collider) {
    if (children[0] != nullptr) {
        int index = getIndex(collider->getAABB());

        if (index != -1) {
            children[index]->insert(collider);
            return;
        }
    }

    colliders.push_back(collider);

    if (colliders.size() > MAX_OBJECTS && level < MAX_LEVELS) {
        if (children[0] == nullptr) {
            subdivide();
        }

        auto it = colliders.begin();
        while (it != colliders.end()) {
            int index = getIndex((*it)->getAABB());
            if (index != -1) {
                children[index]->insert(*it);
                it = colliders.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void QuadtreeNode::retrieve(const Collider::AABB& area, std::vector<Collider*>& returnedColliders) {
    int index = getIndex(area);
    if (index != -1 && children[0] != nullptr) {
        children[index]->retrieve(area, returnedColliders);
    }

    returnedColliders.insert(returnedColliders.end(), colliders.begin(), colliders.end());
}
