#include "Core/Game.h"
#include "ColliderGrid.h"
#include "../Core/GameRenderer.h"
#include "Core/Entity.h"
#include <vector>

using namespace SimpleECS;
using namespace UtilSimpleECS;

ColliderGrid::ColliderGrid(const int w, const int h)
{
	cellWidth = w;
	cellHeight = h;

	numRow = static_cast<int>(ceil(GameRenderer::SCREEN_HEIGHT / (double)cellHeight));
	numColumn = static_cast<int>(ceil(GameRenderer::SCREEN_WIDTH / (double)cellWidth));

	grid.resize(numRow * numColumn + 1); // Last index represents out of bounds cell
	cellBounds.resize(numRow * numColumn + 1); 
	boxPool = Game::getInstance().getCurrentScene()->getComponents<BoxCollider>();
}

void SimpleECS::ColliderGrid::populateGrid()
{
	for (auto& collide : *boxPool)
	{
		insertToGrid(&collide);
	}
}

constexpr const int& clamp(const int& v, const int& lo, const int& hi)
{
	if (v < lo) { return lo; }
	if (v > hi) { return hi; }
	return v;
}

void SimpleECS::ColliderGrid::insertToGrid(Collider* collider)
{
	if (collider->entity == NULL) return;

	Collider::AABB bound;
	collider->getBounds(bound);

	// Get the left most column index this collider exists in, rightMost, etc.
	int columnLeft	= static_cast<int>((bound.xMin + GameRenderer::SCREEN_WIDTH / 2.0) / cellWidth);
	int columnRight = static_cast<int>((bound.xMax + GameRenderer::SCREEN_WIDTH / 2.0) / cellWidth);
	int rowTop		= static_cast<int>((-bound.yMin + GameRenderer::SCREEN_HEIGHT / 2.0) / cellHeight);
	int rowBottom	= static_cast<int>((-bound.yMax + GameRenderer::SCREEN_HEIGHT / 2.0) / cellHeight);

	int colLeftClamped	= clamp(columnLeft, 0, numColumn - 1);
	int colRightClamped = clamp(columnRight, 0, numColumn - 1);
	int rowBotClamped	= clamp(rowBottom, 0, numRow - 1);
	int rowTopClamped	= clamp(rowTop, 0, numRow - 1);

	// Add to cells this object potentially resides in
	for (int r = rowBotClamped; r <= rowTopClamped; ++r)
	{
		for (int c = colLeftClamped; c <= colRightClamped; ++c)
		{
			// Get effective index
			int index = r * numColumn + c;
			grid[index].insert(collider);
		}
	}

	// If resides in no cells, add to out of bounds
	if (columnLeft != colLeftClamped || columnRight != colRightClamped
		|| rowTop != rowTopClamped || rowBottom != rowBotClamped)
	{
		grid.back().insert(collider);
	}
}

void SimpleECS::ColliderGrid::updateGrid()
{
	// Add to cells this object resides in
	Collider::AABB cellBound;
	Collider::AABB colliderBound;

	// Remove collider reference in each cell if collider no longer inhabits cell
	for (int i = 0; i < grid.size(); ++i)
	{
		if (grid[i].size() == 0) continue;
		getCellBounds(cellBound, i);
		cellBounds[i] = cellBound;
		for (auto colliderIter = grid[i].begin(); colliderIter != grid[i].end();)
		{
			// If not in this cell, remove reference
			(*colliderIter)->getBounds(colliderBound);
			if (colliderBound.xMin > cellBound.xMax || colliderBound.xMax < cellBound.xMin
				|| colliderBound.yMax < cellBound.yMin || colliderBound.yMin > cellBound.yMax)
			{
				colliderIter = grid[i].erase(colliderIter);
			}
			else
			{
				colliderIter++;
			}
		}
	}
	populateGrid();
}

size_t SimpleECS::ColliderGrid::size() const
{
	return grid.size();
}

int SimpleECS::ColliderGrid::cellSize(const int index)
{
	return 0;
}

const ColliderCell* ColliderGrid::getCellContents(const int index) const
{
	return &grid[index];
}

const ColliderCell* ColliderGrid::getOutBoundContent() const
{
	return getCellContents(static_cast<int>(size() - 1));
}

void SimpleECS::ColliderGrid::getCellBounds(Collider::AABB& output, const int index)
{
	// index = row * numColumn + c
	int column = index % numColumn;
	int row = (index - column) / numColumn;

	output.xMin = -GameRenderer::SCREEN_WIDTH / 2 + column * cellWidth;
	output.xMax = output.xMin + cellWidth;

	output.yMax = GameRenderer::SCREEN_HEIGHT / 2 - row * cellHeight;
	output.yMin = output.yMax - cellHeight;
}

std::vector<ColliderCell> *SimpleECS::ColliderGrid::getRawGrid()
{
    return &grid;
}
