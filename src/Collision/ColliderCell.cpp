#include "ColliderCell.h"
#include "boost/container/static_vector.hpp"

using namespace SimpleECS;

SimpleECS::ColliderCell::ColliderCell(const ColliderCell& other)
{
    colList = other.colList;
}

ColliderCell::ColliderCell(int defaultSize)
{
    colList.reserve(defaultSize);
}

ColliderCell::ColliderCell()
{
    colList.reserve(10);
}

ColliderCell::~ColliderCell() {}

ColliderCell& SimpleECS::ColliderCell::operator=(const ColliderCell& other)
{
    colList = other.colList;
    return *this;
}

size_t ColliderCell::size()
{
    return colList.size();
}

void ColliderCell::insert(Collider* col)
{
    if (find(col) == end()) 
    {
        // Insert to the end
        colList.push_back(col);
    }
}

ColliderCellIterator ColliderCell::erase(ColliderCellIterator o)
{
    // Special case if last element
    if (o - colList.begin() == colList.size() - 1) {
        colList.pop_back();
        return colList.end();
    }

    // Replace element with back element and remove from back
    *o = colList.back();
    colList.pop_back();

    return o;
}

ColliderCellIterator ColliderCell::erase(Collider* col)
{
    return this->erase(find(col));
}

ColliderCellIterator ColliderCell::find(Collider* col)
{
    auto res = colList.end();
    for (auto iter = colList.begin(); iter != colList.end(); ++iter)
    {
        if (*iter == col) { return iter;  }
    }
    return res;
}

ColliderConstCellIterator ColliderCell::begin() const
{
    return colList.begin();
}

ColliderCellIterator ColliderCell::begin()
{
    return colList.begin();
}

Collider* ColliderCell::back()
{
    return colList.back();
}

ColliderCellIterator ColliderCell::end()
{
    return colList.end();
}

ColliderConstCellIterator ColliderCell::end() const
{
    return colList.end();
}