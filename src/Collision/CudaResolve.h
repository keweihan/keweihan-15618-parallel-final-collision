#pragma once


#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "Core/Entity.h"
#include "ColliderCell.h"
#include "Collision/BoxCollider.h"


// Plain Old Data (POD) structure
struct ColliderEntity {
    ColliderEntity() = default;
    ColliderEntity(int eid) {}; // stub - 
    ColliderEntity(SimpleECS::BoxCollider* col);

    friend std::ostream& operator<<(std::ostream& os, const ColliderEntity& entity);

    // entity id
    double mass;

    // phys params
    double x_vel, y_vel, x_pos, y_pos;
    double fx_vel, fy_vel;

    // aabb
    double x_min, x_max, y_min, y_max;

    // w/h
    int w, h;
    int eid;

    // is it a static physics object (i.e. doesn't move)
    bool is_static;

    // has this entity changed after entering->leaving cuda kernel?
    bool is_dirty;
};

// Complex Class
class CudaResolve {
public:
    CudaResolve(std::vector<SimpleECS::ColliderCell>* cells) : _cells(cells) {}

    /// @brief Flatten all collider cells and copy to CUDA device
    void flattenCopyToDevice();

    /// @brief Launch CUDA kernel
    void kernelResolvePhysics(int numThreads);

    ~CudaResolve();

private:
    std::vector<SimpleECS::ColliderCell>* _cells;

    // Device pointers
    ColliderEntity* d_flattenedData = nullptr;
    int* d_lengths = nullptr;
    int* d_offsets = nullptr;
    int numCells = 0;

    int num_references = 0;

    /// @brief Helper function to allocate and copy data
    void allocateAndCopyToDevice(const std::vector<ColliderEntity>& flattenedData, 
                                 const std::vector<int>& lengths, 
                                 const std::vector<int>& offsets);
};