#pragma once


#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "Core/Entity.h"
#include "ColliderCell.h"
#include "Collision/BoxCollider.h"


// Plain Old Data (POD) structure
struct ColliderEntity {
    ColliderEntity(int eid) {}; // stub - 
    ColliderEntity(SimpleECS::BoxCollider* col);
    // entity id
    int eid;

    // phys params
    double x_vel, y_vel, x_pos, y_pos;

    // aabb
    double x_min, x_max, y_min, y_max;    
};

// Complex Class
class CudaResolve {
public:
    CudaResolve(std::vector<SimpleECS::ColliderCell>* cells) : _cells(cells) {}

    /// @brief Flatten all collider cells and copy to CUDA device
    void flattenCopyToDevice();

    /// @brief Launch CUDA kernel
    void launchKernel(int numThreads);

    ~CudaResolve();

private:
    std::vector<SimpleECS::ColliderCell>* _cells;

    // Device pointers
    ColliderEntity* d_flattenedData = nullptr;
    int* d_lengths = nullptr;
    int* d_offsets = nullptr;

    /// @brief Helper function to allocate and copy data
    void allocateAndCopyToDevice(const std::vector<ColliderEntity>& flattenedData, 
                                 const std::vector<int>& lengths, 
                                 const std::vector<int>& offsets);
};