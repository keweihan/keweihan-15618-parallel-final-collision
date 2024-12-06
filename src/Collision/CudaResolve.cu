#include "CudaResolve.h"
#include "ColliderCell.h"
#include "Core/Entity.h"
#include "Physics/PhysicsBody.h"
#include "Collision/BoxCollider.h"
#include <cuda_runtime.h>
#include <iostream>
#include "Core/Game.h"
#include "Core/Scene.h"
#include "Core/CHandle.h"

using namespace SimpleECS;

ColliderEntity::ColliderEntity(SimpleECS::BoxCollider* col) {
    Collider::AABB bounds;
    col->getBounds(bounds);
    eid = col->entity->id;

    x_max = bounds.xMax;
    x_min = bounds.xMin;
    y_min = bounds.yMin;
    y_max = bounds.yMin;
    
    y_pos = col->entity->transform->position.y;
    x_pos = col->entity->transform->position.x;

    y_vel = col->entity->phys->velocity.y;
    x_vel = col->entity->phys->velocity.x;
}

// Flatten and copy to device
void CudaResolve::flattenCopyToDevice() {
    std::vector<ColliderEntity> flattenedData;
    std::vector<int> lengths; // sizes of cells in flattened
    std::vector<int> offsets; // starts of cells in flattened

    flattenedData.reserve(_cells->size());
    lengths.reserve(_cells->size());
    offsets.reserve(_cells->size());
    
    int offset = 0;
    for (ColliderCell& cell : *_cells) {
        if(cell.size() > 0)
        {
            lengths.push_back(cell.size());
            offsets.push_back(offset);

            for(Collider* col : cell) {
                BoxCollider* box = static_cast<BoxCollider*>(col);
                ColliderEntity flatCollider(box);
                flattenedData.push_back(flatCollider);
            }
        }
    }

    numCells = flattenedData.size();
    allocateAndCopyToDevice(flattenedData, lengths, offsets);
}

// Allocate and copy to device
void CudaResolve::allocateAndCopyToDevice(const std::vector<ColliderEntity>& flattenedData, 
                                          const std::vector<int>& lengths, 
                                          const std::vector<int>& offsets) {
    cudaMalloc(&d_flattenedData, flattenedData.size() * sizeof(ColliderEntity));
    cudaMalloc(&d_lengths, lengths.size() * sizeof(int));
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));

    cudaMemcpy(d_flattenedData, flattenedData.data(), flattenedData.size() * sizeof(ColliderEntity), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths.data(), lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
}

// Kernel
__global__ void kernel(ColliderEntity* d_flattenedData, int* d_lengths, int* d_offsets, int numCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCells) {
        int length = d_lengths[idx];
        int offset = d_offsets[idx];

        for (int i = 0; i < length; ++i) {
            ColliderEntity entity_a = d_flattenedData[offset + i];
            
            for (int j = i + 1; j < length; ++j) {
                ColliderEntity entity_b = d_flattenedData[offset + j];
                
                // Resolve physics here
                
            }
        }
    }
}

// Kernel launch
void CudaResolve::launchKernel(int numThreads) {
    // Launch kernel process parallel
    dim3 blockDim(256);
    dim3 gridDim((numCells + blockDim.x - 1) / blockDim.x);
    kernel<<<gridDim, blockDim>>>(d_flattenedData, d_lengths, d_offsets, numCells);
    cudaDeviceSynchronize();


    // Copy results back to host and apply to entities
    std::vector<ColliderEntity> host_flattenedData(numCells);
    cudaMemcpy(host_flattenedData.data(), d_flattenedData, numCells * sizeof(ColliderEntity), cudaMemcpyDeviceToHost);
    
    Scene* currScene = Game::getInstance().getCurrentScene();
    for(const ColliderEntity& ent : host_flattenedData) {
        Entity* scene_ent = currScene->getEntity(ent.eid);
        scene_ent->phys->velocity.x = ent.x_vel;
        scene_ent->phys->velocity.y = ent.y_vel;

        scene_ent->transform->position.x = ent.x_pos;
        scene_ent->transform->position.y = ent.y_pos;
    }
}

// Destructor
CudaResolve::~CudaResolve() {
    if (d_flattenedData) cudaFree(d_flattenedData);
    if (d_lengths) cudaFree(d_lengths);
    if (d_offsets) cudaFree(d_offsets);
}

