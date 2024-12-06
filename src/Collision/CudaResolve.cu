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

    // TODO: hacky solution. Instead check for if PhysicsBody exists in outer loop 
    try
    {
        Handle<PhysicsBody> physHandle = col->entity->getComponent<PhysicsBody>();
        y_vel = physHandle->velocity.y;
        x_vel = physHandle->velocity.x;
    }
    catch(const std::exception& e)
    {
        // std::cerr << e.what() << '\n';
    }
}

// Flatten and copy to device
void CudaResolve::flattenCopyToDevice() {
    std::vector<ColliderEntity> flattenedData;
    std::vector<int> lengths; // sizes of cells in flattened
    std::vector<int> offsets; // starts of cells in flattened
    
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
        // TODO: hacky solution. Instead check for if PhysicsBody exists in outer loop 
        try
        {
            Handle<PhysicsBody> physHandle = currScene->getComponent<PhysicsBody>(ent.eid);
            physHandle->velocity.x = ent.x_vel;
            physHandle->velocity.y = ent.y_vel;

            physHandle->entity->transform->position.y = ent.y_pos;
            physHandle->entity->transform->position.x = ent.x_pos;
        }
        catch(const std::exception& e)
        {

        }
    }
}

// Destructor
CudaResolve::~CudaResolve() {
    if (d_flattenedData) cudaFree(d_flattenedData);
    if (d_lengths) cudaFree(d_lengths);
    if (d_offsets) cudaFree(d_offsets);
}

