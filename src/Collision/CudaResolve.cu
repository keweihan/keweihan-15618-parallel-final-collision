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
    y_max = bounds.yMax;
    
    w = col->width;
    h = col->height;

    y_pos = col->entity->transform->position.y;
    x_pos = col->entity->transform->position.x;

    y_vel = col->entity->phys->velocity.y;
    x_vel = col->entity->phys->velocity.x;

    mass = col->entity->phys->mass;
    is_static = col->entity->phys->is_static;

    is_dirty = false;
}

std::ostream& operator<<(std::ostream& os, const ColliderEntity& entity) {
        os << "ColliderEntity { "
           << "eid: " << entity.eid
           << ", mass: " << entity.mass
           << ", x_vel: " << entity.x_vel
           << ", y_vel: " << entity.y_vel
           << ", x_pos: " << entity.x_pos
           << ", y_pos: " << entity.y_pos
           << ", x_min: " << entity.x_min
           << ", x_max: " << entity.x_max
           << ", y_min: " << entity.y_min
           << ", y_max: " << entity.y_max
           << " }";
        return os;
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
    int _numCells = 0;
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
            offset += cell.size();
        }
    }

    numCells = lengths.size();
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



class CudaVector {
public:
    __device__ CudaVector() : x(0), y(0) {}
    __device__ CudaVector(double x, double y) : x(x), y(y) {}

    __device__ double dotProduct(const CudaVector& other) const {
        return other.x * this->x + other.y * this->y;
    }

    __device__ CudaVector orthogonalVec() const {
        return CudaVector(this->y, -this->x);
    }
    
    __device__ double getMagnitude() const {
        return sqrt(this->x * this->x + this->y * this->y);
    }

    __device__ double distance(const CudaVector& other) const {
        return std::sqrt( std::pow(other.x - this->x, 2) + std::pow(other.y - this->y, 2));
    }

    __device__ void normalize() {
        double magnitude = getMagnitude();
        this->x = this->x / magnitude;
        this->y = this->y / magnitude;
    }

    __device__ CudaVector operator+(const CudaVector& other) const {
        return CudaVector(this->x + other.x, this->y + other.y);
    }

    __device__ CudaVector operator-(const CudaVector& other) const {
        return CudaVector(this->x - other.x, this->y - other.y);
    }

    __device__ CudaVector operator*(const double& other) const {
        return CudaVector(this->x * other, this->y * other);
    }

    __device__ CudaVector operator*(const int& other) const {
        return CudaVector(this->x * other, this->y * other);
    }

    double x, y;
};

/**
* Data container for collision information
*/
class CudaCollision {
public:
    __device__ CudaCollision() = default;

    bool is_colliding = false;
    double penetration = 0;
    
    double normal_x;
    double normal_y;
};

// Detect collision. Adpated version of SimpleECS::ColliderSystem::getCollisionBoxBox
__device__ CudaCollision getCollisionBoxBox(const ColliderEntity a, const ColliderEntity b)
{
    CudaCollision collide;
    collide.is_colliding = true;

    //If any of the sides from A are outside of B, no collision occuring.
    double epsilon = 0.01;  // Small margin value
    if (a.y_min >= b.y_max - epsilon || a.y_max <= b.y_min + epsilon
        || a.x_max <= b.x_min + epsilon || a.x_min >= b.x_max - epsilon)
    {
        collide.is_colliding = false;
        return collide;
    }

    // Boxes are colliding. Find axis of least penetration
    double aExtentX = a.w / 2.0;
    double bExtentX = b.w / 2.0;
    double aExtentY = a.h / 2.0;
    double bExtentY = b.h / 2.0;

    double xDistance = std::abs(a.x_pos - b.x_pos);
    double xOverlap = (aExtentX + bExtentX) - xDistance;

    double yDistance = std::abs(a.y_pos - b.y_pos);
    double yOverlap = (aExtentY + bExtentY) - yDistance;

    // Least penetration is on y-axis
    if (yOverlap < xOverlap)
    {
        collide.penetration = yOverlap;
        if (a.y_pos < b.y_pos)
        {
            collide.normal_x = 0;
            collide.normal_y = -1;
        }
        else
        {
            collide.normal_x = 0;
            collide.normal_y = 1;
        }
    }
    // Least penetration is on x-axis
    else
    {
        collide.penetration = xOverlap;
        if (a.x_pos < b.x_pos)
        {
            collide.normal_x = -1;
            collide.normal_y = 0;
        }
        else
        {
            collide.normal_x = 1;
            collide.normal_y = 0;
        }
    }

	return collide;
}

__device__ void resolveCollide(CudaCollision collide, ColliderEntity* a, ColliderEntity* b)
{
    if(a->eid == b->eid) return;
    if(a->is_static) return;
	if(!collide.is_colliding) return;
    
    a->is_dirty = true;

	double massCoef;
	massCoef = 2 * b->mass / (b->mass + a->mass);
    if(b->is_static) {
        massCoef = 2;
    }

	// Shift position out of overlap (weighted shift amount based on relative mass)
    double shift_x = collide.normal_x * collide.penetration * massCoef/2;
    double shift_y = collide.normal_y * collide.penetration * massCoef/2;
	a->x_pos += shift_x;
	a->y_pos += shift_y;

    // if(a->eid == 4) {
    //     printf("4 collided with %d with collision normal of (%f, %f) and pen %f\n", b->eid, collide.normal_x, collide.normal_y, collide.penetration);
    //     printf("4 is shifting: (%f, %f)\n", shift_x, shift_y);
    // }

    //printf("%d is shifting x %f, y %f, from a collision with %d. End x y is (%f, %f)\n", a->eid, shift_x, shift_y, b->eid, a->x_pos, a->y_pos);
	
    // Calculate new velocity (mass velocity 2D calculation)
	// Adapted from https://en.wikipedia.org/wiki/Elastic_collision#Two-Dimensional_Collision_With_Two_Moving_Objects
	// using collision normals instead of position vectors 
    CudaVector a_pos = {a->x_pos, a->y_pos};
    CudaVector b_pos = {b->x_pos, b->y_pos};

    CudaVector a_vel = {a->x_vel, a->y_vel};
    CudaVector b_vel = {b->x_vel, b->y_vel};

    CudaVector norm_vec = { static_cast<double>(collide.normal_x), static_cast<double>(collide.normal_y)};

	CudaVector velocityChange = norm_vec * ((a_vel - b_vel).dotProduct(norm_vec)) * massCoef;

	CudaVector futureVelocity = a_vel - velocityChange;
    a->fx_vel = futureVelocity.x;
    a->fy_vel = futureVelocity.y;
}


// Kernel
__global__ void kernel(ColliderEntity* d_flattenedData, int* d_lengths, int* d_offsets, int numCells) {
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx < numCells) {
        int length = d_lengths[cellIdx];
        int offset = d_offsets[cellIdx];

        for (int i = 0; i < length; ++i) {
            ColliderEntity* entity_a = &d_flattenedData[offset + i];
            
            for (int j = i + 1; j < length; ++j) {
                ColliderEntity* entity_b = &d_flattenedData[offset + j];
                
                // Resolve physics here
                CudaCollision colA = getCollisionBoxBox(*entity_a, *entity_b);
                CudaCollision colB = getCollisionBoxBox(*entity_b, *entity_a);
                // if(entity_a->eid == 3) {
                //     printf("(outer) 3 collided with %d with collision normal of (%d, %d) and pen %f\n", entity_b->eid, col.normal_x, col.normal_y, col.penetration);
                // }

                resolveCollide(colA, entity_a, entity_b);
                resolveCollide(colB, entity_b, entity_a);
            }
        }
    }
}

// Kernel launch
void CudaResolve::launchKernel(int numThreads) {
    // Launch kernel process parallel
    dim3 blockDim(numThreads);
    dim3 gridDim((numCells + numThreads - 1) / numThreads);
    kernel<<<gridDim, blockDim>>>(d_flattenedData, d_lengths, d_offsets, numCells);
    cudaDeviceSynchronize();

    // Copy results back to host and apply to entities
    std::vector<ColliderEntity> host_flattenedData(numCells);
    cudaMemcpy(host_flattenedData.data(), d_flattenedData, numCells * sizeof(ColliderEntity), cudaMemcpyDeviceToHost);
    
    Scene* currScene = Game::getInstance().getCurrentScene();

    int fiveCount = 0;
    for(const ColliderEntity& ent : host_flattenedData) {
        Entity* scene_ent = currScene->getEntity(ent.eid);
        if(ent.is_dirty)
        {
            scene_ent->phys->futureVelocity.x = ent.fx_vel;
            scene_ent->phys->futureVelocity.y = ent.fy_vel;

            scene_ent->transform->position.x = ent.x_pos;
            scene_ent->transform->position.y = ent.y_pos;
        }
    }
}

// Destructor
CudaResolve::~CudaResolve() {
    if (d_flattenedData) cudaFree(d_flattenedData);
    if (d_lengths) cudaFree(d_lengths);
    if (d_offsets) cudaFree(d_offsets);
}

