# Parallelization Schemes for Collision Detection
**Team**: Kewei Han (keweihan), Jiya Zhang (jiyaz)

**URL**: https://keweihan.github.io/keweihan-15618-parallel-final-collision/

# Summary
We are going to explore parallelizing spatial grid structures such as a fixed spatial grid and quadtree to optimize collision detection computation using CUDA.

# Background
Our project focuses on enhancing a compute-intensive application for 2D collision detection among 15,000+ particles. Particles are assigned to a fixed-grid spatial structure, where each grid cell contains references to particles in that area. For each frame, the program currently checks sequentially for collisions within each cell abstracted as the following logic. We plan to parallelize the particle-to-cell assignment (updateGrid) with either per-grid or per-particle or both for parallelization strategy. 

To simulate imbalanced workload, integrating a quadtree structure into our parallelized approach may improve performance, particularly in densely populated grid areas. The quadtree structure allows for hierarchical partitioning, reducing the number of collision checks necessary by only examining particles within relevant quadrants.

```C++
// update grid with collider/entity references
colliderGrid.updateGrid(); 

// Iterate through every cell
for (int i = 0; i < colliderGrid.size(); ++i)
{
 // Only brute force check colliders contained within each cell
 for (auto& colliderA : colliderGrid.getCellContents(i))
 {
  for (auto& colliderB : colliderGrid.getCellContents(i))
  {
   if (colliderA == colliderB) continue;
   resolveCollision(colliderA, colliderB);
  }
 }
}

```

# Challenge
Collision detection is challenging to parallelize due to potential data access conflicts and uneven workload distribution across threads. We anticipate the following key issues:

- Workload Imbalance: Unequal particle distribution within grid cells may result in some threads having more work than others, causing inefficiencies.
- Memory Access Patterns: Parallel updates to the grid structure introduce risks of data races and conflicts, especially when particles span multiple grid cells or quadrants.
- Algorithmic Complexity with Quadtrees: While quadtree integration could reduce collision detection costs with unbalanced particle distribution, it introduces complexities in parallel updates and may not suit evenly distributed particles.

By exploring multiple parallelization strategies, including CUDA thread management and potential quadtree partitioning, we hope to identify efficient methods for different workload scenarios.

# Resources
- Hardware: GHC Machines (GPUs and multicore CPUs); personal Windows operating system.
- Starter Code: https://github.com/keweihan/SimpleECS/tree/main (A basic sequential 2D particle simulator implemented in C++).
- Quadtree-related Resources: https://edwardsjohnmartin.github.io/publications/papers/morrical2017parallel.pdf.

# Goals and Deliverables
## Plan to Achieve

#### **Spatial grid construction parallelization**
- Implement parallelization of the spatial grid structure where particles are assigned to grid cells based on their positions. This grid-based approach will initially assume a balanced workload, using CUDA to parallelize on a per-grid or per-particle basis.
- Ensure conflict-free access to the grid when multiple particles update grid cells simultaneously, leveraging CUDA’s thread management to efficiently handle memory access. This involves experimenting with atomic operations or other thread-safe techniques to avoid data races.

#### **Quadtree construction parallelization for uneven work distribution**
- Implement a sequential quadtree structure, which will provide a hierarchical spatial partitioning that adapts to particle density.
- To simulate an imbalanced workload, vary particle density in different regions of the 2D space and assess the quadtree’s ability to dynamically allocate workload based on particle clustering.
- Parallelize the quadtree construction phase with CUDA after testing the sequential implementation to ensure that each CUDA thread can efficiently handle subsets of the quadtree with minimal conflicts.

## Hope to Achieve
#### **Parallelization of collision resolution step too**
- Explore CUDA-based parallelization of the collision resolution phase within each grid cell or quadtree node, allowing multiple collisions to be processed simultaneously.
- Address challenges in cases where particles span multiple cells or quadrants, potentially using boundary conditions or inter-thread communication to resolve such conflicts efficiently.

## Demo
We aim to show performance metrics, including speedup graphs and memory usage visualizations for CUDA implementation under different distribution simulation. We will also try to display particle movements in real-time.

# Platform Choice
For implementation, we will use CUDA, as it's beneficial to handle the high degree of parallelism in collision detection, particularly with proper memory access handling and kernel management. 
We will mainly use GHC to conduct performance tests.

# Schedule
| Week         | Task       |
|:--------------|:-----------|
| Nov 17 - 23  | Finalize research on spatial grid and quadtree structures. Begin with CUDA-based parallelization of updateGrid based a sequential baseline version, using a per-grid approach for balanced workload scenarios.|
| Nov 24 - 30  | Implement and test per-particle parallelization of updateGrid with CUDA. Compare per-grid vs. per-particle methods and optimize based on initial performance data.|
| Dec 1  -  7  | Implement a sequential quadtree to handle imbalanced particle distributions. If time permits, begin parallelizing the quadtree construction with CUDA, focusing on efficient workload partitioning across threads.|
| Dec 8  - 15  | Finalize performance testing for both grid-based and quadtree implementations, focusing on speedup and memory usage metrics. Prepare demo visualizations, speedup graphs, and project documentation.|
