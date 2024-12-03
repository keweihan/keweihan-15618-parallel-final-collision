#include <cstdio>

// A simple CUDA function
__global__ void gpu_print_kernel() {
    printf("Hello from CUDA kernel!\n");
}

// Wrapper function callable from C++
extern "C" void call_gpu_function() {
    gpu_print_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
