#ifndef SHARED_DATA_H
#define SHARED_DATA_H

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Plain Old Data (POD) structure
struct PODClass {
    int id;
    float value;

    __host__ __device__ PODClass() : id(0), value(0.0f) {}
};

// Complex Class
class ComplexClass {
public:
    int id;
    float value;

    ComplexClass() : id(0), value(0.0f) {}

    PODClass toPOD() const {
        PODClass pod;
        pod.id = id;
        pod.value = value;
        return pod;
    }

    static ComplexClass fromPOD(const PODClass &pod) {
        ComplexClass complex;
        complex.id = pod.id;
        complex.value = pod.value;
        return complex;
    }
};

// CUDA Manager Class
class CudaManager {
private:
    PODClass *d_data;  // Device data pointer
    size_t data_size;  // Size of the data array

public:
    CudaManager() : d_data(nullptr), data_size(0) {}

    ~CudaManager() {
        if (d_data) cudaFree(d_data);
    }

    // Allocate memory on the device
    void allocate(size_t num_elements) {
        data_size = num_elements;
        cudaMalloc(&d_data, data_size * sizeof(PODClass));
    }

    // Copy data from host to device
    void copyToDevice(const std::vector<PODClass> &host_data) {
        if (host_data.size() != data_size) {
            throw std::runtime_error("Data size mismatch!");
        }
        cudaMemcpy(d_data, host_data.data(), data_size * sizeof(PODClass), cudaMemcpyHostToDevice);
    }

    // Copy data from device to host
    void copyToHost(std::vector<PODClass> &host_data) {
        if (host_data.size() != data_size) {
            host_data.resize(data_size);
        }
        cudaMemcpy(host_data.data(), d_data, data_size * sizeof(PODClass), cudaMemcpyDeviceToHost);
    }

    // Launch a CUDA kernel
    void launchKernel(void (*kernel)(PODClass *, size_t), int threadsPerBlock = 256) {
        int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;
        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, data_size);
        cudaDeviceSynchronize();
    }
};

// CUDA Kernel Example
__global__ void exampleKernel(PODClass *data, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx].value *= 2.0f; // Example operation
    }
}

#endif // SHARED_DATA_H
