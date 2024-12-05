#ifndef CUDA_MAP_H
#define CUDA_MAP_H

#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>

// Kernel definition
template <typename T, typename Func>
__global__ void mapKernel(T* d_array, int size, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = func(d_array[idx]); // Apply the functor
    }
}

// Map function implementation
template <typename T, typename Func>
void map(std::vector<T>& container, Func func) {
    size_t size = container.size();
    T* d_array = nullptr;
    size_t bytes = size * sizeof(T);

    // Allocate memory on the device
    cudaError_t err = cudaMalloc(&d_array, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Copy data to the device
    err = cudaMemcpy(d_array, container.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_array);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Define the block and grid size
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch the kernel
    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_array);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Copy the result back to the host
    err = cudaMemcpy(container.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_array);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_array);
}

#endif // CUDA_MAP_H
