#include "cuda_map.h"

template <typename T, typename Func>
__global__ void mapKernel(T* d_array, int size, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = func(d_array[idx]); // Apply the functor directly
    }
}

// Non-template helper for kernel launch
template <typename T, typename Func>
void launchMapKernel(T* d_array, size_t size, Func func) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

// Explicit instantiation for supported types
template void launchMapKernel<float, Square>(float* d_array, size_t size, Square func);
template void launchMapKernel<int, Increment>(int* d_array, size_t size, Increment func);
