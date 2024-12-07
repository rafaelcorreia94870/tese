#include <iostream>
#include <vector>
#include <cuda_runtime.h>


#define DEVICE __device__


template <typename T, typename Func>
__global__ void mapKernel(T* d_array, int size, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = func(d_array[idx]);
    }
}

template <typename T, typename Func>
void map(std::vector<T>& container, Func func) {
    size_t size = container.size();
    T* d_array;
    size_t bytes = size * sizeof(T);


    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, container.data(), bytes, cudaMemcpyHostToDevice);


    auto device_func = [=] DEVICE(T x) { return func(x); };


    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    mapKernel<<<numBlocks, blockSize>>>(d_array, size, device_func);

    cudaMemcpy(container.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}


float square(float x) {
    return x * x;
}

int main() {

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};


    map(vec, square);


    std::cout << "Squared values:\n";
    for (float v : vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    return 0;
}
