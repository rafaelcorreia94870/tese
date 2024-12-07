#include <iostream>
#include <vector>
#include <list>
#include <array>
#include <map>
#include <deque>
#include <iterator> 
#include <cuda_runtime.h>
#include <type_traits>


template <typename T, typename Func>
__global__ void mapKernel(T* d_array, int size, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        d_array[idx] = func(d_array[idx]);
    }
}
template <typename Iterator, typename Func>
void map(Iterator& container, Func& func) {
    using T = typename Iterator::value_type;
    std::vector<T> temp;  
    
    for (auto it = container.begin(); it != container.end(); ++it) {
        temp.push_back(*it);
    }

    size_t size = temp.size(); 
    T* d_array;
    size_t bytes = size * sizeof(T);
    
    
    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, temp.data(), bytes, cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    __device__ auto device_func = [=] __device__ (T x) { return func(x); };
    mapKernel<<<numBlocks, blockSize>>>(d_array, size, device_func);

    
    cudaMemcpy(temp.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);

    
    std::copy(temp.begin(), temp.end(), container.begin());
}

int increment(int x) {
    return x + 1;
}

int main() {
    std::vector<int> intvec = {0, 1, 2, 3};

    map(intvec, increment);

    std::cout << "\nInt vec\n";
    for (int v : intvec) {
        std::cout << v << " ";
    }

    return 0;
}
