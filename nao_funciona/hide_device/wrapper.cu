#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <type_traits>

template <typename T, T (*Func)(T)>
struct FunctionWrapper {
    __device__ T operator()(T x) const {
        return Func(x);
    }
};

template <typename T, typename Func>
__global__ void mapKernel(T* d_array, int size, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = func(d_array[idx]);
    }
}

template <typename Iterator, typename T, T (*func)(T)>
void map(Iterator& container) {
    using ValueType = typename Iterator::value_type;

    std::vector<T> temp(container.begin(), container.end());

    size_t size = temp.size();
    T* d_array;
    size_t bytes = size * sizeof(T);

    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, temp.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    FunctionWrapper<T, func> device_func;
    mapKernel<<<numBlocks, blockSize>>>(d_array, size, device_func);
    cudaDeviceSynchronize();

    cudaMemcpy(temp.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);

    std::copy(temp.begin(), temp.end(), container.begin());
}

int increment(int x) {
    return x + 1;
}

int main() {
    std::vector<int> intvec = {0, 1, 2, 3};

    map<std::vector<int>, int, increment>(intvec);

    std::cout << "\nInt vec\n";
    for (int v : intvec) {
        std::cout << v << " ";
    }

    return 0;
}
