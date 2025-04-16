#include "includes/framework/rafa.cuh"
#include <iostream>

// Device function to double the input value
struct DoubleIt {
    __device__ int operator()(int x) const {
        return x * 2;
    }
};


// Device function to add two input values
__device__ int add(int x, int y) {
    return x + y;
}

template <typename T, typename Func, typename... Args>
    __global__ void mapKernel(T* d_array, int size, Func func, Args... args) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            printf("func: %p\n", func);
            printf("d_array[%d]: %i\n", idx, d_array[idx]);
            d_array[idx] = func(d_array[idx], args...);
            printf("d_array[%d] after: %i\n", idx, d_array[idx]);
        }
    }

    template <typename T>
    __global__ void device_print(T* d_array, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            printf("output[%d]: %i\n", idx, d_array[idx]);
        }
    }

int main() {

    std::cout << "Test 1: Basic vector initialization and data synchronization\n";
    rafa::vector<int> input(5);
    for (int i = 0; i < 5; ++i) {
        input[i] = i;
    }
    input.sync_host_to_device();
    for (int i = 0; i < 5; ++i) {
        input[i] = 0;
    }
    input.sync_device_to_host();
    input.print();

    rafa::vector<int> output(5,1);
    
    size_t size = input.size();
    size_t bytes = size * sizeof(int);

    cudaHostAlloc(reinterpret_cast<void**>(&output.device_data), bytes, cudaHostAllocDefault);
    
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int blockSize = 1024;
    int numBlocks = (size + blockSize - 1) / blockSize;
    device_print<<<numBlocks, blockSize, 0>>>(output.device_data, size);
    cudaMemcpyAsync(output.device_data, input.data(), bytes, cudaMemcpyHostToDevice, stream);
    device_print<<<numBlocks, blockSize, 0>>>(output.device_data, size);
    mapKernel<<<numBlocks, blockSize, 0, stream>>>(output.device_data, size, DoubleIt{});
    device_print<<<numBlocks, blockSize, 0>>>(output.device_data, size);
    
    output.sync_device_to_host(stream);
    std::cout << "device data: " << output.device_data << std::endl;
    std::cout << "output data: " << output.device_data << std::endl;
    device_print<<<numBlocks, blockSize, 0>>>(output.device_data, size);
    cudaStreamDestroy(stream);

    output.print();
}