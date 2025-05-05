#ifndef MAP_KNERNEL_CUH
#define MAP_KNERNEL_CUH
#include <iostream>

template <typename T, typename Func, typename... Args>
__global__ void mapKernel(T* d_array, int size, Func func, Args... args) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        //printf("Running kernel idx = %d, d_array = %d\n", idx, d_array[idx]);
        //printf("func: %p\n", func);
        //printf("d_array[%zu]: %i\n", idx, d_array[idx]);
        d_array[idx] = func(d_array[idx], args...);
        //printf("d_array[%zu] after: %i\n", idx, d_array[idx]);
    }
}

template <typename T>
__global__ void device_print(T* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        printf("output[%i]: %i\n", idx, d_array[idx]);
    }
}

template <typename T, typename Func, typename... Args>
__global__ void mapKernel2inputs(T* d_array, T* d_array2, int size, Func func, Args... args) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = func(d_array[idx], d_array2[idx], args...);
        //printf("d_array[%zu]: %i\n", idx, d_array[idx]);
    }
}

template <typename T, typename Func, typename... Args>
__global__ void mapKernel2inputsOut(const T* input1, const T* input2, size_t size, Func func, T* output, Args... args) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = func(input1[idx], input2[idx], args...); 
        //printf("output[%zu]: %d\n", idx, output[idx]);
    }
}

#endif // MAP_KNERNEL_CUH