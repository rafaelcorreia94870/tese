#include "../types/vector.cuh"


template <typename T, typename Func, typename... Args>
__global__ void mapKernel(T* d_array, int size, Func func, Args... args) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = func(d_array[idx], args...);
    }
}

template <typename T, typename Func, typename... Args>
__global__ void mapKernel2inputs(T* d_array, T* d_array2, int size, Func func, Args... args) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = func(d_array[idx], d_array2[idx], args...);
    }
}



template <VectorLike Container, typename Func, typename... Args>
void map_impl(Container& container, Func func, Args... args) {
    using T = typename Container::value_type;
    size_t size = container.size();
    T* d_array;
    size_t bytes = size * sizeof(T);

    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, container.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func, args...);
    cudaDeviceSynchronize();
    cudaMemcpy(container.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}


template <VectorLike Container, typename Func, typename... Args>
void map_impl(Container& input, Func func, Container& output, Args... args) {
    using T = typename Container::value_type;
    size_t size = input.size();
    T* d_array;
    size_t bytes = size * sizeof(T);

    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, input.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func, args...);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

template <VectorLike Container, typename Func, typename... Args>
void map_impl(Container& input1, Container& input2, Func func, Args... args) {
    using T = typename Container::value_type;
    size_t size = input1.size();
    T* d_array;
    T* d_array2;
    size_t bytes = size * sizeof(T);

    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, input1.data(), bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_array2, bytes);
    cudaMemcpy(d_array2, input2.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel2inputs<<<numBlocks, blockSize>>>(d_array, d_array2, size, func, args...);

    cudaDeviceSynchronize();
    cudaMemcpy(input1.data(), d_array, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_array2);

}

template <VectorLike Container, typename Func, typename... Args>
void map_impl(Container& input1, Container& input2, Func func, Container& output, Args... args) {
    using T = typename Container::value_type;
    size_t size = input1.size();
    T* d_array;
    T* d_array2;
    size_t bytes = size * sizeof(T);

    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, input1.data(), bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_array2, bytes);
    cudaMemcpy(d_array2, input2.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel2inputs<<<numBlocks, blockSize>>>(d_array,d_array2, size, func, args...);
    
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_array, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_array2);

}

template <VectorLike Container, typename Func, typename... Args>
void map(Container& container, Func func, Args... args) {
    map_impl(container, func, args...);
}

template <VectorLike Container, typename Func, typename... Args>
void map(Container& container, Func func, Container& output, Args... args) {
    map_impl(container, func, output, args...); 
}

template <VectorLike Container, typename Func, typename... Args>
void map(Container& container1, Container& container2, Func func, Args... args) {
    map_impl(container1,container2, func, args...);
}

template <VectorLike Container, typename Func, typename... Args>
void map(Container& container1,Container& container2, Func func, Container& output, Args... args) {
    map_impl(container1,container2, func, output, args...); 
}
