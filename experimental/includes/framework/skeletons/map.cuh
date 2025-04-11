#include "../types/types.cuh"

#define CUDACHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

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
    size_t bytes = size * sizeof(T);
    
    T* d_array;
    CUDACHECK(cudaMalloc(&d_array, bytes));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    CUDACHECK(cudaMemcpyAsync(d_array, container.data(), bytes, cudaMemcpyHostToDevice, stream));

    int blockSize = 1024;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel<<<numBlocks, blockSize, 0, stream>>>(d_array, size, func, args...);
    
    CUDACHECK(cudaMemcpyAsync(container.data(), d_array, bytes, cudaMemcpyDeviceToHost, stream));

    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaStreamDestroy(stream));
    
    CUDACHECK(cudaFree(d_array));
}

template <VectorLike Container, typename Func, typename... Args>
void map_impl(Container& input, Func func, Container& output, Args... args) {
    using T = typename Container::value_type;
    size_t size = input.size();
    size_t bytes = size * sizeof(T);

    T* d_array;
    cudaMalloc(&d_array, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_array, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    int blockSize = 1024;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel<<<numBlocks, blockSize, 0, stream>>>(d_array, size, func, args...);
    
    cudaStreamSynchronize(stream);
    cudaMemcpy(output.data(), d_array, bytes, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);
    cudaFree(d_array);
}

template <VectorLike Container, typename Func, typename... Args>
void map_impl(Container& input1, Container& input2, Func func, Args... args) {
    using T = typename Container::value_type;
    size_t size = input1.size();
    size_t bytes = size * sizeof(T);

    T* d_array;
    T* d_array2;
    cudaMalloc(&d_array, bytes);
    cudaMalloc(&d_array2, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_array, input1.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_array2, input2.data(), bytes, cudaMemcpyHostToDevice, stream);

    int blockSize = 1024;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel2inputs<<<numBlocks, blockSize, 0, stream>>>(d_array, d_array2, size, func, args...);
    
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(input1.data(), d_array, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamDestroy(stream);

    cudaFree(d_array);
    cudaFree(d_array2);
}

template <VectorLike Container, typename Func, typename... Args>
void map_impl(Container& input1, Container& input2, Func func, Container& output, Args... args) {
    using T = typename Container::value_type;
    size_t size = input1.size();
    size_t bytes = size * sizeof(T);

    T* d_array;
    T* d_array2;
    cudaMalloc(&d_array, bytes);
    cudaMalloc(&d_array2, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_array, input1.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_array2, input2.data(), bytes, cudaMemcpyHostToDevice, stream);

    int blockSize = 1024;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel2inputs<<<numBlocks, blockSize, 0, stream>>>(d_array, d_array2, size, func, args...);
    
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(output.data(), d_array, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamDestroy(stream);

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
