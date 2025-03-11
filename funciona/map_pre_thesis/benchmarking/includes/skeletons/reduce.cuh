#include "../types/vector.cuh"
#include "../types/operation.cuh"


#include <cstdio>



template <typename T, typename BinaryOp, typename... Args>
__global__ void partialReduceKernel(T* d_in, T* d_out, int N, BinaryOp op, Args... args) {
    extern __shared__ T sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;


    T localValue = (i < N) ? d_in[i] : T();
    if (i + blockDim.x < N) {
        localValue = op(localValue, d_in[i + blockDim.x], args...);
    }
    sdata[tid] = localValue;
    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            sdata[tid] = op(sdata[tid], sdata[tid + s], args...);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

template <typename T, typename BinaryOp, typename... Args>
void reduce_kernel(T* d_in, T* d_out, int N, BinaryOp op, Args... args) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    while (blocksPerGrid > 1) {
        partialReduceKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(d_in, d_out, N, op, args...);
        N = blocksPerGrid;
        d_in = d_out;
        blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    }

    // Last reduction step to get d_out[0]
    partialReduceKernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(d_in, d_out, N, op, args...);
}

template <typename T, typename BinaryOp, typename... Args>
T reduce(const std::vector<T>& h_in, BinaryOp op, Args... args) {
    int N = h_in.size();
    if (N == 0) return 0;

    T *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(T));
    cudaMalloc(&d_out, N * sizeof(T));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(T), cudaMemcpyHostToDevice);

    reduce_kernel(d_in, d_out, N, op, args...);

    T result;
    cudaMemcpy(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return result;
}