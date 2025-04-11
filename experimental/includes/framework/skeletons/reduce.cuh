#include "../types/types.cuh"

#include <cstdio>
#include <cuda_runtime.h>
//ver https://nvidia.github.io/cccl/cub/api/enum_namespacecub_1add0251c713859b8974806079e498d10a.html
//https://github.com/dmlc/cub/blob/master/cub/device/device_reduce.cuh

template <typename T, typename BinaryOp, typename... Args>
__global__ void partialReduceKernel(T* d_in, T* d_out, int N, T identity, BinaryOp op, Args... args) {
    extern __shared__ T sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    T localValue = (i < N) ? d_in[i] : identity;
    if (i + blockDim.x < N) {
        localValue = op(localValue, d_in[i + blockDim.x], args...);
    }
    sdata[tid] = localValue;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = op(sdata[tid], sdata[tid + s], args...);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

template <typename T, typename BinaryOp, typename... Args>
void reduce_kernel(cudaStream_t stream, T* d_in, T* d_out, int N, T identity, BinaryOp op, Args... args) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    while (blocksPerGrid > 1) {
        partialReduceKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(T), stream>>>(d_in, d_out, N, identity, op, args...);
        d_in = d_out; 
        N = blocksPerGrid;
        blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    }

    partialReduceKernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(T), stream>>>(d_in, d_out, N, identity, op, args...);
}

template <VectorLike Collection, typename T, typename BinaryOp, typename... Args>
T reduce_impl(const Collection &h_in, T identity, BinaryOp op, Args... args) {
    int N = h_in.size();
    if (N == 0) return identity;

    T *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(T));
    cudaMalloc(&d_out, N * sizeof(T));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_in, h_in.data(), N * sizeof(T), cudaMemcpyHostToDevice, stream);

    reduce_kernel(stream, d_in, d_out, N, identity, op, args...);

    T result;
    cudaMemcpyAsync(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);

    return result;
}

template <VectorLike Collection, typename T, typename BinaryOp, typename... Args>
auto reduce(const Collection &h_in, T identity, BinaryOp op, Args... args) {
    return reduce_impl(h_in, identity, op, args...);
}

template <VectorLike Collection, typename T, typename BinaryOp, typename... Args>
auto reduce(const Collection &h_in, Collection &h_out, T identity, BinaryOp op, Args... args) {
    h_out = reduce_impl(h_in, identity, op, args...);
    return h_out;
}
