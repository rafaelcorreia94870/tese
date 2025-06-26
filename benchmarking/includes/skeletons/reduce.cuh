#include "../types/vector.cuh"
#include "../types/operation.cuh"

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

#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>

template <typename T, typename BinaryOp, typename... Args>
__device__ T warpReduce(T val, BinaryOp op, Args... args) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = op(val, __shfl_down_sync(0xffffffff, val, offset), args...);
    }
    return val;
}

template <typename T, typename BinaryOp, typename... Args>
__global__ void reduceKernelCoarsened(
    const T* __restrict__ d_in, T* __restrict__ d_out,
    int N, T identity, BinaryOp op, Args... args)
{
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + tid;

    T val = identity;

    // Coarsened loading: each thread sums 8 elements spaced by blockDim.x
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int offset = idx + i * blockDim.x;
        if (offset < N) {
            val = op(val, d_in[offset], args...);
        }
    }

    // Warp shuffle reduction
    val = warpReduce(val, op, args...);

    // Inter-warp reduction using shared memory
    if ((threadIdx.x & 31) == 0) {
        sdata[threadIdx.x / 32] = val;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        val = (threadIdx.x < (blockDim.x / 32)) ? sdata[threadIdx.x] : identity;
        val = warpReduce(val, op, args...);
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = val;
    }
}

template <typename T, typename BinaryOp, typename... Args>
void recursiveReduce(cudaStream_t stream, const T* d_in, T* d_temp, int N, T identity, BinaryOp op, Args... args) {
    const int threads = 256;
    const int itemsPerThread = 8;
    int blocks = (N + threads * itemsPerThread - 1) / (threads * itemsPerThread);

    reduceKernelCoarsened<<<blocks, threads, threads / 32 * sizeof(T), stream>>>(d_in, d_temp, N, identity, op, args...);

    if (blocks > 1) {
        recursiveReduce(stream, d_temp, d_temp, blocks, identity, op, args...);
    }
}

template <typename Vector, typename T, typename BinaryOp, typename... Args>
T reduce_fast(const Vector& h_in, T identity, BinaryOp op, Args... args) {
    int N = h_in.size();
    if (N == 0) return identity;

    T* d_in;
    T* d_temp;
    cudaMalloc(&d_in, N * sizeof(T));
    cudaMalloc(&d_temp, N * sizeof(T));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_in, h_in.data(), N * sizeof(T), cudaMemcpyHostToDevice, stream);

    recursiveReduce(stream, d_in, d_temp, N, identity, op, args...);

    T result;
    cudaMemcpyAsync(&result, d_temp, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_in);
    cudaFree(d_temp);
    cudaStreamDestroy(stream);

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
