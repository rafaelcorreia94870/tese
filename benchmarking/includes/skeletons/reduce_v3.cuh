#include "../types/vector.cuh"
#include "../types/operation.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>

template <typename T, typename BinaryOp>
__device__ T warpReduce(T val, BinaryOp op) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = op(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T_raw, typename BinaryOp>
__global__ void reduceKernelPersistent(const T_raw* d_in, T_raw* d_out, int N, T_raw identity, BinaryOp op) {
    using T = Promote<T_raw>;
    extern __shared__ __align__(sizeof(T)) unsigned char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);
    T sum = static_cast<T>(identity);
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int totalThreads = gridDim.x * blockDim.x;
    for (int i = global_tid; i < N; i += totalThreads) {
        sum = op(sum, static_cast<T>(d_in[i]));
    }
    sum = warpReduce(sum, op);
    if (threadIdx.x % warpSize == 0) {
        sdata[threadIdx.x / warpSize] = sum;
    }
    __syncthreads();
    if (threadIdx.x < warpSize) {
        T final = (threadIdx.x < (blockDim.x / warpSize)) ? sdata[threadIdx.x] : identity;
        final = warpReduce(final, op);
        if (threadIdx.x == 0) {
            d_out[blockIdx.x] = static_cast<T_raw>(final);
        }
    }
}

template <typename Vector, typename T, typename BinaryOp>
T reduce_v3_impl(const Vector& h_in, T identity, BinaryOp op) {
    int N = h_in.size();
    if (N == 0) return identity;
    int threads = 256;
    int maxBlocks = 1024;
    T *d_in = nullptr, *d_temp = nullptr;
    if(cudaMalloc(&d_in, N * sizeof(T)) != cudaSuccess) {
        std::cerr << "Error allocating device memory for input array." << std::endl;
        return identity;
    }
    if(cudaMalloc(&d_temp, N * sizeof(T)) != cudaSuccess) {
        std::cerr << "Error allocating device memory for temporary array." << std::endl;
        cudaFree(d_in);
        return identity;
    }
    cudaMemcpy(d_in, h_in.data(), N * sizeof(T), cudaMemcpyHostToDevice);
    size_t shmem = threads / 32 * sizeof(Promote<T>);
    int blocks = (N + threads - 1) / threads;
    if (blocks > maxBlocks) blocks = maxBlocks;
    reduceKernelPersistent<<<blocks, threads, shmem>>>(d_in, d_temp, N, identity, op);
    while (blocks > 1) {
        int nextBlocks = (blocks + threads - 1) / threads;
        if (nextBlocks > maxBlocks) nextBlocks = maxBlocks;
        reduceKernelPersistent<<<nextBlocks, threads, shmem>>>(d_temp, d_temp, blocks, identity, op);
        blocks = nextBlocks;
    }
    T result;
    cudaMemcpy(&result, d_temp, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_temp);
    return result;
}

template <typename Vector, typename T, typename BinaryOp>
T reduce_v3(const Vector& h_in, T identity, BinaryOp op) {
    #ifdef REDUCE_V3
    return reduce_v3_impl(h_in, identity, op);
    #else
    return identity;
    #endif
}
