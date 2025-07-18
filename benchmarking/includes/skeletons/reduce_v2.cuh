#include "../types/vector.cuh"
#include "../types/operation.cuh"

#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>



template <typename T, typename BinaryOp, typename... Args>
__device__ T warpReduce(T val, BinaryOp op, Args... args) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T val_from_lane = __shfl_down_sync(0xffffffff, val, offset);
        val = op(val, val_from_lane, args...);
    }
    return val;
}

template <typename T_raw, typename BinaryOp, typename... Args>
__global__ void reduceKernelCoarsened(
    const T_raw* __restrict__ d_in, T_raw* __restrict__ d_out,
    size_t N_current, T_raw identity, BinaryOp op, Args... args)
{
    using T = Promote<T_raw>;
    extern __shared__ __align__(sizeof(T)) unsigned char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);

    int tid = threadIdx.x;
    size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    size_t totalThreads = static_cast<size_t>(gridDim.x) * blockDim.x;

    T val = static_cast<T>(identity);
    int coarsening = 8; 

    for (size_t i = global_tid; i < N_current; i += totalThreads * coarsening) {
        #pragma unroll
        for (int j = 0; j < coarsening; ++j) {
            size_t current_index = i + static_cast<size_t>(j) * totalThreads; 
            if (current_index < N_current) {
                val = op(val, static_cast<T>(d_in[current_index]), args...);
            }
        }
    }

    val = warpReduce<T>(val, op, args...);

    if ((threadIdx.x & 31) == 0) {
        sdata[threadIdx.x / 32] = val;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        val = (threadIdx.x < (blockDim.x / 32)) ? sdata[threadIdx.x] : static_cast<T>(identity);
        val = warpReduce<T>(val, op, args...);
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = static_cast<T_raw>(val);
    }
}

template <typename T_raw, typename BinaryOp, typename... Args>
T_raw* recursiveReduce(cudaStream_t stream, T_raw* d_src, T_raw* d_dst, size_t N, T_raw identity, BinaryOp op, Args... args) {
    using T = Promote<T_raw>;
    const int threads = 1024;
    const int itemsPerThread = 8; 
    size_t blocks = (N + static_cast<size_t>(threads) * itemsPerThread - 1) / (static_cast<size_t>(threads) * itemsPerThread);
    if (blocks == 0) return d_src; 

    size_t shmem_size = static_cast<size_t>(threads / 32) * sizeof(T);

    reduceKernelCoarsened<<<blocks, threads, shmem_size, stream>>>(d_src, d_dst, N, identity, op, args...);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error in recursiveReduce (reduce_v2.cuh): %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    if (blocks > 1) {
        return recursiveReduce(stream, d_dst, d_src, blocks, identity, op, args...);
    } else {
        return d_dst;
    }
}

template <typename Vector, typename T_raw, typename BinaryOp, typename... Args>
T_raw reduce_v2_impl(const Vector& h_in, T_raw identity, BinaryOp op, Args... args) {
    using T = Promote<T_raw>;
    size_t N = h_in.size();
    if (N == 0) return identity;

    T_raw* d_in_raw;
    T_raw* d_temp_raw;

    if (cudaMalloc(&d_in_raw, N * sizeof(T_raw)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_in_raw failed (reduce_v2.cuh)!\n");
        return identity;
    }
    if (cudaMalloc(&d_temp_raw, N * sizeof(T_raw)) != cudaSuccess) { 
        fprintf(stderr, "cudaMalloc for d_temp_raw failed (reduce_v2.cuh)!\n");
        cudaFree(d_in_raw);
        return identity;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_in_raw, h_in.data(), N * sizeof(T_raw), cudaMemcpyHostToDevice, stream);

    T_raw* d_final_result_ptr = recursiveReduce(stream, d_in_raw, d_temp_raw, N, identity, op, args...);
    if (d_final_result_ptr == nullptr) {
        cudaStreamDestroy(stream);
        cudaFree(d_in_raw);
        cudaFree(d_temp_raw);
        return identity;
    }

    T_raw result;
    cudaMemcpyAsync(&result, d_final_result_ptr, sizeof(T_raw), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_in_raw); 
    cudaFree(d_temp_raw); 
    cudaStreamDestroy(stream);

    return result;
}

template <typename Vector, typename T_raw, typename BinaryOp, typename... Args>
T_raw reduce_v2(const Vector& h_in, T_raw identity, BinaryOp op, Args... args) {
    #ifdef REDUCE_V2
    return reduce_v2_impl(h_in, identity, op, args...);
    #else
    return identity; 
    #endif
}