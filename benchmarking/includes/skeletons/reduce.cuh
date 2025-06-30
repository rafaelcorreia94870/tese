#include "../types/vector.cuh"
#include "../types/operation.cuh"

#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>
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



////// FAST REDUCE //////

//Para funcionar com tipos menores que int, como char e uint8_t, _shfl_down_sync nao funciona corretamente com esses tipos.
template <typename T>
using Promote = typename std::conditional<sizeof(T) < 4, int, T>::type;

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
    int N, T_raw identity, BinaryOp op, Args... args)
{
    using T = Promote<T_raw>;
    extern __shared__ __align__(sizeof(T)) unsigned char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + tid;

    T val = static_cast<T>(identity);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int offset = idx + i * blockDim.x;
        if (offset < N) {
            val = op(val, static_cast<T>(d_in[offset]), args...);
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

template <typename T, typename BinaryOp, typename... Args>
void recursiveReduce(cudaStream_t stream, const T* d_in, T* d_temp, int N, T identity, BinaryOp op, Args... args) {
    const int threads = 1024;
    const int itemsPerThread = 8;
    int blocks = (N + threads * itemsPerThread - 1) / (threads * itemsPerThread);

    reduceKernelCoarsened<<<blocks, threads, threads / 32 * sizeof(T), stream>>>(d_in, d_temp, N, identity, op, args...);

    if (blocks > 1) {
        recursiveReduce(stream, d_temp, d_temp, blocks, identity, op, args...);
    }
}

template <typename Vector, typename T_raw, typename BinaryOp, typename... Args>
T_raw reduce_fast(const Vector& h_in, T_raw identity, BinaryOp op, Args... args) {
    using T = Promote<T_raw>;
    int N = h_in.size();
    if (N == 0) return identity;

    T* d_in;
    T* d_temp;
    cudaMalloc(&d_in, N * sizeof(T));
    cudaMalloc(&d_temp, N * sizeof(T));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<T> h_in_promoted(h_in.begin(), h_in.end());
    cudaMemcpyAsync(d_in, h_in_promoted.data(), N * sizeof(T), cudaMemcpyHostToDevice, stream);

    recursiveReduce(stream, d_in, d_temp, N, static_cast<T>(identity), op, args...);

    T result;
    cudaMemcpyAsync(&result, d_temp, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_in);
    cudaFree(d_temp);
    cudaStreamDestroy(stream);

    return static_cast<T_raw>(result);
}



