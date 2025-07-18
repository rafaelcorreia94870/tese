// m_harris_reduce.cuh

#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

template <typename T_raw, typename BinaryOp>
__global__ void harrisReduceKernel(const T_raw* g_in, T_raw* g_out, int N, BinaryOp op, T_raw identity) {
    using T = Promote<T_raw>;
    extern __shared__ __align__(sizeof(T)) unsigned char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

    T mySum = identity;

    if (i < N) mySum = g_in[i];
    if (i + blockDim.x < N) mySum = op(mySum, g_in[i + blockDim.x]);

    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = op(mySum, static_cast<T>(sdata[tid + s]));
        }
        __syncthreads();
    }

    if (tid < 32) {
        mySum = warpReduce(mySum, op);
    }

    if (tid == 0) g_out[blockIdx.x] = mySum;
}

template <typename T, typename BinaryOp>
T m_harris_reduce_impl(const thrust::device_vector<T>& d_vec, BinaryOp op, T identity) {
    int N = d_vec.size();
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    thrust::device_vector<T> d_out(blocksPerGrid);

    harrisReduceKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(
        thrust::raw_pointer_cast(d_vec.data()),
        thrust::raw_pointer_cast(d_out.data()),
        N,
        op,
        identity
    );

    // If multiple blocks, do another round recursively
    while (blocksPerGrid > 1) {
        int oldBlocks = blocksPerGrid;
        N = blocksPerGrid;
        blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
        thrust::device_vector<T> d_tmp(blocksPerGrid);

        harrisReduceKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(
            thrust::raw_pointer_cast(d_out.data()),
            thrust::raw_pointer_cast(d_tmp.data()),
            N,
            op,
            identity
        );

        d_out.swap(d_tmp);
    }

    T result;
    cudaMemcpy(&result, thrust::raw_pointer_cast(d_out.data()), sizeof(T), cudaMemcpyDeviceToHost);
    return result;
}

template <typename T, typename BinaryOp>
T m_harris_reduce(const thrust::device_vector<T>& d_vec, BinaryOp op, T identity) {
    #ifdef M_HARRIS_REDUCE
    return m_harris_reduce_impl(d_vec, op, identity);
    #else
    return identity;
    #endif
}