#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <cstdio> 

constexpr int WARP_SIZE = 32;



template <typename T_raw, typename BinaryOp>
__global__ void reduceKernel_v4(
    const T_raw* __restrict__ d_in, T_raw* __restrict__ d_out,
    size_t N_current, T_raw identity, BinaryOp op, int coarsening)
{
    using T = Promote<T_raw>;
    extern __shared__ __align__(sizeof(T)) unsigned char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);

    int tid = threadIdx.x;
    T sum = static_cast<T>(identity);

    size_t global_idx_base = static_cast<size_t>(blockIdx.x) * blockDim.x * coarsening + tid;

    size_t grid_total_stride = static_cast<size_t>(gridDim.x) * blockDim.x * coarsening;

    for (size_t current_chunk_start_for_thread = global_idx_base;
         current_chunk_start_for_thread < N_current;
         current_chunk_start_for_thread += grid_total_stride)
    {
        #pragma unroll
        for (int k = 0; k < coarsening; ++k) {
            size_t current_element_index = current_chunk_start_for_thread + static_cast<size_t>(k) * blockDim.x;

            if (current_element_index < N_current) {
                sum = op(sum, static_cast<T>(d_in[current_element_index]));
            } else {
                break;
            }
        }
    }

    sum = warpReduce(sum, op);

    if ((tid & (WARP_SIZE - 1)) == 0) {
        sdata[tid / WARP_SIZE] = sum;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        T final = (tid < (blockDim.x / WARP_SIZE)) ? sdata[tid] : identity;
        final = warpReduce(final, op); 
        if (tid == 0) {
            d_out[blockIdx.x] = static_cast<T_raw>(final);
        }
    }
}

template <typename Vector, typename T_raw, typename BinaryOp>
T_raw reduce_v4(const Vector& h_in, T_raw identity, BinaryOp op, int coarsening = 8) {
    using T = Promote<T_raw>;
    size_t N = h_in.size();
    if (N == 0) return identity;

    T_raw* d_in_raw = nullptr;
    T_raw* d_temp_raw = nullptr;


    cudaError_t err;
    if ((err = cudaMalloc(&d_in_raw, N * sizeof(T_raw))) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_in_raw failed (reduce_v4.cuh)! %s\n", cudaGetErrorString(err));
        return identity;
    }

    if ((err = cudaMalloc(&d_temp_raw, N * sizeof(T_raw))) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_temp_raw failed (reduce_v4.cuh)! %s\n", cudaGetErrorString(err));
        cudaFree(d_in_raw); 
        return identity;
    }

    if ((err = cudaMemcpy(d_in_raw, h_in.data(), N * sizeof(T_raw), cudaMemcpyHostToDevice)) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy h_in to d_in_raw failed (reduce_v4.cuh): %s\n", cudaGetErrorString(err));
        cudaFree(d_in_raw);
        cudaFree(d_temp_raw);
        return identity;
    }

    int threads = 256; 


    size_t blocks_for_first_pass = (N + static_cast<size_t>(threads) * coarsening - 1) / (static_cast<size_t>(threads) * coarsening);

    size_t shmem = static_cast<size_t>(threads / WARP_SIZE) * sizeof(T);
    if (shmem == 0) shmem = WARP_SIZE * sizeof(T);

    T_raw* d_src = d_in_raw;
    T_raw* d_dst = d_temp_raw;

    size_t current_N_elements_to_reduce = N;
    size_t current_blocks = blocks_for_first_pass;


    while (current_N_elements_to_reduce > 1) {
        size_t current_coarsening_param;
        if (current_N_elements_to_reduce == N) { 
            current_coarsening_param = coarsening;
        } else { 
            current_coarsening_param = 1;
        }


        current_blocks = (current_N_elements_to_reduce + static_cast<size_t>(threads) * current_coarsening_param - 1) / (static_cast<size_t>(threads) * current_coarsening_param);

        reduceKernel_v4<<<current_blocks, threads, shmem>>>(d_src, d_dst, current_N_elements_to_reduce, identity, op, current_coarsening_param);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error (reduce_v4.cuh) for N_current=%zu, blocks=%zu, threads=%d: %s\n",
                    current_N_elements_to_reduce, current_blocks, threads, cudaGetErrorString(err));
            cudaFree(d_in_raw);
            cudaFree(d_temp_raw);
            return identity;
        }

        current_N_elements_to_reduce = current_blocks;
        T_raw* temp_ptr = d_src;
        d_src = d_dst;
        d_dst = temp_ptr;
    }

    T_raw result;

    if ((err = cudaMemcpy(&result, d_src, sizeof(T_raw), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy result from device to host failed (reduce_v4.cuh): %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_in_raw);
    cudaFree(d_temp_raw);

    return result;
}