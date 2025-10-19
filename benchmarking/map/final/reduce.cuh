template<typename T>
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
__global__ void reduceKernel(
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

    reduceKernel<<<blocks, threads, threads / 32 * sizeof(T), stream>>>(d_in, d_temp, N, identity, op, args...);

    if (blocks > 1) {
        recursiveReduce(stream, d_temp, d_temp, blocks, identity, op, args...);
    }
}

