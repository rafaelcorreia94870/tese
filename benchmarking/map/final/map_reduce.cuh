template <typename T_raw, typename UnaryOp, typename BinaryOp, typename... Args>
__global__ void mapReduceFusedKernel(
    const T_raw* __restrict__ d_in, T_raw* __restrict__ d_out,
    int N, T_raw identity, UnaryOp op_map, BinaryOp op_reduce, Args... args)
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
            T mapped_val = op_map(static_cast<T>(d_in[offset]));
            val = op_reduce(val, mapped_val, args...);
        }
    }

    val = warpReduce<T>(val, op_reduce, args...);

    if ((threadIdx.x & 31) == 0) {
        sdata[threadIdx.x / 32] = val;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        val = (threadIdx.x < (blockDim.x / 32)) ? sdata[threadIdx.x] : static_cast<T>(identity);
        val = warpReduce<T>(val, op_reduce, args...);
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = static_cast<T_raw>(val);
    }
}

template <typename T_raw, typename BinaryOpMap, typename BinaryOpReduce, typename... Args>
__global__ void mapReduceFusedKernelBinary(
    const T_raw* __restrict__ d_in1, const T_raw* __restrict__ d_in2, T_raw* __restrict__ d_out,
    int N, T_raw identity, BinaryOpMap op_map, BinaryOpReduce op_reduce, Args... args)
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
            T mapped_val = op_map(static_cast<T>(d_in1[offset]), static_cast<T>(d_in2[offset]));
            val = op_reduce(val, mapped_val, args...);
        }
    }

    val = warpReduce<T>(val, op_reduce, args...);

    if ((threadIdx.x & 31) == 0) {
        sdata[threadIdx.x / 32] = val;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        val = (threadIdx.x < (blockDim.x / 32)) ? sdata[threadIdx.x] : static_cast<T>(identity);
        val = warpReduce<T>(val, op_reduce, args...);
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = static_cast<T_raw>(val);
    }
}
