#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

#define CUDA_CHECK(err) if((err) != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; exit(1); }


template<typename F1, typename F2>
struct ComposeUnary {
    F1 f1;
    F2 f2;
    __host__ __device__ ComposeUnary(F1 a, F2 b) : f1(a), f2(b) {}
    template<typename In>
    __host__ __device__ auto operator()(In x) const {
        return f2(f1(x));
    }
};

template<typename F1, typename F2>
struct ComposeBinary {
    F1 f1;
    F2 f2;
    __host__ __device__ ComposeBinary(F1 a, F2 b) : f1(a), f2(b) {}
    template<typename In1, typename In2>
    __host__ __device__ auto operator()(In1 x, In2 y) const {
        return f2(f1(x, y));
    }
};

template<typename T>
struct VectorExt;

template<typename T, typename Op>
struct ReduceExpr {
    const T* d_in;
    size_t size;
    Op op;
    T identity;

    ReduceExpr(const T* d, size_t s, Op o, T id) : d_in(d), size(s), op(o), identity(id) {}

    T result() const {
        if (size == 0) return identity;

        T* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, sizeof(T)));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        recursiveReduce(stream, (d_in), d_temp, size, static_cast<T>(identity), op);

        T result_promoted;
        CUDA_CHECK(cudaMemcpyAsync(&result_promoted, d_temp, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_temp));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return (result_promoted);
    }
};

template<typename T, typename Op_map, typename Op_reduce>
struct MapReduceExpr {
    const T* d_in;
    size_t size;
    Op_map op_map;
    Op_reduce op_reduce;
    T identity;

    MapReduceExpr(const T* d, size_t s, Op_map o1, Op_reduce o2, T id) : d_in(d), size(s), op_map(o1), op_reduce(o2), identity(id) {}


    T result() const {
        T* d_reduce_temp;
        CUDA_CHECK(cudaMalloc(&d_reduce_temp, sizeof(T)));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        const int threads = 1024;
        const int itemsPerThread = 8;
        int blocks = (size + threads * itemsPerThread - 1) / (threads * itemsPerThread);

        
        mapReduceFusedKernel<<<blocks, threads, threads / 32 * sizeof(T), stream>>>(
            reinterpret_cast<const T*>(d_in), d_reduce_temp, size, identity, op_map, op_reduce
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (blocks > 1) {
            recursiveReduce(stream, d_reduce_temp, d_reduce_temp, blocks, identity, op_reduce);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        T result_promoted;
        CUDA_CHECK(cudaMemcpyAsync(&result_promoted, d_reduce_temp, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_reduce_temp));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return (result_promoted);
    }

    
};

template<typename T, typename Op_map, typename Op_reduce>
struct MapReduceExprBinary {
    const T* d_in1;
    const T* d_in2;
    size_t size;
    Op_map op_map;
    Op_reduce op_reduce;
    T identity;

    MapReduceExprBinary(const T* d1, const T* d2, size_t s, Op_map o1, Op_reduce o2, T id)
        : d_in1(d1), d_in2(d2), size(s), op_map(o1), op_reduce(o2), identity(id) {}

    T result() const {
        T* d_reduce_temp;
        CUDA_CHECK(cudaMalloc(&d_reduce_temp, sizeof(T)));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        const int threads = 1024;
        const int itemsPerThread = 8;
        int blocks = (size + threads * itemsPerThread - 1) / (threads * itemsPerThread);

        mapReduceFusedKernelBinary<<<blocks, threads, threads / 32 * sizeof(T), stream>>>(
            d_in1, d_in2, d_reduce_temp, size, identity, op_map, op_reduce
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (blocks > 1) {
            recursiveReduce(stream, d_reduce_temp, d_reduce_temp, blocks, identity, op_reduce);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        T result_promoted;
        CUDA_CHECK(cudaMemcpyAsync(&result_promoted, d_reduce_temp, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_reduce_temp));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return (result_promoted);
    }
};


template<typename T, typename Op>
struct MapExprBinary {
    const T* d_in1;
    const T* d_in2;
    size_t size;
    Op op;

    MapExprBinary(const T* a, const T* b, size_t s, Op o) : d_in1(a), d_in2(b), size(s), op(o) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprBinary<T, ComposeUnary<Op, Op2>>(d_in1, d_in2, size, ComposeUnary<Op, Op2>(op, op2));
    }

    template<typename Op2>
    auto reduce(Op2 op2, T identity) const {
        return MapReduceExprBinary<T, Op, Op2>(d_in1, d_in2, size, op, op2, identity).result();
    }
};

template<typename T, typename Op>
struct MapExprUnary {
    const T* d_in;
    size_t size;
    Op op;

    MapExprUnary(const T* d, size_t s, Op o) : d_in(d), size(s), op(o) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprUnary<T, ComposeUnary<Op, Op2>>(d_in, size, ComposeUnary<Op, Op2>(op, op2));
    }

    template<typename Op2>
    auto map(const VectorExt<T>& other, Op2 op2) const {
        return MapExprBinary<T, Op2>(d_in, other.d_data, size, op2);
    }

    template<typename Op2>
    auto reduce(Op2 op2, T identity) const {
        return MapReduceExpr<T, Op, Op2>(d_in, size, op, op2, identity).result();
    }

};

//REDUCE
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


template<typename T_raw>
struct VectorExt {
    using T = Promote<T_raw>;
    T* d_data = nullptr;
    size_t size = 0;

    VectorExt() = default;

    VectorExt(size_t n) : size(n) {
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(T) * size));
    }

    VectorExt(size_t n, T init_value) : size(n) {
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(T) * size));
        std::vector<T> host_vec(size, init_value);
        CUDA_CHECK(cudaMemcpy(d_data, host_vec.data(), sizeof(T) * size, cudaMemcpyHostToDevice));
    }

    VectorExt(const std::vector<T>& host_vec) : size(host_vec.size()) {
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(T) * size));
        CUDA_CHECK(cudaMemcpy(d_data, host_vec.data(), sizeof(T) * size, cudaMemcpyHostToDevice));
    }

    template<typename Op>
    VectorExt(const MapExprUnary<T_raw, Op>& expr) : size(expr.size) {
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(T_raw) * size));
        size_t threads = 1024;
        size_t blocks = (expr.size + threads - 1) / threads;
        kernelUnary<<<blocks, threads>>>(expr.d_in, expr.size, expr.op, d_data);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    template<typename Op>
    VectorExt(const MapExprBinary<T_raw, Op>& expr) : size(expr.size) {
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(T_raw) * size));
        size_t threads = 1024;
        size_t blocks = (expr.size + threads - 1) / threads;
        kernelBinary<<<blocks, threads>>>(expr.d_in1, expr.d_in2, expr.size, expr.op, d_data);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~VectorExt() {
        if (d_data) cudaFree(d_data);
    }

    
    __device__ T& operator[](size_t idx) {
        return d_data[idx];
    }

    VectorExt(const VectorExt&) = delete;
    VectorExt& operator=(const VectorExt&) = delete;

    void copyToHost(std::vector<T>& host_vec) const {
        host_vec.resize(size);
        CUDA_CHECK(cudaMemcpy(host_vec.data(), d_data, sizeof(T) * size, cudaMemcpyDeviceToHost));
    }

    void print(const char* msg) const {
        std::vector<T> host_vec;
        copyToHost(host_vec);
        std::cout << msg << ": ";
        for (auto v : host_vec) std::cout << v << " ";
        std::cout << std::endl;
    }

    template<typename Op>
    MapExprUnary<T, Op> map(Op op) const {
        return MapExprUnary<T, Op>(d_data, size, op);
    }

    template<typename Op>
    MapExprBinary<T, Op> map(const VectorExt& other, Op op) const {
        if (other.size != size) {
            std::cerr << "Error: Vector sizes must match for binary map." << std::endl;
            exit(1);
        }
        return MapExprBinary<T, Op>(d_data, other.d_data, size, op);
    }

    template<typename Op, typename T>
    ReduceExpr<T, Op> reduce(Op op, T identity) const {
        return ReduceExpr<T, Op>(d_data, size, op, identity);
    }

    template<typename Op>
    VectorExt& operator=(const MapExprUnary<T, Op>& expr) {
        size_t threads = 1024;
        size_t blocks = (expr.size + threads - 1) / threads;
        kernelUnary<<<blocks, threads>>>(expr.d_in, expr.size, expr.op, d_data);
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    template<typename Op>
    VectorExt& operator=(const MapExprBinary<T, Op>& expr) {
        size_t threads = 1024;
        size_t blocks = (expr.size + threads - 1) / threads;
        kernelBinary<<<blocks, threads>>>(expr.d_in1, expr.d_in2, expr.size, expr.op, d_data);
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    template<typename Op>
    T& operator=(const ReduceExpr<T, Op>& expr) {
        if (size != 1) {
            std::cerr << "Error: Target vector size must be 1 for reduce assignment." << std::endl;
            exit(1);
        }
        T_raw identity = expr.identity;
        T* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, sizeof(T) * size));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        recursiveReduce(stream, d_data, d_temp, size, static_cast<T>(identity), expr.op);

        T result;
        CUDA_CHECK(cudaMemcpyAsync(&result, d_temp, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_temp));
        CUDA_CHECK(cudaStreamDestroy(stream));

        std::vector<T_raw> host_result(1, static_cast<T_raw>(result));
        CUDA_CHECK(cudaMemcpy(d_data, host_result.data(), sizeof(T_raw), cudaMemcpyHostToDevice));

        return result;
    }

    template<typename Op, typename Op2>
    T& operator=(const MapReduceExpr<T, Op, Op2>& expr) {
        if (size != 1) {
            std::cerr << "Error: Target vector size must be 1 for reduce assignment." << std::endl;
            exit(1);
        }
        T_raw identity = expr.identity;
        T* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, sizeof(T) * size));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        recursiveReduce(stream, d_data, d_temp, size, static_cast<T>(identity), expr.op);

        T result;
        CUDA_CHECK(cudaMemcpyAsync(&result, d_temp, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_temp));
        CUDA_CHECK(cudaStreamDestroy(stream));

        std::vector<T_raw> host_result(1, static_cast<T_raw>(result));
        CUDA_CHECK(cudaMemcpy(d_data, host_result.data(), sizeof(T_raw), cudaMemcpyHostToDevice));

        return result;
    }

    template<typename Op, typename Op2>
    T& operator=(const MapReduceExprBinary<T, Op, Op2>& expr) {
        if (size != 1) {
            std::cerr << "Error: Target vector size must be 1 for reduce assignment." << std::endl;
            exit(1);
        }
        T_raw identity = expr.identity;
        T* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, sizeof(T) * size));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        T result = expr.result();

        std::vector<T_raw> host_result(1, static_cast<T_raw>(result));
        CUDA_CHECK(cudaMemcpy(d_data, host_result.data(), sizeof(T_raw), cudaMemcpyHostToDevice));

        return result;
    }

    template<typename Op>
    T reduce(Op op, T identity) const {
        if (size == 0) return identity;

        T* d_temp;
        cudaMalloc(&d_temp, size * sizeof(T));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        recursiveReduce(stream, d_data, d_temp, size, static_cast<T>(identity), op);
        T result;
        cudaMemcpyAsync(&result, d_temp, sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaFree(d_temp);
        cudaStreamDestroy(stream);

        return result;
    } 
};



template<typename T, typename Op>
__global__ void kernelUnary(const T* input, size_t n, Op op, T* output) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = op(input[idx]);
}

template<typename T, typename Op>
__global__ void kernelBinary(const T* input1, const T* input2, size_t n, Op op, T* output) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = op(input1[idx], input2[idx]);
}



