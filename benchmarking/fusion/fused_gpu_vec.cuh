#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include "kernel_op.cuh"

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
    auto map(const VectorExt<T>& other, Op2 op2) const;
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
};

template<typename T>
struct VectorExt {
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

    ~VectorExt() {
        if (d_data) cudaFree(d_data);
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
};

template<typename T, typename Op>
template<typename Op2>
auto MapExprUnary<T, Op>::map(const VectorExt<T>& other, Op2 op2) const {
    return MapExprBinary<T, Op2>(d_in, other.d_data, size, op2);
}

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

std::chrono::duration<double> twointensivecomputations_gpu_vec(size_t n, int loop_count) {
    auto start = std::chrono::high_resolution_clock::now();

    VectorExt<float> vec1(n, 1.0f);

    VectorExt<float> result(n);
    result = vec1.map(BenchmarkingComputations(loop_count))
                 .map(BenchmarkingComputations(loop_count));

    auto end = std::chrono::high_resolution_clock::now();
    //result.print("Result of two intensive computations");
    return end - start;
}

std::chrono::duration<double> tensimplecomputations_gpu_vec(size_t n){
    auto start = std::chrono::high_resolution_clock::now();

    VectorExt<float> vec1(n, 1.0f);
    VectorExt<float> result(n);

    result = vec1.map(SimpleComputation()).map(SimpleComputation())
                                  .map(SimpleComputation()).map(SimpleComputation())
                                  .map(SimpleComputation()).map(SimpleComputation())
                                  .map(SimpleComputation()).map(SimpleComputation())
                                  .map(SimpleComputation()).map(SimpleComputation());

    auto end = std::chrono::high_resolution_clock::now();

    return end - start;
}
