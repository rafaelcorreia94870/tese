#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>

#define CUDA_CHECK(err) if((err) != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; exit(1); }

struct DoubleIt {
    __host__ __device__ int operator()(int x) const { return 2 * x; }
};
struct AddTen {
    __host__ __device__ int operator()(int x) const { return x + 10; }
};
struct SquareIt {
    __host__ __device__ int operator()(int x) const { return x * x; }
};
struct Add2Input {
    __host__ __device__ int operator()(int x, int y) const { return x + y; }
};

template<typename F1, typename F2>
struct ComposeUnary {
    F1 f1;
    F2 f2;
    __host__ __device__ ComposeUnary(F1 a, F2 b) : f1(a), f2(b) {}
    __host__ __device__ int operator()(int x) const {
        return f2(f1(x));
    }
};

template<typename F1, typename F2>
struct ComposeBinary {
    F1 f1;
    F2 f2;
    __host__ __device__ ComposeBinary(F1 a, F2 b) : f1(a), f2(b) {}
    __host__ __device__ int operator()(int x, int y) const {
        return f2(f1(x, y));
    }
};

struct VectorExt;

template<typename Op>
struct MapExprUnary {
    const int* d_in;
    size_t size;
    Op op;

    MapExprUnary(const int* d, size_t s, Op o) : d_in(d), size(s), op(o) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprUnary<ComposeUnary<Op, Op2>>(d_in, size, ComposeUnary<Op, Op2>(op, op2));
    }

    template<typename Op2>
    auto map(const VectorExt& other, Op2 op2) const;
};

template<typename Op>
struct MapExprBinary {
    const int* d_in1;
    const int* d_in2;
    size_t size;
    Op op;

    MapExprBinary(const int* a, const int* b, size_t s, Op o) : d_in1(a), d_in2(b), size(s), op(o) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprBinary<ComposeUnary<Op, Op2>>(d_in1, d_in2, size, ComposeUnary<Op, Op2>(op, op2));
    }
};

struct VectorExt {
    int* d_data = nullptr;
    size_t size = 0;

    VectorExt() = default;

    VectorExt(size_t n) : size(n) {
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(int) * size));
    }

    VectorExt(const std::vector<int>& host_vec) : size(host_vec.size()) {
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(int) * size));
        CUDA_CHECK(cudaMemcpy(d_data, host_vec.data(), sizeof(int) * size, cudaMemcpyHostToDevice));
    }

    ~VectorExt() {
        if (d_data) cudaFree(d_data);
    }

    VectorExt(const VectorExt&) = delete;
    VectorExt& operator=(const VectorExt&) = delete;

    void copyToHost(std::vector<int>& host_vec) const {
        host_vec.resize(size);
        CUDA_CHECK(cudaMemcpy(host_vec.data(), d_data, sizeof(int) * size, cudaMemcpyDeviceToHost));
    }

    void print(const char* msg) const {
        std::vector<int> host_vec;
        copyToHost(host_vec);
        std::cout << msg << ": ";
        for (auto v : host_vec) std::cout << v << " ";
        std::cout << std::endl;
    }

    template<typename Op>
    MapExprUnary<Op> map(Op op) const {
        return MapExprUnary<Op>(d_data, size, op);
    }

    template<typename Op>
    MapExprBinary<Op> map(const VectorExt& other, Op op) const {
        if (other.size != size) {
            std::cerr << "Error: Vector sizes must match for binary map." << std::endl;
            exit(1);
        }
        return MapExprBinary<Op>(d_data, other.d_data, size, op);
    }

    template<typename Op>
    VectorExt& operator=(const MapExprUnary<Op>& expr) {
        size_t threads = 256;
        size_t blocks = (expr.size + threads - 1) / threads;
        kernelUnary<<<blocks, threads>>>(expr.d_in, expr.size, expr.op, d_data);
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    template<typename Op>
    VectorExt& operator=(const MapExprBinary<Op>& expr) {
        size_t threads = 256;
        size_t blocks = (expr.size + threads - 1) / threads;
        kernelBinary<<<blocks, threads>>>(expr.d_in1, expr.d_in2, expr.size, expr.op, d_data);
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }
};

template<typename Op>
template<typename Op2>
auto MapExprUnary<Op>::map(const VectorExt& other, Op2 op2) const {
    return MapExprBinary<Op2>(d_in, other.d_data, size, op2);
}

template<typename Op>
__global__ void kernelUnary(const int* input, size_t n, Op op, int* output) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = op(input[idx]);
}

template<typename Op>
__global__ void kernelBinary(const int* input1, const int* input2, size_t n, Op op, int* output) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = op(input1[idx], input2[idx]);
}

int main() {
    std::vector<int> initial(5);
    std::iota(initial.begin(), initial.end(), 1);

    VectorExt v(initial);
    v.print("Original v");

    VectorExt result(5);

    result = v.map(DoubleIt()).map(AddTen()).map(SquareIt());
    result.print("Result unary chain");

    result = v.map(SquareIt()).map(DoubleIt());
    result.print("Result unary chain 2");

    VectorExt v2(initial);
    std::vector<int> initial2(5);
    std::iota(initial2.begin(), initial2.end(), 11);
    VectorExt v3(initial2);

    VectorExt result2(5);
    result2 = v2.map(DoubleIt()).map(v3, Add2Input());
    result2.print("Result binary map");

    return 0;
}
