#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

struct DoubleIt {
    __device__ int operator()(int x) const { return 2 * x; }
};

struct AddTen {
    __device__ int operator()(int x) const { return x + 10; }
};

struct SquareIt {
    __device__ int operator()(int x) const { return x * x; }
};

struct Add2Input {
    __device__ int operator()(int x, int y) const { return x + y; }
};

template <typename F1, typename F2>
struct Composed {
    F1 f1;
    F2 f2;

    template <typename Tin>
    __device__ auto operator()(Tin x) const -> decltype(f2(f1(x))) {
        return f2(f1(x));
    }
};


template <typename Expr, typename T>
__global__ void evaluateExprKernel(Expr expr, T* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expr[idx];
    }
}

template <typename T, typename Op>
struct MapExpr {
    const T* d_data;
    size_t len;
    Op op;

    MapExpr(const T* data_ptr, size_t size, Op operation)
        : d_data(data_ptr), len(size), op(operation) {}

    __device__ T operator[](size_t i) const {
        return op(d_data[i]);
    }

    size_t size() const {
        return len;
    }

    template <typename NextOp>
    MapExpr<T, Composed<Op, NextOp>> map(NextOp next_op) const {
        return MapExpr<T, Composed<Op, NextOp>>(d_data, len, Composed<Op, NextOp>{op, next_op});
    }
};

template <typename T>
class Vector {
public:
    std::vector<T> data;
    T* d_data;
    cudaStream_t stream;

    explicit Vector(size_t n) : data(n), d_data(nullptr) {
        cudaStreamCreate(&stream);
    }

    Vector(std::vector<T>& initial_data) : data(initial_data), d_data(nullptr) {
        cudaStreamCreate(&stream);
    }

    size_t size() const { return data.size(); }

    std::vector<T>& get_data() { return data; }

    const std::vector<T>& get_data() const { return data; }

    void print(const std::string& label = "") const {
        if (!label.empty()) {
            std::cout << label << ": ";
        }
        for (const T& val : data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    template <typename Op>
    MapExpr<T, Op> map(Op op) {
        if (!d_data) {
            cudaMalloc(&d_data, data.size() * sizeof(T));
            cudaMemcpyAsync(d_data, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        return MapExpr<T, Op>(d_data, data.size(), op);
    }

    void sync_device_to_host() {
        cudaMemcpyAsync(data.data(), d_data, data.size() * sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    template <typename BaseExpr, typename Op>
    Vector& operator=(const MapExpr<BaseExpr, Op>& expr) {
        if (!d_data) {
            cudaMalloc(&d_data, data.size() * sizeof(T));
        }
        int blockSize = 256;
        int numBlocks = (data.size() + blockSize - 1) / blockSize;
        evaluateExprKernel<<<numBlocks, blockSize, 0, stream>>>(expr, d_data, data.size());
        sync_device_to_host();
        return *this;
    }

    Vector& operator=(const Vector& other) {
        if (this != &other) {
            data = other.data;
        }
        return *this;
    }
};

int main() {
    Vector<int> my_vec(5);
    std::iota(my_vec.get_data().begin(), my_vec.get_data().end(), 1);
    my_vec.print("Original Vector");

    Vector<int> result_vec(5);

    result_vec = my_vec.map(DoubleIt()).map(AddTen()).map(SquareIt());
    result_vec.print("Result 1");

    result_vec = my_vec.map(SquareIt()).map(DoubleIt());
    result_vec.print("Result 2");

    return 0;
}
