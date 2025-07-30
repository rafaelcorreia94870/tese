#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

#include "compose.cuh"
#include "expression.cuh"
#include "map.cuh"
#include "reduce.cuh"
#include "map_reduce.cuh"

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
    explicit VectorExt(const MapExprBinary<T_raw, Op>& expr) : size(expr.size) {
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

    /* template<typename Op, typename T_out1, typename T_out2>
    MapExprUnaryOutput<T, Op, T_out1, T_out2> map(Op op, VectorExt<T_out1>& out1, VectorExt<T_out2>& out2) const {
        if (out1.size != this->size || out2.size != this->size) {
            std::cerr << "Error: Output vector sizes must match input size for two-output map." << std::endl;
            exit(1);
        }
        return MapExprUnaryOutput<T, Op, T_out1, T_out2>(d_data, size, op, out1.d_data, out2.d_data);
    }

    template<typename Op, typename T_in2, typename T_out1, typename T_out2>
    MapExprBinaryOutput<T, Op, T_out1, T_out2> map(const VectorExt<T_in2>& in2, Op op, VectorExt<T_out1>& out1, VectorExt<T_out2>& out2) const {
        if (in2.size != this->size || out1.size != this->size || out2.size != this->size) {
            std::cerr << "Error: All vector sizes must match for binary two-output map." << std::endl;
            exit(1);
        }
        return MapExprBinaryOutput<T, Op, T_out1, T_out2>(d_data, in2.d_data, size, op, out1.d_data, out2.d_data);
    } */

    template<typename Op, typename T_out2>
    void map(Op op, VectorExt<T_out2>& out2) {
        if (out2.size != this->size) {
            std::cerr << "Error: Output vector size must match input size." << std::endl;
            exit(1);
        }
        size_t threads = 1024;
        size_t blocks = (this->size + threads - 1) / threads;
        kernelUnaryTwoOutputs<<<blocks, threads>>>(this->d_data, this->size, op, this->d_data, out2.d_data);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    template<typename Op, typename T_in2, typename T_out2>
    void map(const VectorExt<T_in2>& in2, Op op, VectorExt<T_out2>& out2) {
        if (in2.size != this->size || out2.size != this->size) {
            std::cerr << "Error: All vector sizes must match." << std::endl;
            exit(1);
        }
        size_t threads = 1024;
        size_t blocks = (this->size + threads - 1) / threads;
        kernelBinaryTwoOutputs<<<blocks, threads>>>(this->d_data, in2.d_data, this->size, op, this->d_data, out2.d_data);
        CUDA_CHECK(cudaDeviceSynchronize());
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
    T& operator=(const MapExprUnaryOutput<T, Op, T, T>& expr) {
        if (size != 1) {
            std::cerr << "Error: Target vector size must be 1 for unary output assignment." << std::endl;
            exit(1);
        }
        size_t threads = 1024;
        size_t blocks = (expr.size + threads - 1) / threads;
        kernelUnaryOutput<<<blocks, threads>>>(expr.d_in, expr.size, expr.op, d_data, expr.out1, expr.out2);
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    template<typename Op>
    T& operator=(const MapExprBinaryOutput<T, Op, T, T>& expr) {
        if (size != 1) {
            std::cerr << "Error: Target vector size must be 1 for binary output assignment." << std::endl;
            exit(1);
        }
        size_t threads = 1024;
        size_t blocks = (expr.size + threads - 1) / threads;
        kernelBinaryOutput<<<blocks, threads>>>(expr.d_in1, expr.d_in2, expr.size, expr.op, d_data, expr.out1, expr.out2);
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
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



