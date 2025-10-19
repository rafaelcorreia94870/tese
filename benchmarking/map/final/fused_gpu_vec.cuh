#ifndef FUSED_GPU_VEC_CUH
#define FUSED_GPU_VEC_CUH
#pragma once
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
#include "scalar_ops.cuh"

//lazyCUDA namespace
namespace lcuda {

template<typename T_raw>
struct Vector {
    using T = Promote<T_raw>;
    T* d_data = nullptr;
    T* h_data = nullptr;
    mutable int device_changed = 0;
    size_t size = 0;
    cudaStream_t stream = nullptr;

    Vector(){
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    explicit Vector(size_t n) : size(n) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaHostAlloc(&h_data, size * sizeof(T), cudaHostAllocDefault));
        if (h_data == nullptr) {
            throw std::bad_alloc();
        }
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(T)));
    }

    explicit Vector(size_t n, T value) : size(n) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaHostAlloc(&h_data, size * sizeof(T), cudaHostAllocDefault));
        if (h_data == nullptr) {
            throw std::bad_alloc();
        }
        std::fill(h_data, h_data + size, value);
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    explicit Vector(const std::vector<T_raw>& host_vec) : size(host_vec.size()) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaHostAlloc(&h_data, size * sizeof(T), cudaHostAllocDefault));
        if (h_data == nullptr) {
            throw std::bad_alloc();
        }
        std::copy(host_vec.begin(), host_vec.end(), h_data);
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    explicit Vector(const thrust::device_vector<T_raw>& d_vec) : size(d_vec.size()) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaHostAlloc(&h_data, size * sizeof(T), cudaHostAllocDefault));
        if (h_data == nullptr) {
            throw std::bad_alloc();
        }
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_data, thrust::raw_pointer_cast(d_vec.data()), size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    Vector(Vector&& other) noexcept {
        d_data = other.d_data;
        h_data = other.h_data;
        size = other.size;
        stream = other.stream;
        device_changed = other.device_changed;

        other.d_data = nullptr;
        other.h_data = nullptr;
        other.size = 0;
        other.stream = nullptr;
    }

    template<typename Op>
    Vector(const MapExprUnary<T, Op>& expr) {
        this->size = expr.size;
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaHostAlloc(&h_data, size * sizeof(T), cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(T)));
        *this = expr;
    }

    template<typename Op>
    Vector(const MapExprBinary<T, Op>& expr) {
        this->size = expr.size;
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaHostAlloc(&h_data, size * sizeof(T), cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(T)));
        *this = expr;
    }


    ~Vector() {
        if (d_data) {
            cudaFree(d_data);
            d_data = nullptr;
        }
        if (h_data) {
            cudaFreeHost(h_data);
            h_data = nullptr;
        }
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

    __host__ __device__ T& operator[](size_t idx) {
        #ifdef __CUDA_ARCH__
            // Device code path: Direct access to device memory
            return this->d_data[idx];
        #else
            // Host code path: Handles lazy copy from device to host
            if (idx >= size) {
                throw std::out_of_range("Index out of bounds");
            }
            if (device_changed) {
                CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(T), cudaMemcpyDeviceToHost));
                this->device_changed = 0;
            }
            return this->h_data[idx];
        #endif
    }


    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;

    // Synchronous copy to a standard host vector
    void copyToHost(std::vector<T>& host_vec) const {
        host_vec.resize(size);
        CUDA_CHECK(cudaMemcpy(host_vec.data(), this->d_data, sizeof(T) * size, cudaMemcpyDeviceToHost));
    }

    // Asynchronous copy to a standard host vector using a stream
    void copyToHost(std::vector<T>& host_vec, cudaStream_t stream) const {
        host_vec.resize(size);
        CUDA_CHECK(cudaMemcpyAsync(host_vec.data(), this->d_data, sizeof(T) * size, cudaMemcpyDeviceToHost, stream));
    }

    // Asynchronous copy to a standard host vector using a stream
    void copyToDeviceAsync(const std::vector<T_raw>& host_vec) {
        if(host_vec.size() != size) {
            throw std::runtime_error("Vector sizes must match for copy.");
        }
        CUDA_CHECK(cudaMemcpyAsync(d_data, host_vec.data(), sizeof(T) * size, cudaMemcpyHostToDevice, this->stream));
        this->device_changed = 0;
    }

    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(this->stream));
    }

    void fill_with_sequence() {
        size_t threads = 1024;
        size_t blocks = (this->size + threads - 1) / threads;
        sequenceKernel<<<blocks, threads, 0, this->stream>>>(this->d_data, this->size);
        this->device_changed = 1;
    }

    void print(const char* msg) const {
        std::vector<T> host_vec;
        copyToHost(host_vec);
        std::cout << msg << ": ";
        for (auto v : host_vec) std::cout << v << " ";
        std::cout << std::endl;
    }

    void print() const {
        print("Vector");
    }

    template<typename Op>
    MapExprUnary<T, Op> map(Op op) const {
        return MapExprUnary<T, Op>(d_data, size, op, stream);
    }

    template<typename Op>
    MapExprBinary<T, Op> map(const Vector& other, Op op) const {
        if (other.size != size) {
            std::cerr << "Error: Vector sizes must match for binary map." << std::endl;
            exit(1);
        }
        return MapExprBinary<T, Op>(d_data, other.d_data, size, op, stream);
    }

    template<typename Op, typename T_out1, typename T_out2>
    void map(Op op, Vector<T_out1>& out1, Vector<T_out2>& out2) {
        if (out1.size != this->size || out2.size != this->size) {
            std::cerr << "Error: Output vector sizes must match input size for two-output map." << std::endl;
            exit(1);
        }
        size_t threads = 1024;
        size_t blocks = (this->size + threads - 1) / threads;
        kernelUnaryTwoOutputs<<<blocks, threads, 0, this->stream>>>(this->d_data, this->size, op, out1.d_data, out2.d_data);
        out1.device_changed = 1;
        out2.device_changed = 1;
    }

    template<typename Op, typename T_in2, typename T_out1, typename T_out2>
    void map(const Vector<T_in2>& in2, Op op, Vector<T_out1>& out1, Vector<T_out2>& out2) {
        if (in2.size != this->size || out1.size != this->size || out2.size != this->size) {
            std::cerr << "Error: All vector sizes must match for binary two-output map." << std::endl;
            exit(1);
        }
        size_t threads = 1024;
        size_t blocks = (this->size + threads - 1) / threads;
        kernelBinaryTwoOutputs<<<blocks, threads, 0, this->stream>>>(this->d_data, in2.d_data, this->size, op, out1.d_data, out2.d_data);
        out1.device_changed = 1;
        out2.device_changed = 1;
    }

    template<typename Op, typename T_out2>
    void map(Op op, Vector<T_out2>& out2) {
        if (out2.size != this->size) {
            std::cerr << "Error: Output vector size must match input size." << std::endl;
            exit(1);
        }
        size_t threads = 1024;
        size_t blocks = (this->size + threads - 1) / threads;
        kernelUnaryTwoOutputs<<<blocks, threads, 0, this->stream>>>(this->d_data, this->size, op, this->d_data, out2.d_data);
        this->device_changed = 1;
        out2.device_changed = 1;
    }

    template<typename Op, typename T_in2, typename T_out2>
    void map(const Vector<T_in2>& in2, Op op, Vector<T_out2>& out2) {
        if (in2.size != this->size || out2.size != this->size) {
            std::cerr << "Error: All vector sizes must match." << std::endl;
            exit(1);
        }
        size_t threads = 1024;
        size_t blocks = (this->size + threads - 1) / threads;
        kernelBinaryTwoOutputs<<<blocks, threads, 0, this->stream>>>(this->d_data, in2.d_data, this->size, op, this->d_data, out2.d_data);
        this->device_changed = 1;
        out2.device_changed = 1;
    }

    template<typename Op>
    ReduceExpr<T, Op> reduce(Op op, T identity) const {
        return ReduceExpr<T, Op>(d_data, size, op, identity, stream);
    }

    Vector& operator=(Vector&& other) noexcept {
        if (this != &other) {
            if (d_data) cudaFree(d_data);
            if (h_data) cudaFreeHost(h_data);
            if (stream) cudaStreamDestroy(stream);

            d_data = other.d_data;
            h_data = other.h_data;
            size = other.size;
            stream = other.stream;
            device_changed = other.device_changed;

            other.d_data = nullptr;
            other.h_data = nullptr;
            other.size = 0;
            other.stream = nullptr;
        }
            return *this;
        }
    

    template<typename Op>
    Vector& operator=(const MapExprUnary<T, Op>& expr) {
        if (this->d_data == nullptr || this->size != expr.size) {
             if (this->d_data) cudaFree(this->d_data);
             if (this->h_data) cudaFreeHost(this->h_data);
             this->size = expr.size;
             CUDA_CHECK(cudaHostAlloc(&h_data, size * sizeof(T), cudaHostAllocDefault));
             CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(T)));
        }
        size_t threads = 1024;
        size_t blocks = (expr.size + threads - 1) / threads;
        kernelUnary<<<blocks, threads>>>(expr.d_in, expr.size, expr.op, this->d_data);
        this->device_changed = 1;
        return *this;
    }

    template<typename Op>
    Vector& operator=(const MapExprBinary<T, Op>& expr) {
        if (this->d_data == nullptr || this->size != expr.size) {
            if (this->d_data) cudaFree(this->d_data);
            if (this->h_data) cudaFreeHost(this->h_data);
            this->size = expr.size;
            CUDA_CHECK(cudaHostAlloc(&h_data, size * sizeof(T), cudaHostAllocDefault));
            CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(T)));
        }
        size_t threads = 1024;
        size_t blocks = (expr.size + threads - 1) / threads;
        kernelBinary<<<blocks, threads>>>(expr.d_in1, expr.d_in2, expr.size, expr.op, this->d_data);
        this->device_changed = 1;
        return *this;
    }

    template<typename Op>
    T operator=(const ReduceExpr<T, Op>& expr) {
        /* if (size != 1) {
            std::cerr << "Error: Target vector size must be 1 for reduce assignment." << std::endl;
            exit(1);
        }
        T identity = expr.identity;
        T* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, sizeof(T) * size));

        recursiveReduce(stream, this->d_data, d_temp, size, identity, expr.op);

        T result;
        CUDA_CHECK(cudaMemcpyAsync(&result, d_temp, sizeof(T), cudaMemcpyDeviceToHost, this->stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_temp));

        std::vector<T> host_result(1, result);
        CUDA_CHECK(cudaMemcpy(d_data, host_result.data(), sizeof(T), cudaMemcpyHostToDevice));

        return result; */

        return expr.result();
    }

    template<typename Op, typename Op2>
    T operator=(const MapReduceExpr<T, Op, Op2>& expr) {
        /* if (size != 1) {
            std::cerr << "Error: Target vector size must be 1 for reduce assignment." << std::endl;
            exit(1);
        }
        T identity = expr.identity;
        T result;
        T* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, sizeof(T) * size));

        recursiveReduce(stream, this->d_data, d_temp, size, identity, expr.op);

        CUDA_CHECK(cudaMemcpyAsync(&result, d_temp, sizeof(T), cudaMemcpyDeviceToHost, this->stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_temp));

        std::vector<T> host_result(1, result);
        CUDA_CHECK(cudaMemcpy(d_data, host_result.data(), sizeof(T), cudaMemcpyHostToDevice));

        return result; */
        return expr.result();
    }

    template<typename Op, typename Op2>
    T operator=(const MapReduceExprBinary<T, Op, Op2>& expr) {
       /*  if (size != 1) {
            std::cerr << "Error: Target vector size must be 1 for reduce assignment." << std::endl;
            exit(1);
        }
        T_raw identity = expr.identity;
        T* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, sizeof(T) * size));


        T result = expr.result();

        std::vector<T_raw> host_result(1, static_cast<T_raw>(result));
        CUDA_CHECK(cudaMemcpyAsync(d_temp, &result, sizeof(T_raw), cudaMemcpyHostToDevice, this->stream));

        return result; */

        return expr.result();
    }

    //[] operator for const access
    __host__ __device__ const T& operator[](size_t idx) const {
        #ifdef __CUDA_ARCH__
            // Device code path: Direct access to device memory
            return this->d_data[idx];
        #else
            // Host code path: Handles lazy copy from device to host
            if (idx >= size) {
                throw std::out_of_range("Index out of bounds");
            }
            if (device_changed) {
                CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(T), cudaMemcpyDeviceToHost));
                this->device_changed = 0;
            }
            return this->h_data[idx];
        #endif
    }
};

}



#endif // FUSED_GPU_VEC_CUH