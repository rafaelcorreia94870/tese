#ifndef COMPOSE_CUH
#define COMPOSE_CUH
#pragma once
#include <stdexcept>
#include <string>

class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t err, const char* file, int line)
        : std::runtime_error(buildErrorMessage(err, file, line)) {}
private:
    static std::string buildErrorMessage(cudaError_t err, const char* file, int line) {
        return std::string("CUDA Error: ") + cudaGetErrorString(err) +
               " in " + file + " at line " + std::to_string(line);
    }
};

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        throw CudaException(e, __FILE__, __LINE__); \
    } \
} while(0)


template<typename T1, typename T2>
struct Pair {
    T1 first;
    T2 second;
    __host__ __device__ Pair() = default;
    __host__ __device__ Pair(T1 f, T2 s) : first(f), second(s) {}
};

template<typename F1, typename F2>
struct ComposeUnaryUnary {
    F1 f1;
    F2 f2;
    __host__ __device__ ComposeUnaryUnary(F1 a, F2 b) : f1(a), f2(b) {}

    template<typename In>
    __host__ __device__ auto operator()(In x) const {
        return f2(f1(x));
    }
};


template<typename F1, typename F2>
struct ComposeBinaryUnary {
    F1 f1;
    F2 f2;
    __host__ __device__ ComposeBinaryUnary(F1 a, F2 b) : f1(a), f2(b) {}

    template<typename In1, typename In2>
    __host__ __device__ auto operator()(In1 x, In2 y) const {
        return f2(f1(x, y));
    }
};

template<typename F1, typename F2>
struct ComposeUnaryBinary {
    F1 f1;
    F2 f2;
    __host__ __device__ ComposeUnaryBinary(F1 a, F2 b) : f1(a), f2(b) {}

    template<typename In1, typename In2>
    __host__ __device__ auto operator()(In1 x, In2 y) const {
        return f2(f1(x), y);
    }
};

template<typename F1, typename F2>
struct ComposeBinaryBinary {
    F1 f1;
    F2 f2;
    __host__ __device__ ComposeBinaryBinary(F1 a, F2 b) : f1(a), f2(b) {}

    template<typename In1, typename In2, typename In3>
    __host__ __device__ auto operator()(In1 x, In2 y, In3 z) const {
        return f2(f1(x, y), z);
    }
};

#endif // COMPOSE_CUH






















