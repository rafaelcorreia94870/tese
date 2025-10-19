#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <cuda_runtime.h>
#include <cmath>

template<typename T>
struct Add {
    __host__ __device__ T operator()(T x, T y) const {
        return x + y;
    }
};

struct Multiply {
    __host__ __device__ float operator()(float x, float y) const {
        return x * y;
    }
};

struct Maxf {
    __host__ __device__ float operator()(float x, float y) const {
        return fmaxf(x, y);
    }
};

struct Minf {
    __host__ __device__ float operator()(float x, float y) const {
        return fminf(x, y);
    }
};


template<typename T>
struct Square {
    __host__ __device__ T operator()(T x) const {
        return x * x;
    }
};

struct SimpleComputation {
    __host__ __device__ float operator()(float x) const {
        return x * 2.0f;
    }
};

struct TwoSimpleComputations {
    __host__ __device__ float operator()(float x) const {
        float result = x * 2.0f;
        return result * 2.0f;
    }
}; 

template<typename T>
struct DoubleIt {
    __host__ __device__ T operator()(T x) const {
        return x * 2.0f;
    }
};

struct SumSquares {
    __host__ __device__ float operator()(float x) const {
        return x * x;
    }
};

struct Product {
    __host__ __device__ float operator()(float x, float y) const {
        return x * y;
    }
};

struct BenchmarkingComputations {
    int iterations;

    __host__ __device__ BenchmarkingComputations(int iters) : iterations(iters) {}

    __host__ __device__ float operator()(float x) const {
        float result = x;
        for (int i = 0; i < iterations; ++i) {
            result += sinf(result) * cosf(result) + logf(result + 1.0f);
        }
        return result;
    }
};

template<typename T>
struct IntensiveComputation {
    __host__ __device__ T operator()(T x) const {
        for (int i = 0; i < 100; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
        }
        return x;
    }
};

template<typename T>
struct TwoIntensiveComputations {
    __host__ __device__ T operator()(T x) const {
        for (int i = 0; i < 100; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
        }
        for (int i = 0; i < 100; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
        }
        return x;
    }
};

template<typename T>
struct ComplexAdd {
    __host__ __device__ T operator()(T x, T y) const {
        T x1 = x * 0.01f;
        T y1 = y * 0.01f;
        x1 = sinf(x1) * cosf(x1) + logf(abs(x1) + 1.0f);
        y1 = sinf(y1) * cosf(y1) + logf(abs(y1) + 1.0f);
        return x1 + y1 + (x1 * y1);
    }
};

struct Plus {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b) const {
        return a + b;
    }
};

struct Min {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b) const {
        return (a < b) ? a : b;
    }
};

struct Max {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b) const {
        return (a > b) ? a : b;
    }
};

//forward declaration of Pair
template<typename T1, typename T2>
struct Pair; 

template<typename T>
struct SineCosine {
    __host__ __device__ Pair<T, T> operator()(T x) const {
        return Pair<T,T>(sinf(x), cosf(x));
    }
};

template<typename T>
struct SumAndProduct {
    __host__ __device__ Pair<T, T> operator()(T x, T y) const {
        return Pair<T,T>(x + y, x * y);
    }
};

#endif // OPERATIONS_H
