#pragma once
struct BenchmarkingComputations {
    __host__ __device__ BenchmarkingComputations(int iters) : iterations(iters) {}
    int iterations;
    __device__ float operator()(float x) const {
        for (int i = 0; i < iterations; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
        }
        return x;
    }
};

struct SimpleComputation {
    __host__ __device__ SimpleComputation() {}
    __device__ float operator()(float x) const {
        return sin(x) * cos(x) + log(x + 1.0f);
    }
};
