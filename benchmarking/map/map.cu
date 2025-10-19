#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cmath>

#include "final/fused_gpu_vec.cuh"

// Functor for a simple multiplication operation
struct SimpleComputation {
    __host__ __device__ float operator()(float x) const {
        return x * 2.0f;
    }
};

struct TenSimpleComputations {
    __host__ __device__ float operator()(float x) const {
        return SimpleComputation()(SimpleComputation()(SimpleComputation()(SimpleComputation()(SimpleComputation()(SimpleComputation()(SimpleComputation()(SimpleComputation()(SimpleComputation()(SimpleComputation()(x))))))))));
    }
};

// Benchmark a single map operation, repeated N times
double benchmark_lcuda_simple_map_repetition(size_t n, int repetitions) {
    lcuda::Vector<float> vec1(n, 1.0f);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < repetitions; ++i) {
        vec1 = vec1.map(SimpleComputation());
    }

    vec1.synchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(milliseconds);
}

// Benchmark a single transform operation, repeated N times
double benchmark_thrust_simple_map_repetition(size_t n, int repetitions) {
    thrust::device_vector<float> d_vec(n, 1.0f);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < repetitions; ++i) {
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), SimpleComputation());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(milliseconds);
}

// Benchmark a chained map expression evaluated once
double benchmark_lcuda_chained_map_expression(size_t n, int chain_length) {
    lcuda::Vector<float> vec1(n, 1.0f);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    auto expr = vec1.map(SimpleComputation());
    for (int i = 1; i < chain_length; ++i) {
        expr.map(SimpleComputation());
    }
    vec1 = expr;

    vec1.synchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(milliseconds);
}

double benchmark_lcuda_chained_map_expression_manual(size_t n, int chain_length) {
    lcuda::Vector<float> vec1(n, 1.0f);
    
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vec1 = vec1.map(TenSimpleComputations());

    vec1.synchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(milliseconds);
}

// Benchmark a series of non-fused Thrust transforms
double benchmark_thrust_chained_map_expression(size_t n, int chain_length) {
    thrust::device_vector<float> d_vec(n, 1.0f);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < chain_length; ++i) {
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), SimpleComputation());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(milliseconds);
}



int main() {
    const int num_runs = 20;
    const size_t max_size = 1 << 28; // 2^28 elements
    const size_t min_size = 1 << 20; // 2^20 elements
    const int chain_length = 10;

/*     // Benchmark 1: Simple Map Repetition
    std::ofstream simple_map_csv_file("../sheet/simple_map_repetition.csv");
    if (!simple_map_csv_file.is_open()) {
        std::cerr << "Error: Could not open file ../sheet/simple_map_repetition.csv" << std::endl;
        return 1;
    }
    simple_map_csv_file << "Vector Size,Repetitions,lcuda Time (ms),Thrust Time (ms)" << std::endl;

    std::cout << "Starting Simple Map Repetition benchmark..." << std::endl;
    for (size_t size = min_size; size <= max_size; size *= 2) {
        double total_lcuda_time = 0.0;
        double total_thrust_time = 0.0;
        
        // Warm-up run
        benchmark_lcuda_simple_map_repetition(size, chain_length);
        benchmark_thrust_simple_map_repetition(size, chain_length);

        for (int i = 0; i < num_runs; ++i) {
            total_lcuda_time += benchmark_lcuda_simple_map_repetition(size, chain_length);
            total_thrust_time += benchmark_thrust_simple_map_repetition(size, chain_length);
        }

        double avg_lcuda_time = total_lcuda_time / num_runs;
        double avg_thrust_time = total_thrust_time / num_runs;
        simple_map_csv_file << size << "," << chain_length << "," << avg_lcuda_time << "," << avg_thrust_time << std::endl;
        std::cout << "  Vector Size: " << size << ", Repetitions: " << chain_length << ", lcuda Avg Time: " << avg_lcuda_time << " ms, Thrust Avg Time: " << avg_thrust_time << " ms" << std::endl;
    }
    simple_map_csv_file.close();
    std::cout << "\nBenchmark results written to ../sheet/simple_map_repetition.csv\n" << std::endl;
 */
    // Benchmark 2: Chained Map Expression
    std::ofstream chained_map_csv_file("../sheet/chained_map_bench2.csv");
    if (!chained_map_csv_file.is_open()) {
        std::cerr << "Error: Could not open file ../sheet/chained_map_bench2.csv" << std::endl;
        return 1;
    }
    chained_map_csv_file << "Vector Size,Chain Length,lcuda Time (ms),Thrust Time (ms),lcuda Manual (ms)" << std::endl;

    std::cout << "Starting Chained Map Expression benchmark..." << std::endl;
    for (size_t size = min_size; size <= max_size; size *= 2) {
        double total_lcuda_time = 0.0;
        double total_thrust_time = 0.0;
        double total_lcuda_manual_time = 0.0;    
        // Warm-up run
        benchmark_lcuda_chained_map_expression(size, chain_length);
        benchmark_thrust_chained_map_expression(size, chain_length);
        benchmark_lcuda_chained_map_expression_manual(size, chain_length);
        
        for (int i = 0; i < num_runs; ++i) {
            total_lcuda_time += benchmark_lcuda_chained_map_expression(size, chain_length);
            total_thrust_time += benchmark_thrust_chained_map_expression(size, chain_length);
            total_lcuda_manual_time += benchmark_lcuda_chained_map_expression_manual(size, chain_length);

        }

        double avg_lcuda_time = total_lcuda_time / num_runs;
        double avg_thrust_time = total_thrust_time / num_runs;
        double avg_lcuda_manual_time = total_lcuda_manual_time / num_runs;
        chained_map_csv_file << size << "," << chain_length << "," << avg_lcuda_time << "," << avg_thrust_time << "," << avg_lcuda_manual_time << std::endl;
        std::cout << "  Vector Size: " << size << ", Chain Length: " << chain_length << ", lcuda Avg Time: " << avg_lcuda_time << " ms, Thrust Avg Time: " << avg_thrust_time << " ms, lcuda Manual Avg Time: " << avg_lcuda_manual_time << " ms" << std::endl;
    }
    chained_map_csv_file.close();
    std::cout << "\nBenchmark results written to ../sheet/chained_map_bench2.csv" << std::endl;

    return 0;
}