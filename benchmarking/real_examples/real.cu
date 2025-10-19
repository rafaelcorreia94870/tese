#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <cmath>
#include <algorithm>

#include "../map/final/fused_gpu_vec.cuh"


#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

struct RandomGenerator {
    unsigned int seed;

    __host__ __device__ RandomGenerator(unsigned int s) : seed(s) {}

    __host__ __device__ double operator()(const size_t n) const {
        thrust::default_random_engine rng(seed + n);
        thrust::uniform_real_distribution<double> dist(-100.0, 100.0);
        return dist(rng);
    }
};



void generate_random_data(thrust::device_vector<double>& d_vec) {
    thrust::transform(thrust::make_counting_iterator(static_cast<size_t>(0)),
                      thrust::make_counting_iterator(d_vec.size()),
                      d_vec.begin(),
                      RandomGenerator(12345));
}

bool check_results(const std::vector<double>& h_result, const std::vector<double>& h_expected) {
    for (size_t i = 0; i < h_expected.size(); ++i) {
        if (std::abs(h_result[i] - h_expected[i]) > std::numeric_limits<double>::epsilon() * 100) {
            std::cerr << "Mismatch at index " << i << ": expected " << h_expected[i] << ", got " << h_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

void cpu_normalize(std::vector<double>& vec) {
    double sum_of_squares = 0.0f;
    for (double x : vec) {
        sum_of_squares += x * x;
    }
    double norm = sqrt(sum_of_squares);
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] /= norm;
    }
}

double cpu_relu_sum(const std::vector<double>& vec) {
    double sum = 0.0;
    for (double x : vec) {
        sum += std::max(0.0, x);
    }
    return sum;
}


struct Square {
    __host__ __device__ double operator()(double x) const {
        return x * x;
    }
};

struct Plus {
    __host__ __device__ double operator()(double x, double y) const {
        return x + y;
    }
};

__global__ void fused_normalize_kernel(double* data, double norm, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] /= norm;
    }
}


struct ReLU {
    __host__ __device__ double operator()(double x) const {
        return fmax(0.0, x);
    }
};

// --- VECTOR NORMALIZATION BENCHMARK ---
// Fused Lcuda version: map (square) -> reduce (sum) -> map (divide)
double benchmark_lcuda_normalize_fused(lcuda::Vector<double>& vec) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    //vec.print("\n\nInitial vector");

    double sum_of_squares = vec.map(Square()).reduce(Plus(), 0.0f);

    //vec.print("\n\nBefore normalization");

    double norm = sqrt(sum_of_squares);

    vec = vec / norm;

    //vec.print("\n\nAfter normalization");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(milliseconds);
}

double benchmark_thrust_normalize_manually_fused(lcuda::Vector<double>& vec){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    //vec.print("\n\nInitial vector");

    double sum_of_squares = vec.map(Square()).reduce(Plus(), 0.0f);

    //vec.print("\n\nBefore normalization");

    double norm = sqrt(sum_of_squares);

    vec = vec / norm;

    //vec.print("\n\nAfter normalization");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(milliseconds);
}

// Non-fused Thrust version
double benchmark_thrust_normalize_non_fused(thrust::device_vector<double>& d_vec) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    thrust::device_vector<double> d_squared(d_vec.size());
    thrust::transform(d_vec.begin(), d_vec.end(), d_squared.begin(), Square());

    double sum_of_squares = thrust::reduce(d_squared.begin(), d_squared.end());

    double norm = sqrt(sum_of_squares);
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), thrust::placeholders::_1 / norm);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(milliseconds);
}


// --- FUSED THRUST IMPLEMENTATION ---
// Fused Thrust version
double benchmark_thrust_normalize_fused(thrust::device_vector<double>& d_vec) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);


    thrust::device_vector<double> d_squared(d_vec.size());
    thrust::transform(d_vec.begin(), d_vec.end(), d_squared.begin(), Square());

    double sum_of_squares = thrust::reduce(d_squared.begin(), d_squared.end());

    double norm = sqrt(sum_of_squares);
    
    fused_normalize_kernel<<<256, 128>>>(thrust::raw_pointer_cast(d_vec.data()), norm, d_vec.size());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(milliseconds);
}

// --- RELU ACTIVATION + SUM BENCHMARK ---
// Correctness function: returns the final sum, with NO timing.
double get_lcuda_relu_sum_fused(lcuda::Vector<double>& vec) {
    return vec.map(ReLU()).reduce(Plus(), 0.0);
}

// Timing function: runs the benchmark and returns the time in milliseconds.
double benchmark_lcuda_relu_sum_fused(lcuda::Vector<double>& vec) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    double sum = vec.map(ReLU()).reduce(Plus(), 0.0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(milliseconds);
}

// Correctness function: runs the benchmark and returns the final sum.
double get_thrust_relu_sum_non_fused(thrust::device_vector<double>& d_vec) {
    thrust::device_vector<double> d_relu(d_vec.size());
    thrust::transform(d_vec.begin(), d_vec.end(), d_relu.begin(), ReLU());
    return thrust::reduce(d_relu.begin(), d_relu.end());
}

// Timing function: runs the benchmark and returns the time in milliseconds.
double benchmark_thrust_relu_sum_non_fused(thrust::device_vector<double>& d_vec) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    thrust::device_vector<double> d_relu(d_vec.size());
    thrust::transform(d_vec.begin(), d_vec.end(), d_relu.begin(), ReLU());
    thrust::reduce(d_relu.begin(), d_relu.end());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(milliseconds);
}

int main() {
    const int num_runs = 100;
    const size_t max_size = 1 << 27;  
    const size_t min_size = 1 << 15;

    // --- BENCHMARK 1: VECTOR NORMALIZATION ---
    std::ofstream norm_csv_file("../sheet/real_examples/vector_normalization_bench.csv");
    if (!norm_csv_file.is_open()) {
        std::cerr << "Error: Could not open file ../sheet/real_examples/vector_normalization_bench.csv" << std::endl;
        return 1;
    }
    norm_csv_file << "Vector Size,lcuda Time (ms),Thrust Time (ms)" << std::endl;

    std::cout << "Starting Vector Normalization benchmark..." << std::endl;
    for (size_t size = min_size; size <= max_size; size *= 2) {
        thrust::device_vector<double> d_data(size);
        generate_random_data(d_data);

        // Correctness check
        std::vector<double> h_data(size);
        thrust::copy(d_data.begin(), d_data.end(), h_data.begin());
        std::vector<double> h_expected_norm = h_data;
        cpu_normalize(h_expected_norm);

        // Thrust correctness check
        thrust::device_vector<double> d_thrust_check = d_data;
        benchmark_thrust_normalize_non_fused(d_thrust_check);
        thrust::host_vector<double> h_thrust_result = d_thrust_check;

        std::vector<double> h_thrust_result_vec(size);
        thrust::copy(h_thrust_result.begin(), h_thrust_result.end(), h_thrust_result_vec.begin());
        if (!check_results(h_thrust_result_vec, h_expected_norm)) {
            std::cerr << "Thrust normalization results are incorrect!" << std::endl;
            return 1;
        }

        // Lcuda correctness check
        lcuda::Vector<double> l_vec_check(d_data);
        benchmark_lcuda_normalize_fused(l_vec_check);
        std::vector<double> h_lcuda_result(size);
        l_vec_check.copyToHost(h_lcuda_result); // **FIXED**: Explicitly copy to host for comparison.
        if (!check_results(h_lcuda_result, h_expected_norm)) {
            std::cerr << "Lcuda normalization results are incorrect!" << std::endl;
            return 1;
        }

        // Timing runs
        thrust::device_vector<double> d_vec_warmup1 = d_data;
        lcuda::Vector<double> l_vec_warmup1(d_data);
        benchmark_lcuda_normalize_fused(l_vec_warmup1);
        benchmark_thrust_normalize_non_fused(d_vec_warmup1);

        double total_lcuda_time = 0.0;
        double total_thrust_time = 0.0;
        double total_thrust_fused_time = 0.0;
        for (int i = 0; i < num_runs; ++i) {
            thrust::device_vector<double> d_vec_run = d_data;
            lcuda::Vector<double> l_vec_run(d_data);
            total_lcuda_time += benchmark_lcuda_normalize_fused(l_vec_run);
            total_thrust_time += benchmark_thrust_normalize_non_fused(d_vec_run);
            total_thrust_fused_time += benchmark_thrust_normalize_fused(d_vec_run);
        }

        double avg_lcuda_time = total_lcuda_time / num_runs;
        double avg_thrust_time = total_thrust_time / num_runs;
        double avg_thrust_fused_time = total_thrust_fused_time / num_runs;
        norm_csv_file << size << "," << avg_lcuda_time << "," << avg_thrust_time << "," << avg_thrust_fused_time << std::endl;
        std::cout << "  Vector Size: " << size << ", lcuda Avg Time: " << avg_lcuda_time << " ms, Thrust Avg Time: " << avg_thrust_time << " ms, Thrust Fused Avg Time: " << avg_thrust_fused_time << " ms" << std::endl;
    }
    norm_csv_file.close();
    std::cout << "\nBenchmark results written to ../sheet/real_examples/vector_normalization_bench.csv\n" << std::endl;

    // --- BENCHMARK 2: RELU + SUM ---
    std::ofstream relu_csv_file("../sheet/real_examples/relu_sum_bench.csv");
    if (!relu_csv_file.is_open()) {
        std::cerr << "Error: Could not open file ../sheet/real_examples/relu_sum_bench.csv" << std::endl;
        return 1;
    }
    relu_csv_file << "Vector Size,lcuda Time (ms),Thrust Time (ms)" << std::endl;

    std::cout << "Starting ReLU + Sum benchmark..." << std::endl;
    for (size_t size = min_size; size <= max_size; size *= 2) {
        thrust::device_vector<double> d_data(size);
        generate_random_data(d_data);
        
        std::vector<double> h_data(size);
        thrust::copy(d_data.begin(), d_data.end(), h_data.begin());
        
        double h_expected_sum = cpu_relu_sum(h_data);
        
        // Thrust correctness check
        thrust::device_vector<double> d_thrust_check = d_data;
        double d_thrust_sum = get_thrust_relu_sum_non_fused(d_thrust_check);
        if (std::abs(d_thrust_sum - h_expected_sum) > 1.0e-5) {
                std::cerr << "Thrust ReLU+Sum results are incorrect!" << std::endl;
                std::cerr << "Expected: " << h_expected_sum << ", Got: " << d_thrust_sum << std::endl;
                return 1;
        }
        
        // Lcuda correctness check
        lcuda::Vector<double> l_vec_check(d_data);
        double d_lcuda_sum = get_lcuda_relu_sum_fused(l_vec_check);
        if (std::abs(d_lcuda_sum - h_expected_sum) > 1.0e-5) {
                std::cerr << "Lcuda ReLU+Sum results are incorrect!" << std::endl;
                std::cerr << "Expected: " << h_expected_sum << ", Got: " << d_lcuda_sum << std::endl;
                return 1;
        }
        
        // Timing runs
        thrust::device_vector<double> d_vec_warmup2 = d_data;
        lcuda::Vector<double> l_vec_warmup2(d_data);
        benchmark_lcuda_relu_sum_fused(l_vec_warmup2);
        benchmark_thrust_relu_sum_non_fused(d_vec_warmup2);
        
        double total_lcuda_time = 0.0;
        double total_thrust_time = 0.0;
        for (int i = 0; i < num_runs; ++i) {
                thrust::device_vector<double> d_vec_run = d_data;
                lcuda::Vector<double> l_vec_run(d_data);
                total_lcuda_time += benchmark_lcuda_relu_sum_fused(l_vec_run);
                total_thrust_time += benchmark_thrust_relu_sum_non_fused(d_vec_run);
        }
        
        double avg_lcuda_time = total_lcuda_time / num_runs;
        double avg_thrust_time = total_thrust_time / num_runs;
        relu_csv_file << size << "," << avg_lcuda_time << "," << avg_thrust_time << std::endl;
        std::cout << "Vector Size: " << size << ", lcuda Avg Time: " << avg_lcuda_time << " ms, Thrust Avg Time: " << avg_thrust_time << " ms" << std::endl;
    }
    relu_csv_file.close();
    std::cout << "\nBenchmark results written to ../sheet/real_examples/relu_sum_bench.csv\n" << std::endl;

    return 0;
}