#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include "fused_gpu_vec.cuh"  
#include "experimental_fusion.cuh"  
#include "kernel_op.cuh"

#include "../../experimental/includes/framework/rafa.cuh"

std::chrono::duration<double> twointensivecomputations_old(size_t N, size_t loop_count) {
    auto start = std::chrono::high_resolution_clock::now();
    rafa::vector<float> vec(N, 1.0f);
    rafa::vector<float> result(N);

    vec.smart_map(BenchmarkingComputations(loop_count)).smart_map(BenchmarkingComputations(loop_count), result).execute();
    
    auto end = std::chrono::high_resolution_clock::now();
    vec.clear();
    result.clear();

    //result.print();
    cudaDeviceReset();


    //std::cout << result[0];
    return end - start;

}

std::chrono::duration<double> singlecomputation_old(size_t N){
    auto start = std::chrono::high_resolution_clock::now();
    rafa::vector<float> vec(N, 1.0f);
    rafa::vector<float> result(N);

    vec.smart_map(SimpleComputation()).execute();
    
    auto end = std::chrono::high_resolution_clock::now();
    vec.clear();
    result.clear();

    cudaDeviceReset();

    return end - start;

}

std::chrono::duration<double> tensimplecomputations_old(size_t N) {
    auto start = std::chrono::high_resolution_clock::now();
    rafa::vector<float> vec(N, 1.0f);
    rafa::vector<float> result(N);

    vec.smart_map(SimpleComputation()).smart_map(SimpleComputation()).smart_map(SimpleComputation()).smart_map(SimpleComputation()).smart_map(SimpleComputation()).smart_map(SimpleComputation()).smart_map(SimpleComputation()).smart_map(SimpleComputation()).smart_map(SimpleComputation()).smart_map(SimpleComputation()).execute();
    
    auto end = std::chrono::high_resolution_clock::now();
    vec.clear();
    result.clear();

    cudaDeviceReset();

    return end - start;
}

void two_intenvisve_benchmark(size_t N, size_t loop_count_1, size_t loop_count_2, size_t loop_count_3, size_t it) {
    std::cout << "Iteration,Loop_Count,Expression Time (ns),GPU Vector Time (ns),Old Implementation Time (ns)\n";
    for (int i = 0; i < it; i++) {
        auto expr = twointensivecomputations_expr(N, loop_count_1);
        auto gpu_vec = twointensivecomputations_gpu_vec(N, loop_count_1);
        auto old_impl = twointensivecomputations_old(N, loop_count_1);

        std::cout << i + 1 << "," << loop_count_1 << ","
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(expr).count() << ","
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_vec).count() << ","
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(old_impl).count() << "\n";


        auto expr2 = twointensivecomputations_expr(N, loop_count_2);
        auto gpu_vec2 = twointensivecomputations_gpu_vec(N, loop_count_2);
        auto old_impl2 = twointensivecomputations_old(N, loop_count_2);
        std::cout << i + 1 << "," << loop_count_2 << ","
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(expr2).count() << ","
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_vec2).count() << ","
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(old_impl2).count() << "\n";
        auto expr3 = twointensivecomputations_expr(N, loop_count_3);
        auto gpu_vec3 = twointensivecomputations_gpu_vec(N, loop_count_3);
        auto old_impl3 = twointensivecomputations_old(N, loop_count_3);

        std::cout << i + 1 << "," << loop_count_3 << ","
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(expr3).count() << ","
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_vec3).count() << ","
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(old_impl3).count() << "\n";
    }
}

void ten_simple_computations_benchmark(size_t MIN_N, size_t MAX_N, size_t it) {

    std::cout << "Iteration,N,Expression Time (ns),GPU Vector Time (ns),Old Implementation Time (ns)" << "\n";
    for (size_t N = MIN_N; N <= MAX_N; N *= 2) {
        for(int i = 0; i < it; i++) {
            auto expr = tensimplecomputations_expr(N);
            auto gpu_vec = tensimplecomputations_gpu_vec(N);
            auto old_impl = tensimplecomputations_old(N);

            std::cout << i + 1 << "," << N << ","
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(expr).count() << ","
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_vec).count() << ","
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(old_impl).count() << "\n";
        }
    }
}  

void single_computation_benchmark(size_t MIN_N, size_t MAX_N, size_t it) {
    std::cout << "Iteration,N,Expression Time (ns),GPU Vector Time (ns),Old Implementation Time (ns)" << "\n";
    for (size_t N = MIN_N; N <= MAX_N; N *= 2) {
        
        for (size_t i = 0; i < it; i++) {
            auto expr = singlecomputation_expr(N);
            auto gpu_vec = singlecomputation_gpu_vec(N);
            auto old_impl = singlecomputation_old(N);

            std::cout << i + 1 << "," << N << ","
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(expr).count() << ","
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_vec).count() << ","
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(old_impl).count() << "\n";
        }
    }
}
    

int main() {
    const size_t N = 50'000'000;
    const size_t loop_count_1 = 10;
    const size_t loop_count_2 = 100;
    const size_t loop_count_3 = 1000;

    const size_t it = 20;
    
    auto warmup_expr = twointensivecomputations_expr(N, 1);
    auto warmup_gpu_vec = twointensivecomputations_gpu_vec(N, 1);
    auto warmup_old = twointensivecomputations_old(N, 1);

    

    std::cout << "Warmup completed :" << warmup_expr.count() << "s, "
              << warmup_gpu_vec.count() << "s, "
              << warmup_old.count() << "s" << std::endl;

              
              
    std::cout << "Starting benchmarking two intensive computations..." << std::endl;
    std::ofstream outFile("../sheet/fusion_data_optimized_O2.csv");
    auto cout_buf = std::cout.rdbuf();
    std::cout.rdbuf(outFile.rdbuf());
    two_intenvisve_benchmark(N, loop_count_1, loop_count_2, loop_count_3, it);

    std::cout.rdbuf(cout_buf);
    std::cout << "Benchmarking two intensive computations completed." << std::endl;

    std::ofstream outFile2("..\\sheet\\fusion_data_optimized_O2_2.csv");
    std::cout.rdbuf(outFile2.rdbuf());
    ten_simple_computations_benchmark(10'000, 50'000'000, it);

    std::cout.rdbuf(cout_buf);
    std::cout << "Benchmarking completed." << std::endl;

    std::cout << "Starting single computation benchmark...\n";
    std::ofstream outFile3("../sheet/single_computation_benchmark_O2.csv");
    auto cout_buf3 = std::cout.rdbuf();
    std::cout.rdbuf(outFile3.rdbuf());
    single_computation_benchmark(10'000, 50'000'000, it);
    std::cout.rdbuf(cout_buf3);
    std::cout << "Single computation benchmark completed.\n";



    /* two_intenvisve_benchmark(N, loop_count_1, loop_count_2, loop_count_3, it);
    ten_simple_computations_benchmark(50'000'000, 50'000'000, it); */
                


}