#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <list>
#include <cuda_runtime.h>


template <typename T>
concept VectorLike = requires(T a, size_t i) {
    { a.size() } -> std::convertible_to<size_t>; 
    { a.begin() };  
    { a.end() };    
    { a.data() };
    { a[i] }; 
} && std::ranges::range<T>; 



template <typename T, typename Func, typename... Args>
__global__ void mapKernel(T* d_array, int size, Func func, Args... args) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = func(d_array[idx], args...);
    }
}


struct IntensiveComputation {
    __device__ float operator()(float x) const {
        for (int i = 0; i < 100; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
        }
        return x;
    }
};

struct IntensiveComputationParams {
    __device__ float operator()(float x, int a, double b, bool flag) const {
        for (int i = 0; i < 100; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
        }
        return flag ? (x * a + b) : (x / a - b);
    }
};

float CpuIntensiveComputation(float x) {
    for (int i = 0; i < 100; ++i) { 
        x = sin(x) * cos(x) + log(x + 1.0f);
    }
    return x;
}

float CpuIntensiveComputationParams(float x) {
    int a=5;
    double b=2.3; 
    bool flag = true;
    for (int i = 0; i < 100; ++i) { 
        x = sin(x) * cos(x) + log(x + 1.0f);
    }
    return flag ? (x * a + b) : (x / a - b);
}



template <VectorLike Container, typename Func, typename... Args>
void map_impl(Container& container, Func func, Args... args) {
    using T = typename Container::value_type;
    size_t size = container.size();
    T* d_array;
    size_t bytes = size * sizeof(T);

    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, container.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func, args...);
    cudaDeviceSynchronize();
    cudaMemcpy(container.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}


template <VectorLike Container, typename Func, typename... Args>
void map_impl(Container& input, Func func, Container& output, Args... args) {
    using T = typename Container::value_type;
    size_t size = input.size();
    T* d_array;
    size_t bytes = size * sizeof(T);

    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, input.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func, args...);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

template <VectorLike Container, typename Func, typename... Args>
void map(Container& container, Func func, Args... args) {
    map_impl(container, func, args...);
}

template <VectorLike Container, typename Func, typename... Args>
void map(Container& container, Func func, Container& output, Args... args) {
    map_impl(container, func, output, args...); 
}



template <typename Container, typename Func>
void cpuMap(Container& container, Func func) {
    for (auto& elem : container) {
        elem = func(elem);
    }
}

template <typename Container1, typename Container2>
void compareAndPrint(const std::string& name1, const Container1& container1,
                     const std::string& name2, const Container2& container2,
                     const std::string& operationName, double duration1, double duration2,
                     double tolerance = 1e-6) {

    if (container1.size() != container2.size()) {
        std::cout << "Error: Containers " << name1 << " and " << name2 << " have different sizes.\n";
        return;
    }

    bool resultsMatch = true;
    auto iter1 = container1.begin();
    auto iter2 = container2.begin();
    for (size_t i = 0; iter1 != container1.end() && iter2 != container2.end(); ++iter1, ++iter2, ++i) {
        if (std::abs(*iter1 - *iter2) > tolerance) {
            std::cout << name1 << "[" << i << "] = " << *iter1 << " "
                      << name2 << "[" << i << "] = " << *iter2 << "\n";
            resultsMatch = false;
            break;
        }
    }

    std::cout << operationName << " " << name1 << " Time: " << duration1 << " ms\n";
    std::cout << operationName << " " << name2 << " Time: " << duration2 << " ms\n";
    std::cout << operationName << " Results Match: " << (resultsMatch ? "Yes" : "No") << "\n\n";
}

int main() {
    const size_t N = 10000000;  

    //N elementos com 2.0f
    std::vector<float> cudaVec(N, 2.0f);
    std::vector<float> cudaVec2(N, 2.0f);
    std::vector<float> cpuVec(N, 2.0f);

    std::vector<float> cudaVec_out(N);



    auto startCuda = std::chrono::high_resolution_clock::now();
    map(cudaVec, IntensiveComputationParams(), 5, 2.3, true);
    auto endCuda = std::chrono::high_resolution_clock::now();
    auto cudaDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endCuda - startCuda);

    auto startCuda_out = std::chrono::high_resolution_clock::now();
    map(cudaVec2, IntensiveComputationParams(), cudaVec_out,5, 2.3, true);
    auto endCuda_out = std::chrono::high_resolution_clock::now();
    auto cudaDuration_out = std::chrono::duration_cast<std::chrono::milliseconds>(endCuda_out - startCuda_out);

    
    auto startCpu = std::chrono::high_resolution_clock::now();
    std::transform(cpuVec.begin(),cpuVec.end(),cpuVec.begin(), CpuIntensiveComputationParams);
    auto endCpu = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endCpu - startCpu);

    // Compare cudaVec and cpuVec
    compareAndPrint("cudaVec", cudaVec, "cpuVec", cpuVec, "Map", cudaDuration.count(), cpuDuration.count());

    // Compare cudaVec_out and cpuVec
    compareAndPrint("cudaVecList", cudaVec_out, "cpuVec", cpuVec, "Map Output", cudaDuration_out.count(), cpuDuration.count());


    return 0;
}
