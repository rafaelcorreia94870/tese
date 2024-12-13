#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>


template <typename T, typename Func>
__global__ void mapKernel(T* d_array, int size, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = func(d_array[idx]);
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



    float CpuIntensiveComputation(float x) {
        for (int i = 0; i < 100; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
        }
        return x;
    }



template <typename T, typename Func>
void map(std::vector<T>& container, Func func) {
    size_t size = container.size();
    T* d_array;
    size_t bytes = size * sizeof(T);

    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, container.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func);
    cudaDeviceSynchronize();

    cudaMemcpy(container.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}


template <typename Container, typename Func>
void cpuMap(Container& container, Func func) {
    for (auto& elem : container) {
        elem = func(elem);
    }
}

int main() {
    const size_t N = 10000000;  

    //N elementos com 2.0f
    std::vector<float> cudaVec(N, 2.0f);
    std::vector<float> cpuVec(N, 2.0f);

    
    auto startCuda = std::chrono::high_resolution_clock::now();
    map(cudaVec, IntensiveComputation());
    auto endCuda = std::chrono::high_resolution_clock::now();
    auto cudaDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endCuda - startCuda);

    
    auto startCpu = std::chrono::high_resolution_clock::now();
    std::transform(cpuVec.begin(),cpuVec.end(),cpuVec.begin(), CpuIntensiveComputation);
    auto endCpu = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endCpu - startCpu);

    
    bool resultsMatch = true;
    for (size_t i = 0; i < N; ++i) {
        //std::cout << "cudaVec[" << i << "] = " << cudaVec[i] << " cpuVec[" << i << "] = " << cpuVec[i] << "\n";
        if (std::abs(cudaVec[i] - cpuVec[i]) > 1e-6) {
            resultsMatch = false;
            break;
        }
    }

    
    std::cout << "CUDA Map Time: " << cudaDuration.count() << " ms\n";
    std::cout << "CPU Map Time: " << cpuDuration.count() << " ms\n";
    std::cout << "Results Match: " << (resultsMatch ? "Yes" : "No") << "\n";

    return 0;
}
