#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <list>
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




template <typename Iterator, typename Func>
void map_impl(Iterator& container, Func& func) {
    using T = typename Iterator::value_type;
    size_t size = container.size();
    T* d_array;
    size_t bytes = size * sizeof(T);
    
    cudaMalloc(&d_array, bytes);
    cudaError_t err = cudaMemcpy(d_array, container.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to device failed: " << cudaGetErrorString(err) << std::endl;
    }

    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func);
    cudaDeviceSynchronize();
    cudaError_t err2 = cudaMemcpy(container.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    if (err2 != cudaSuccess) {
        std::cerr << "CUDA memcpy to host failed: " << cudaGetErrorString(err2) << std::endl;
    }
    cudaFree(d_array);
    
}

template <typename Iterator, typename Func>
void map_impl(Iterator& container, Func& func, Iterator& output) {
    using T = typename Iterator::value_type;
    size_t size = container.size();
    T* d_array;
    size_t bytes = size * sizeof(T);
    
    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, container.data(), bytes, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func);
    cudaDeviceSynchronize();
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(kernelErr) << std::endl;
    }
    cudaMemcpy(output.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    
}




template <typename Iterator, typename Func>
void map_clean(Iterator& container, Func& func) {
    //std::cout << "map_clean\n";
    //std::cout << "container size: " << container.size() << "\n";
    //std::cout << "container type: " << typeid(container).name() << "\n";

    using T = typename Iterator::value_type;
    std::vector<T> temp; 
    if constexpr (!std::is_same_v<Iterator, std::vector<T>>){

        for (auto it = container.begin(); it != container.end(); ++it) {
            temp.push_back(*it);
        }
        map_impl(temp, func);
        std::copy(temp.begin(), temp.end(), container.begin());
    }
    else{
        map_impl(container, func);
    }
}

template <typename Func, typename Tuple, std::size_t Index = 0>
void apply_map_clean_to_tuple(Func& func, Tuple& argsTuple) {
    if constexpr (Index < std::tuple_size_v<Tuple> - 1) { 
        auto& container = std::get<Index>(argsTuple);     
        map_clean(container, func);                       
        apply_map_clean_to_tuple<Func, Tuple, Index + 1>(func, argsTuple); 
    }
}

//vi && no raptor -> estudar melhor r-value reference
template <typename ...Args>
void map(Args &&...args) {
    if constexpr (sizeof...(args) == 2) {
        map_clean(args...);
    }
    else {
        
        auto argsTuple = std::forward_as_tuple(args...);

        constexpr size_t numArgs = sizeof...(args);
        auto& func = std::get<numArgs - 1>(argsTuple);

        apply_map_clean_to_tuple(func, argsTuple);
    }

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
    // Ensure containers have the same size
    if (container1.size() != container2.size()) {
        std::cout << "Error: Containers " << name1 << " and " << name2 << " have different sizes.\n";
        return;
    }

    // Compare elements
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

    // Print results
    std::cout << operationName << " " << name1 << " Time: " << duration1 << " ms\n";
    std::cout << operationName << " " << name2 << " Time: " << duration2 << " ms\n";
    std::cout << operationName << " Results Match: " << (resultsMatch ? "Yes" : "No") << "\n\n";
}

int main() {
    const size_t N = 10000000;  

    //N elementos com 2.0f
    std::vector<float> cudaVec_solo(N, 2.0f);

    std::vector<float> cudaVec(N, 2.0f);
    std::vector<float> cpuVec(N, 2.0f);
    std::list<float> cudaVecList(N, 2.0f);

    std::vector<float> cudaVec_out(N);
    std::list<float> cudaVecList_out(N);

/*
    auto startCuda_out = std::chrono::high_resolution_clock::now();
    map(cudaVec, IntensiveComputation(), cudaVec_out);
    auto endCuda_out = std::chrono::high_resolution_clock::now();
    auto cudaDuration_out = std::chrono::duration_cast<std::chrono::milliseconds>(endCuda_out - startCuda_out);

    auto startCudaList_out = std::chrono::high_resolution_clock::now();
    map(cudaVecList, IntensiveComputation(), cudaVecList_out);
    auto endCudaList_out = std::chrono::high_resolution_clock::now();
    auto cudaDurationList_out = std::chrono::duration_cast<std::chrono::milliseconds>(endCudaList_out - startCudaList_out);
*/
    auto startCuda_solo = std::chrono::high_resolution_clock::now();
    map(cudaVec_solo, IntensiveComputation());
    auto endCuda_solo = std::chrono::high_resolution_clock::now();
    auto cudaDuration_solo = std::chrono::duration_cast<std::chrono::milliseconds>(endCuda_solo - startCuda_solo);
    
    auto startCuda = std::chrono::high_resolution_clock::now();
    map(cudaVec ,cudaVecList, IntensiveComputation());
    auto endCuda = std::chrono::high_resolution_clock::now();
    auto cudaDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endCuda - startCuda);

/*
    auto startCudaList = std::chrono::high_resolution_clock::now();
    map(cudaVecList, IntensiveComputation());
    auto endCudaList = std::chrono::high_resolution_clock::now();
    auto cudaDurationList = std::chrono::duration_cast<std::chrono::milliseconds>(endCudaList - startCudaList);

*/
    
    auto startCpu = std::chrono::high_resolution_clock::now();
    std::transform(cpuVec.begin(),cpuVec.end(),cpuVec.begin(), CpuIntensiveComputation);
    auto endCpu = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endCpu - startCpu);

    // Compare cudaVec and cpuVec
    compareAndPrint("cudaVec", cudaVec, "cpuVec", cpuVec, "Map", cudaDuration.count(), cpuDuration.count());

    // Compare cudaVecList and cpuVec
    compareAndPrint("cudaVecList", cudaVecList, "cpuVec", cpuVec, "Map List", cudaDuration.count(), cpuDuration.count());

    compareAndPrint("cudaVecSolo", cudaVec_solo, "cpuVec", cpuVec, "Map", cudaDuration_solo.count(), cpuDuration.count());

/*
    // Compare cudaVecWithOutput and cpuVec
    compareAndPrint("cudaVecWithOutput", cudaVec, "cpuVec", cpuVec, "Map", cudaDuration_out.count(), cpuDuration.count());

    // Compare cudaVecListWithOutput and cpuVec
    compareAndPrint("cudaVecListWithOutput", cudaVecList, "cpuVec", cpuVec, "Map List", cudaDurationList_out.count(), cpuDuration.count());
*/

    return 0;
}
