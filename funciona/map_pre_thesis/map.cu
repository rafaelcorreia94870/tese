#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <list>
#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>



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

template <typename T, typename Func, typename... Args>
__global__ void mapKernel2inputs(T* d_array, T* d_array2, int size, Func func, Args... args) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = func(d_array[idx], d_array2[idx], args...);
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
    __device__ float operator()(float x, int a=5, double b=2.3, bool flag=true) const {
        for (int i = 0; i < 100; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
        }
        return flag ? (x * a + b) : (x / a - b);
    }
};

struct IntensiveComputation2Inputs {
    __device__ float operator()(float x, float y, int a=5, double b=2.3, bool flag=true) const {
        for (int i = 0; i < 100; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
            y = sin(y) * cos(y) + log(y + 1.0f);
        }
        return flag ? (x * y * a + b) : ((x + y) / a - b);
    }
};



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
void map_impl(Container& input1, Container& input2, Func func, Args... args) {
    using T = typename Container::value_type;
    size_t size = input1.size();
    T* d_array;
    T* d_array2;
    size_t bytes = size * sizeof(T);

    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, input1.data(), bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_array2, bytes);
    cudaMemcpy(d_array2, input2.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel2inputs<<<numBlocks, blockSize>>>(d_array, d_array2, size, func, args...);

    cudaDeviceSynchronize();
    cudaMemcpy(input1.data(), d_array, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_array2);

}

template <VectorLike Container, typename Func, typename... Args>
void map_impl(Container& input1, Container& input2, Func func, Container& output, Args... args) {
    using T = typename Container::value_type;
    size_t size = input1.size();
    T* d_array;
    T* d_array2;
    size_t bytes = size * sizeof(T);

    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, input1.data(), bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_array2, bytes);
    cudaMemcpy(d_array2, input2.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel2inputs<<<numBlocks, blockSize>>>(d_array,d_array2, size, func, args...);
    
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_array, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_array2);

}

template <VectorLike Container, typename Func, typename... Args>
void map(Container& container, Func func, Args... args) {
    map_impl(container, func, args...);
}

template <VectorLike Container, typename Func, typename... Args>
void map(Container& container, Func func, Container& output, Args... args) {
    map_impl(container, func, output, args...); 
}

template <VectorLike Container, typename Func, typename... Args>
void map(Container& container1, Container& container2, Func func, Args... args) {
    map_impl(container1,container2, func, args...);
}

template <VectorLike Container, typename Func, typename... Args>
void map(Container& container1,Container& container2, Func func, Container& output, Args... args) {
    map_impl(container1,container2, func, output, args...); 
}


template <typename Container, typename Func, typename... Args>
void cpuMap(Container& container, Func func, Args... args) {
    for (auto& elem : container) {
        elem = func(elem, args...);
    }
}

template <typename Container1, typename Container2>
void compareAndPrint(const std::string& name1, const Container1& container1,
                     const std::string& name2, const Container2& container2,
                     const std::string& operationName, float duration1, float duration2,
                     float tolerance = 1e-6) {

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

template <typename Func>
auto timeFunction(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}


int main() {
    const size_t N = 100000000;

    /////////////////////////////////////////////////////////////////////////////////////////
    // CUDA map without parameters (single input, modifies in-place)

    std::vector<float> cuda_1in_inplace(N, 2.0f);

    auto cuda_time_1in_inplace = timeFunction([&]() {
        map(cuda_1in_inplace, IntensiveComputation() );
    });
    
    // Thrust map equivalent for single input, modifying in-place

    std::vector<float> thrust_1in_inplace(N, 2.0f);

    auto thrust_time_1in_inplace = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrust_1in_inplace;
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                          IntensiveComputation());
        thrust::copy(d_vec.begin(), d_vec.end(), thrust_1in_inplace.begin());
    });

    compareAndPrint("cuda_1in_inplace", cuda_1in_inplace, "thrust_1in_inplace", thrust_1in_inplace, "Map (1 Input - In-Place)", cuda_time_1in_inplace.count(), thrust_time_1in_inplace.count());
    
    /////////////////////////////////////////////////////////////////////////////////////////
    // CUDA map with output and params

    std::vector<float> cudaInput_1in_output(N, 2.0f);
    std::vector<float> cudaOutput_1in_output(N);

    auto cuda_time_1in_output = timeFunction([&]() {
        map(cudaInput_1in_output, IntensiveComputationParams(), cudaOutput_1in_output, 5, 2.3, true);
    });
    
    // Thrust map equivalent for single input, storing in output vector with parameters

    std::vector<float> thrustIn_1in_output(N, 2.0f);
    std::vector<float> thrust_1in_output(N, 2.0f);

    auto thrust_time_1in_output = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrustIn_1in_output;
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                          IntensiveComputationParams());
        thrust::copy(d_vec.begin(), d_vec.end(), thrust_1in_output.begin());
    });

    compareAndPrint("cudaOutput_1in_output", cudaOutput_1in_output, "thrust_1in_output", thrust_1in_output, "Map (1 Input - Output with params)", cuda_time_1in_output.count(), thrust_time_1in_output.count());

    
    /////////////////////////////////////////////////////////////////////////////////////////
    // CUDA map with ouput 

    std::vector<float> cudaInput_1in_output2(N, 2.0f);
    std::vector<float> cudaOutput_1in_output2(N);

    auto cuda_time_1in_output2 = timeFunction([&]() {
        map(cudaInput_1in_output2, IntensiveComputation(), cudaOutput_1in_output2);
    });

    // Thrust map equivalent for single input, storing in output vector

    std::vector<float> thrustIn_1in_output2(N, 2.0f);
    std::vector<float> thrust_1in_output2(N, 2.0f);

    auto thrust_time_1in_output2 = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrustIn_1in_output2;
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                          IntensiveComputation());
        thrust::copy(d_vec.begin(), d_vec.end(), thrust_1in_output2.begin());
    });

    compareAndPrint("cudaOutput_1in_output2", cudaOutput_1in_output2, "thrust_1in_output2", thrust_1in_output2, "Map (1 Input - Output)", cuda_time_1in_output2.count(), thrust_time_1in_output2.count());
    

    /////////////////////////////////////////////////////////////////////////////////////////
    // CUDA map with two inputs modifying first input
    
    std::vector<float> cudaInput_2in_inplace1(N, 2.0f);
    std::vector<float> cudaInput_2in_inplace2(N, 2.0f);

    auto cuda_time_2in_inplace = timeFunction([&]() {
        map(cudaInput_2in_inplace1, cudaInput_2in_inplace2, IntensiveComputation2Inputs(), 5, 2.3, true);
    });

    // Thrust equivalent for two-inputs modifying first input

    std::vector<float> thrustIn1_2in_inplace(N, 2.0f);
    std::vector<float> thrustIn2_2in_inplace(N, 2.0f);

    auto thrust_time_2in_inplace = timeFunction([&]() {
        thrust::device_vector<float> d_vec1 = thrustIn1_2in_inplace;
        thrust::device_vector<float> d_vec2 = thrustIn2_2in_inplace;
        thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_vec1.begin(),
                          IntensiveComputation2Inputs());
        thrust::copy(d_vec1.begin(), d_vec1.end(), thrustIn1_2in_inplace.begin());
    });

    compareAndPrint("cudaInput_2in_inplace1", cudaInput_2in_inplace1, "thrustIn1_2in_inplace", thrustIn1_2in_inplace, "Map (2 Inputs - In-Place)", cuda_time_2in_inplace.count(), thrust_time_2in_inplace.count());

    /////////////////////////////////////////////////////////////////////////////////////////

    // CUDA map with two inputs storing in output vector

    std::vector<float> cudaInput_2in_output1(N, 2.0f);
    std::vector<float> cudaInput_2in_output2(N, 2.0f);
    std::vector<float> cudaOutput_2in_output(N);

    auto cuda_time_2in_output = timeFunction([&]() {
        map(cudaInput_2in_output1, cudaInput_2in_output2, IntensiveComputation2Inputs(), cudaOutput_2in_output, 5, 2.3, true);
    });

    // Thrust equivalent for two-inputs storing in output vector

    std::vector<float> thrustIn1_2in_output(N, 2.0f);
    std::vector<float> thrustIn2_2in_output(N, 2.0f);
    std::vector<float> thrustOut_2in_output(N);

    auto thrust_time_2in_output = timeFunction([&]() {
        thrust::device_vector<float> d_vec1 = thrustIn1_2in_output;
        thrust::device_vector<float> d_vec2 = thrustIn2_2in_output;
        thrust::device_vector<float> d_out(thrustOut_2in_output.size());
        
        thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_out.begin(),
                          IntensiveComputation2Inputs());
        thrust::copy(d_out.begin(), d_out.end(), thrustOut_2in_output.begin());
    });

    compareAndPrint("cudaOutput_2in_output", cudaOutput_2in_output, "thrustOut_2in_output", thrustOut_2in_output, "Map (2 Inputs - Output)", cuda_time_2in_output.count(), thrust_time_2in_output.count());

    /////////////////////////////////////////////////////////////////////////////////////////
    // CUDA map with parameters (two inputs, modifies in-place) with parameters

    std::vector<float> cudaInput_2in_inplace_params1(N, 2.0f);
    std::vector<float> cudaInput_2in_inplace_params2(N, 2.0f);

    auto cuda_time_2in_inplace_params = timeFunction([&]() {
        map(cudaInput_2in_inplace_params1, cudaInput_2in_inplace_params2, IntensiveComputation2Inputs(), 5, 2.3, true);
    });

    // Thrust equivalent for two-inputs modifying in-place with parameters

    std::vector<float> thrustIn1_2in_inplace_params(N, 2.0f);
    std::vector<float> thrustIn2_2in_inplace_params(N, 2.0f);

    auto thrust_time_2in_inplace_params = timeFunction([&]() {
        thrust::device_vector<float> d_vec1 = thrustIn1_2in_inplace_params;
        thrust::device_vector<float> d_vec2 = thrustIn2_2in_inplace_params;
        thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_vec1.begin(),
                          IntensiveComputation2Inputs());
        thrust::copy(d_vec1.begin(), d_vec1.end(), thrustIn1_2in_inplace_params.begin());
    });

    compareAndPrint("cudaInput_2in_inplace_params1", cudaInput_2in_inplace_params1, "thrustIn1_2in_inplace_params", thrustIn1_2in_inplace_params, "Map (2 Inputs - In-Place with Params)", cuda_time_2in_inplace_params.count(), thrust_time_2in_inplace_params.count());

    return 0;
}
