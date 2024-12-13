#include <iostream>
#include <vector>
#include <list>
#include <array>
#include <map>
#include <deque>
#include <iterator>
#include <cuda_runtime.h>
#include <type_traits>

/*
Flags necessarias e como correro programa:

nvcc --extended-lambda -o cuda_map cuda.cu
.\cuda_map.exe    
*/

#pragma hd_warning_disable
template<typename T,class Function>
__host__ __device__
T invoke(Function f, T x)
{
  return f(x);
}


// Device kernel with callable function
template <typename T, typename Function>
__global__ void mapKernel(T* d_array, int size, Function func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = invoke(func,(d_array[idx]));
    }
}

// Specialization for std::vector
template <typename T, typename Function>
void map(std::vector<T>& container, Function func) {
    size_t size = container.size();
    T* d_array;
    size_t bytes = size * sizeof(T);

    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, container.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Use the wrapper to call the device function
    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func);

    cudaMemcpy(container.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

/////////////////////////////////////////////////////
//////////// Example Device Functions ///////////////

    float square(float x) {
        return x * x;
    }

/////////////////////////////////////////////////////

int main() {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};

    // Pass device functions as pointers
    map(vec, square);


    std::cout << "Float vec \n";
    for (float v : vec) {
        std::cout << v << " ";
    }

    return 0;
}
