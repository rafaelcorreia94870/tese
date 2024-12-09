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

template <typename T, typename Func>
__global__ void mapKernel(T* d_array, int size, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        func(d_array + idx);
    }
}

template <typename Iterator, typename Func>
void map(Iterator& container, Func& func) {
    using T = typename Iterator::value_type;
    std::vector<T> temp;  
    
    for (auto it = container.begin(); it != container.end(); ++it) {
        temp.push_back(*it);
    }

    size_t size = temp.size(); 
    T* d_array;
    size_t bytes = size * sizeof(T);
    
    
    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, temp.data(), bytes, cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func);

    
    cudaMemcpy(temp.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);

    
    std::copy(temp.begin(), temp.end(), container.begin());
}



template <typename T, typename Func>
void map(std::vector<T>& container, Func& func) {

    size_t size = container.size(); 
    T* d_array;
    size_t bytes = size * sizeof(T);
    
    
    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, container.data(), bytes, cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func);

    
    cudaMemcpy(container.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}



/////////////////////////////////////////////////////
/////////// Functors for device functions ///////////
struct Square {
    __device__ void operator()(float* x) const {
        *x = (*x) * (*x);
    }
};

struct Increment {
    __device__ void operator()(int* x) const {
        *x = *x + 1;
    }
};
/////////////////////////////////////////////////////



int main() {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> intvec = {0, 1, 2, 3};
    std::list<int> intlist = {0, 1, 2, 3};
    std::array<int, 4> intarray = {0, 1, 2, 3};
 


    //Functors
    map(vec, Square());
    map(intvec, Increment()); 
    map(intlist, Increment()); 
    map(intarray, Increment()); 

    std::cout << "Float vec \n" ;
    for (float v : vec) {
        std::cout << v << " ";
    }
    std::cout << "\nInt vec \n" ;
    for(int v : intvec){
        std::cout << v << " ";
    }
    std::cout << "\nInt List \n" ;
    for(int v : intlist){
        std::cout << v << " ";
    }
    
    std::cout << "\nInt Array \n" ;
    for(int v : intarray){
        std::cout << v << " ";
    }

    return 0;
}
