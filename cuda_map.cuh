#include <iostream>
#include <vector>
#include <list>
#include <array>
#include <map>
#include <deque>
#include <iterator> 
#include <cuda_runtime.h>
#include <type_traits>

#include <vector>

#include "cuda_map.h"

/*
Flags necessarias e como correro programa:

nvcc --extended-lambda -o cuda_map cuda.cu
.\cuda_map.exe    
*/  


template <typename T, typename Func>
__global__ void mapKernel(T* d_array, int size, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Verifica se o idx n e out of bounds 
    //TODO -> optimização de lançar outro kernel se o idx for maior que o tamanho em vez de ter o if statement
    if (idx < size) {
        d_array[idx] = func(d_array[idx]);
    }
}

template <typename T, typename Func>
struct FunctionWrapper {
    Func func;

    __device__ FunctionWrapper(Func f) : func(f) {}

    __device__ T operator()(T x) const {
        return func(x);
    }
};

template <typename Iterator, typename Func>
void map(Iterator& container, Func& func) {
    /*
    typename -> Necessario usar quando estamos a tratar de classes template
    std::iterator_traits<Iterator> -> extrair as traits do iterador
    ::value_type -> Tipo dos elementos do iterador
    . é pra aceder a membros de um objeto
    :: é pra aceder membros de uma classe ou namespace
    
    using T podia ser so:
    typedef typename std::iterator_traits<Iterator>::value_type T;
    typedef tipo Nome
    tmb podia-se usar template<typename T> antes

    Mas pra templates deve haver menos problemas a usar a keyword using
    */

    //using T = typename std::iterator_traits<Iterator>::value_type; //saber o tipo do iterador
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
    //__device__ auto device_func = [=] __device__ (T x) { return func(x); };
    FunctionWrapper<T, Func> device_func{func};
    mapKernel<<<numBlocks, blockSize>>>(d_array, size, device_func);

    
    cudaMemcpy(temp.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);

    
    std::copy(temp.begin(), temp.end(), container.begin());
}


/*
Podia ter sido feito com 
if constexpr (std::is_same_v<Container, std::vector<T>>){
}
Que o constexpr compila só se a condição for verdadeira o copy
*/
//Template specialization para vector
/* template <typename T, typename Func>
void map(std::vector<T>& container, Func& func) {

    size_t size = container.size(); 
    T* d_array;
    size_t bytes = size * sizeof(T);
    
    
    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, container.data(), bytes, cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    auto device_func = [=] __global__ __device__ (T x) { return func(x); };

    mapKernel<<<numBlocks, blockSize>>>(d_array, size, device_func);

    
    cudaMemcpy(container.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}
 */