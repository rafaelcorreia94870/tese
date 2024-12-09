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

    //Verifica se o idx n e out of bounds 
    //TODO -> optimização de lançar outro kernel se o idx for maior que o tamanho em vez de ter o if statement
    if (idx < size) {
        //func(d_array + idx);
        d_array[idx] = func(d_array[idx]);
    }
}

template <typename T, typename Func>
struct FunctionWrapper {
    Func func;

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
    //FunctionWrapper<T, Func> device_func{func};
    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func);

    
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
template <typename T, typename Func>
void map(std::vector<T>& container, Func& func) {

    size_t size = container.size(); 
    T* d_array;
    size_t bytes = size * sizeof(T);
    
    
    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, container.data(), bytes, cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    //auto device_func = [=] __global__ __device__ (T x) { return func(x); };

    mapKernel<<<numBlocks, blockSize>>>(d_array, size, func);

    
    cudaMemcpy(container.data(), d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

///////////////// FUNCOES PARA O MAP /////////////////
float square(float x) {
    return x * x;
}

int increment(int x) {
    return x + 1;
}

struct ThisIsAStruct {
    int x;
    int y;
};

ThisIsAStruct incrementStruct(ThisIsAStruct st, int inc) {
    st.x += inc;
    st.y += inc;
    return st;
}
/////////////////////////////////////////////////////
/////////// Functors for device functions ///////////
struct Square {
    __device__ float operator()(float x) const { return x * x; }
};

struct Increment {
    __device__ int operator()(int x) const { return x + 1; }
};


struct Transform_String {
    __device__ std::pair<const std::string, std::string> operator()(std::pair<const std::string, std::string> p) const {
        if (!p.second.compare( "Grass")) {
            p.second = "0";
        } else if (!p.second.compare( "Cement")) {
            p.second = "1";
        } else {
            p.second = "2";
        }
        return p;
    }
};
/////////////////////////////////////////////////////



int main() {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> intvec = {0, 1, 2, 3};
    std::list<int> intlist = {0, 1, 2, 3};
    std::array<int, 4> intarray = {0, 1, 2, 3};
    std::vector<ThisIsAStruct> myStruct = {{1, 2}, {3, 4}, {5, 6}};

    //KEY, VALUE collection//
    std::map<std::string, std::string>New_Map;
    New_Map["Ground"] = "Grass";
    New_Map["Floor"] = "Cement";
    New_Map["Table"] = "Wood";
    /////////////////////////


    //NON CONTIGUOUS COLLECTION//
    std::deque<int> myDeque;

    myDeque.push_back(10);
    myDeque.push_back(20);
    myDeque.push_back(30);
    myDeque.push_back(40);
    myDeque.push_front(5);
    /////////////////////////////



    //Functors
    map(vec, Square());
    map(intvec, Increment()); 
    map(intlist, Increment()); 
    map(intarray, Increment()); 
    //map(New_Map, Transform_String());
    map(myDeque, Increment());

   

    //Lambdas
    //map(vec.begin(),vec.end(), [] __device__ (float x) { return square(x); });
    //map(intvec.begin(),intvec.end(), [] __device__ (int x) { return increment(x); });
    //map(intlist.begin(),intlist.end(), [] __device__ (int x) { return increment(x); });
    //map(intarray.begin(),intarray.end(), [] __device__ (int x) { return increment(x); });

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

    /*
    std::cout << "\nMap with strings \n" ;
    for(auto x: New_Map)
    {
        std::cout << x.first << "->" << 
        x.second <<std::endl;
    }
    */

    std::cout << "\nDeque\n";
    for (auto it = myDeque.begin(); it != myDeque.end(); ++it) {
        std::cout << *it << " "; 
    }

    std::cout << "\nStruct Vector\n";
    for (auto s : myStruct) {
        std::cout << s.x << " " << s.y << " ;";
    }

    std::cout << std::endl;

    return 0;
}
