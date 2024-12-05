#include <iostream>
#include <vector>
#include "cuda_map.h"

struct Square {
    __device__ float operator()(float x) const { return x * x; }
};

struct Increment {
    __device__ int operator()(int x) const { return x + 1; }
};

int main() {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> intvec = {0, 1, 2, 3};

    Square squareFunctor;
    Increment incrementFunctor;

    map(vec, squareFunctor);
    map(intvec, incrementFunctor);

    std::cout << "Float vec:\n";
    for (float v : vec) {
        std::cout << v << " ";
    }
    std::cout << "\nInt vec:\n";
    for (int v : intvec) {
        std::cout << v << " ";
    }

    std::cout << std::endl;
    return 0;
}
