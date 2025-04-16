#include "includes/framework/rafa.cuh"
#include <iostream>

// Device function to double the input value
struct DoubleIt {
    __device__ int operator()(int x) const {
        return x * 2;
    }
};

// Device function to add two input values
struct Add {
    __device__ int operator()(int x, int y) const {
        return x + y;
    }
};


int main() {

    std::cout << "Test 1: Basic vector initialization and data synchronization\n";
    rafa::vector<int> vec(5);
    for (int i = 0; i < 5; ++i) {
        vec[i] = i;
    }

    vec.sync_host_to_device();

    std::cout << "Original vector: ";
    vec.print();

    std::cout << "Vector after sync: ";
    vec.sync_device_to_host();
    vec.print(); 

    //////////////////////////////////////////////////////////////////////////
    //cudaDeviceSynchronize(); 
    std::cout << "\n\n\nTest 2: Map operation with a single input vector\n";
    rafa::vector<int> result(5,1);
    vec.map(DoubleIt(), result);
    //rafa::skeletons::map(vec, double_it, result);
    result.sync_device_to_host();

    std::cout << "Vector after map (double_it): ";
    result.print();
    std::cout << "Espected result: [0, 2, 4, 6, 8]\n\n\n";

    //////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize(); 
    std::cout << "Test 3: Map operation with two input vectors\n\n\n";
    rafa::vector<int> vec2(5);
    for (int i = 0; i < 5; ++i) {
        vec2[i] = i + 5;
    }
    vec2.sync_host_to_device();

    rafa::vector<int> result2(5,0);
    vec.print();
    vec2.print();
    rafa::skeletons::map(vec, vec2, Add(), result2);
    result2.sync_device_to_host();

    std::cout << "Vector after map (add): ";
    result2.print();
    std::cout << "Espected result: [5, 7, 9, 11, 13]\n\n\n";

    //////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize(); 
    vec.print();
    std::cout << "Test 4: Reduce operation\n\n\n";
    int reduce_result = vec.reduce(0, Add());
    std::cout << "Reduce result (sum): " << reduce_result << std::endl;
    std::cout << "Espected result: 10\n\n\n";

    //////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize(); 
    std::cout << "Test 5: In-place map operation\n\n\n";
    vec.map(DoubleIt());
    vec.sync_device_to_host();

    std::cout << "Vector after in-place map (double_it): ";
    vec.print(); 
    std::cout << "Espected result: [0, 2, 4, 6, 8]\n\n\n";
 
    return 0;
}
