#include "includes/framework/rafa.cuh"
#include <iostream>

// Device function to double the input value
struct DoubleIt {
    __device__ __host__ int operator()(int x) const {
        return x * 2;
    }
};

// Device function to add two input values
struct Add {
    __device__ int operator()(int x, int y) const {
        return x + y;
    }
};

struct AddAndShift {
    __device__
    int operator()(int x, int y, int z) const {
        return x + y + z;
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
    vec.smart_map(DoubleIt(), result).execute(); ;
    //rafa::map(vec, double_it, result);
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

    rafa::vector<int> result2(5,0);
    vec.smart_map(vec2, Add(), result2).execute();


    std::cout << "Vector after map (add): ";
    result2.print();
    std::cout << "Espected result: [5, 7, 9, 11, 13]\n\n\n";

    //////////////////////////////////////////////////////////////////////////
    /* cudaDeviceSynchronize(); 
    vec.print();
    std::cout << "Test 4: Reduce operation\n\n\n";
    int reduce_result = vec.reduce(0, Add());
    std::cout << "Reduce result (sum): " << reduce_result << std::endl;
    std::cout << "Espected result: 10\n\n\n"; */

    //////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize(); 
    std::cout << "Test 5: In-place map operation\n\n\n";
    vec.smart_map(DoubleIt()).execute();
    vec.sync_device_to_host();

    std::cout << "Vector after in-place map (double_it): ";
    vec.print(); 
    std::cout << "Espected result: [0, 2, 4, 6, 8]\n\n\n"; 
 
    //////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    std::cout << "Test 6: Multiple map operations\n\n\n";
    rafa::vector<int> vec3(5);
    for (int i = 0; i < 5; ++i) {
        vec3[i] = i + 10;
    }
    vec3.sync_host_to_device();
    rafa::vector<int> result3(5,0);
    vec3.smart_map(DoubleIt(), result3).smart_map(vec, Add(), result3).execute();
    result3.sync_device_to_host();
    std::cout << "Vector after multiple map (double_it, add): ";
    result3.print(); 
 

  
    return 0;
}
