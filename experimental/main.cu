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
    __device__ __host__ int operator()(int x, int y) const {
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
 /* 
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
    cudaDeviceSynchronize(); 
    vec.print();
    std::cout << "Test 4: Reduce operation\n\n\n";
    int reduce_result = vec.reduce(0, Add());
    std::cout << "Reduce result (sum): " << reduce_result << std::endl;
    std::cout << "Espected result: 10\n\n\n"; 

    //////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize(); 
    std::cout << "Test 5: In-place map operation\n\n\n";
    vec.smart_map(DoubleIt()).execute();

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
    rafa::vector<int> debug(5,0);
    vec3.smart_map(DoubleIt(), debug).execute();
    std::cout << "Vector after first map (double_it): ";
    debug.print();
    std::cout << "Espected result: [20, 22, 24, 26, 28]\n\n\n";

    rafa::vector<int> debug2(5,0);
    debug.smart_map(vec, Add(), debug2).execute();
    std::cout << "Vector after second map (add): ";
    debug2.print();
    std::cout << "Espected result: [20, 24, 28, 32, 36]\n\n\n";

    std::cout << "vec3 queue size: " << vec3.skel_queue.size() << std::endl;
    vec3.print();
    //10, 11, 12, 13, 14
    //20, 22, 24, 26, 28
    //20, 24, 28, 32, 36
    rafa::vector<int> result3(5,0);
    std::cout << "vec3: ";
    vec3.print();
    std::cout << "vec: ";
    vec.print();


    vec3.smart_map(DoubleIt(), result3).smart_map(vec, Add(), result3).execute();



    std::cout << "Vector after multiple map (double_it, add): ";
    result3.print();
    std::cout << "Espected result: [20, 24, 28, 32, 36]\n\n\n";
    ////////////////////////////////////////////////////////////////////////// */
    rafa::vector<int> input_blucas(5,5);
    rafa::vector<int> input_blucas2(5,7);
    rafa::vector<int> output_muamua(5);
    std::cout << "input_blucas device pointer: " << input_blucas.device_data << "\n";
    std::cout << "input_blucas2 device pointer: " << input_blucas2.device_data << "\n";
    std::cout << "output_muamua device pointer: " << output_muamua.device_data << "\n";
    //input_blucas.smart_map(DoubleIt()).smart_map(DoubleIt(), output_muamua).smart_map(DoubleIt()).execute();
    //input_blucas.smart_map(DoubleIt()).smart_map(DoubleIt(), output_muamua).execute();
    //input_blucas.print();
    input_blucas.smart_map(DoubleIt(), output_muamua).smart_map(input_blucas2, Add()).execute();

//    input_blucas.smart_map(DoubleIt()).smart_map(input_blucas2, Add(), output_muamua).execute();
    //input_blucas.smart_map(DoubleIt()).smart_map(input_blucas2, Add()).execute();
    input_blucas.print();
    output_muamua.sync_device_to_host();
    output_muamua.print();

    return 0;
}
