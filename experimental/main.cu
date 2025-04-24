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
    std::cout << "Espected result: [0, 2, 4, 6, 8]\n\n\n"; */
 
    //////////////////////////////////////////////////////////////////////////
    /* cudaDeviceSynchronize();
    std::cout << "Test 6: Multiple map operations\n\n\n";
    rafa::vector<int> vec3(5);
    for (int i = 0; i < 5; ++i) {
        vec3[i] = i + 10;
    }
    vec3.sync_host_to_device();
    rafa::vector<int> result3(5,0);
    vec3.map(DoubleIt(), result3).map(Add(), vec, result3);
    result3.sync_device_to_host();
    std::cout << "Vector after multiple map (double_it, add): ";
    result3.print();
 */

    rafa::vector<int> input1(5, 1);
    rafa::vector<int> input2(5, 2);
    rafa::vector<int> result3(5);

    /* auto obj1 = new rafa::SkeletonObject<DoubleIt, rafa::vector<int>>(
        "Map", { &input1 }, DoubleIt{}
    ); */

    /* rafa::SkeletonObject<DoubleIt, rafa::vector<int>> obj2(
        "Map", { &input1 }, DoubleIt{}, result3
    ); */

   /*  rafa::SkeletonObject<Add, rafa::vector<int>, int> obj3(
        "Reduce", { &input1 }, Add{}, result3, 5
    );
    
    rafa::SkeletonObject<AddAndShift, rafa::vector<int>, int> obj4(
        "ZipMapWithArgs", { &input1, &input2 }, AddAndShift{}, 5
    ); */

    //obj1->print();
    //obj1->execute();

    //input1.sync_device_to_host();
    std::cout << "Input1 after map: ";
    //input1.print();
    std::cout << "Expected result: [2, 2, 2, 2, 2]\n\n";
    std::cout << "\n";
    //obj2.print();
    std::cout << "\n";
    /* obj3.print();
    std::cout << "\n";
    obj4.print();
    std::cout << "\n";  */

    //input1.smart_map(DoubleIt());
    (rafa::vector<int>)input1.smart_map(DoubleIt()).smart_map(input2, Add(), result3).execute(); 


/* 
    auto func = DoubleIt();
    auto func2 = Add();
    //print func and func2 types
    std::cout << "func type: " << typeid(func).name() << std::endl;
    std::cout << "func2 type: " << typeid(func2).name() << std::endl;

    rafa::SkeletonObject<DoubleIt, rafa::vector<int>> obj(
        "Map",
        {vec},
        DoubleIt{}
    );

    rafa::SkeletonObject<Add, rafa::vector<int>> obj2(
        "Map",
        {vec},
        result2,  // assuming result2 is a rafa::vector<int>
        Add{}
    );

    rafa::vector<int> inputAdd(5, 2);

    rafa::SkeletonObject<Add, rafa::vector<int>, int> obj_with_args(
        "Map",
        { &inputAdd },   // one input
        Add{},         // a functor that takes (x, y)
        10             // additional argument to add
    );
    

    obj.print();

    rafa::vector<int> input1(5, 2);
    rafa::vector<int> input2(5, 3);

    rafa::SkeletonObject<AddAndShift, rafa::vector<int>, int> obj_zip_args(
        "ZipMapWithArgs",
        { &input1, &input2 },  // two inputs
        AddAndShift{},
        5  // z
    ); */

    /* rafa::SkelQueue<rafa::SkeletonObject> skel_queue;
    skel_queue.push(skel_obj);
    skel_queue.print(); */

    return 0;
}
