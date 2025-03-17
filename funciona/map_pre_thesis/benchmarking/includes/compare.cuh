#pragma once
#include "computations.cuh"
#include "utils.cuh"
#include "skeletons/skeletons.cuh"
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <list>
#include "types/two_times_struct.h"


two_times_struct oneInputInPlace(const size_t N, const bool enable_prints = true) {
    two_times_struct times;
    // CUDA map without parameters (single input, modifies in-place)

    std::vector<float> cuda_1in_inplace(N, 2.0f);

    times.cuda_time = timeFunction([&]() {
        map(cuda_1in_inplace, IntensiveComputation() );
    });

    // Thrust map equivalent for single input, modifying in-place

    std::vector<float> thrust_1in_inplace(N, 2.0f);

    times.thrust_time = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrust_1in_inplace;
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                          IntensiveComputation());
        thrust::copy(d_vec.begin(), d_vec.end(), thrust_1in_inplace.begin());
    });

    if(enable_prints){
        compareAndPrint("cuda_1in_inplace", cuda_1in_inplace, "thrust_1in_inplace", thrust_1in_inplace, "Map (1 Input - In-Place)", times.cuda_time.count(), times.thrust_time.count());
    }
    return times;
}

two_times_struct oneInputInPlaceParameters(const size_t N, const bool enable_prints = true){
    two_times_struct times;

    // CUDA map without parameters (single input, modifies in-place)

    std::vector<float> cuda_1in_inplace_params(N, 2.0f);

    times.cuda_time = timeFunction([&]() {
        map(cuda_1in_inplace_params, IntensiveComputationParams(), 5, 2.3, true);
    });

    // Thrust map equivalent for single input, modifying in-place

    std::vector<float> thrust_1in_inplace_params(N, 2.0f);

    times.thrust_time = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrust_1in_inplace_params;
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                          IntensiveComputationParams());
        thrust::copy(d_vec.begin(), d_vec.end(), thrust_1in_inplace_params.begin());
    });

    if(enable_prints){
        compareAndPrint("cuda_1in_inplace_params", cuda_1in_inplace_params, "thrust_1in_inplace_params", thrust_1in_inplace_params, "Map (1 Input - In-Place with Params)", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct oneInputOutput(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    // CUDA map without parameters (single input, single output)

    std::vector<float> cuda_1in_output(N, 2.0f), cudaOutput_1in_output(N);

    times.cuda_time = timeFunction([&]() {
        map(cuda_1in_output, IntensiveComputation(), cudaOutput_1in_output);
    });

    // Thrust map equivalent for single input, single output

    std::vector<float> thrust_1in_output(N, 2.0f);
    std::vector<float> thrustOut_1in_output(N);

    times.thrust_time = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrust_1in_output;
        thrust::device_vector<float> d_out(thrustOut_1in_output.size());
        thrust::transform(d_vec.begin(), d_vec.end(), d_out.begin(),
                          IntensiveComputation());
        thrust::copy(d_out.begin(), d_out.end(), thrustOut_1in_output.begin());
    });

    if(enable_prints){
        compareAndPrint("cudaOutput_1in_output", cudaOutput_1in_output, "thrust_1in_output", thrustOut_1in_output, "Map (1 Input - Output)", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct oneInputOutputParameters(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    // CUDA map without parameters (single input, single output)

    std::vector<float> cuda_1in_output_params(N, 2.0f), cudaOutput_1in_output_params(N);

    times.cuda_time = timeFunction([&]() {
        map(cuda_1in_output_params, IntensiveComputationParams(), cudaOutput_1in_output_params, 5, 2.3, true);
    });

    // Thrust map equivalent for single input, single output

    std::vector<float> thrust_1in_output_params(N, 2.0f);
    std::vector<float> thrustOut_1in_output_params(N);

    times.thrust_time = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrust_1in_output_params;
        thrust::device_vector<float> d_out(thrustOut_1in_output_params.size());
        thrust::transform(d_vec.begin(), d_vec.end(), d_out.begin(),
                          IntensiveComputationParams());
        thrust::copy(d_out.begin(), d_out.end(), thrustOut_1in_output_params.begin());
    });

    if(enable_prints){
        compareAndPrint("cudaOutput_1in_output_params", cudaOutput_1in_output_params, "thrust_1in_output_params", thrustOut_1in_output_params, "Map (1 Input - Output with Params)", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}
    
two_times_struct twoInputsInPlace(const size_t N, const bool enable_prints = true){
    two_times_struct times;

    // CUDA map without parameters (two inputs, modifies in-place)

    std::vector<float> cuda_2in_inplace1(N, 2.0f), cuda_2in_inplace2(N, 2.0f);

    times.cuda_time = timeFunction([&]() {
        map(cuda_2in_inplace1, cuda_2in_inplace2, IntensiveComputation2Inputs());
    });

    // Thrust map equivalent for two inputs, modifying in-place

    std::vector<float> thrust_2in_inplace1(N, 2.0f), thrust_2in_inplace2(N, 2.0f);

    times.thrust_time = timeFunction([&]() {
        thrust::device_vector<float> d_vec1 = thrust_2in_inplace1;
        thrust::device_vector<float> d_vec2 = thrust_2in_inplace2;
        thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_vec1.begin(),
                          IntensiveComputation2Inputs());
        thrust::copy(d_vec1.begin(), d_vec1.end(), thrust_2in_inplace1.begin());
    });

    if(enable_prints){
        compareAndPrint("cuda_2in_inplace1", cuda_2in_inplace1, "thrust_2in_inplace1", thrust_2in_inplace1, "Map (2 Inputs - In-Place)", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct twoInputsInPlaceParameters(const size_t N, const bool enable_prints = true){
    two_times_struct times;

    // CUDA map without parameters (two inputs, modifies in-place)

    std::vector<float> cuda_2in_inplace1(N, 2.0f), cuda_2in_inplace2(N, 2.0f);

    times.cuda_time = timeFunction([&]() {
        map(cuda_2in_inplace1, cuda_2in_inplace2, IntensiveComputation2Inputs(), 5, 2.3, true);
    });

    // Thrust map equivalent for two inputs, modifying in-place

    std::vector<float> thrust_2in_inplace1(N, 2.0f), thrust_2in_inplace2(N, 2.0f);

    times.thrust_time = timeFunction([&]() {
        thrust::device_vector<float> d_vec1 = thrust_2in_inplace1;
        thrust::device_vector<float> d_vec2 = thrust_2in_inplace2;
        thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_vec1.begin(),
                          IntensiveComputation2Inputs());
        thrust::copy(d_vec1.begin(), d_vec1.end(), thrust_2in_inplace1.begin());
    });

    if(enable_prints){
        compareAndPrint("cuda_2in_inplace1", cuda_2in_inplace1, "thrust_2in_inplace1", thrust_2in_inplace1, "Map (2 Inputs - In-Place)", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct twoInputsOutput(const size_t N, const bool enable_prints = true){
    two_times_struct times;

    // CUDA map without parameters (two inputs, single output)

    std::vector<float> cuda_2in_output1(N, 2.0f), cuda_2in_output2(N, 2.0f), cudaOutput_2in_output(N);

    times.cuda_time = timeFunction([&]() {
        map(cuda_2in_output1, cuda_2in_output2, IntensiveComputation2Inputs(), cudaOutput_2in_output);
    });

    // Thrust map equivalent for two inputs, single output

    std::vector<float> thrust_2in_output1(N, 2.0f), thrust_2in_output2(N, 2.0f);
    std::vector<float> thrustOut_2in_output(N);

    times.thrust_time = timeFunction([&]() {
        thrust::device_vector<float> d_vec1 = thrust_2in_output1;
        thrust::device_vector<float> d_vec2 = thrust_2in_output2;
        thrust::device_vector<float> d_out(thrustOut_2in_output.size());
        thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_out.begin(),
                          IntensiveComputation2Inputs());
        thrust::copy(d_out.begin(), d_out.end(), thrustOut_2in_output.begin());
    });

    if(enable_prints){
        compareAndPrint("cudaOutput_2in_output", cudaOutput_2in_output, "thrustOut_2in_output", thrustOut_2in_output, "Map (2 Inputs - Output)", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct twoInputsOutputParameters(const size_t N, const bool enable_prints = true){
    two_times_struct times;

    // CUDA map without parameters (two inputs, single output)

    std::vector<float> cuda_2in_output_params1(N, 2.0f), cuda_2in_output_params2(N, 2.0f), cudaOutput_2in_output_params(N);

    times.cuda_time = timeFunction([&]() {
        map(cuda_2in_output_params1, cuda_2in_output_params2, IntensiveComputation2Inputs(), cudaOutput_2in_output_params, 5, 2.3, true);
    });

    // Thrust map equivalent for two inputs, single output

    std::vector<float> thrust_2in_output_params1(N, 2.0f), thrust_2in_output_params2(N, 2.0f);
    std::vector<float> thrustOut_2in_output_params(N);

    times.thrust_time = timeFunction([&]() {
        thrust::device_vector<float> d_vec1 = thrust_2in_output_params1;
        thrust::device_vector<float> d_vec2 = thrust_2in_output_params2;
        thrust::device_vector<float> d_out(thrustOut_2in_output_params.size());
        thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_out.begin(),
                          IntensiveComputation2Inputs());
        thrust::copy(d_out.begin(), d_out.end(), thrustOut_2in_output_params.begin());
    });

    if(enable_prints){
        compareAndPrint("cudaOutput_2in_output_params", cudaOutput_2in_output_params, "thrustOut_2in_output_params", thrustOut_2in_output_params, "Map (2 Inputs - Output with Params)", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct doublePlusA(const size_t N, const bool enable_prints = true){
    two_times_struct times;

    // CUDA map without parameters (single input, modifies in-place)

    std::vector<float> cuda_doublePlusA(N, 2.0f);

    times.cuda_time = timeFunction([&]() {
        map(cuda_doublePlusA, DoublePlusA(), 1);
    });

    // Thrust map equivalent for single input, modifying in-place

    std::vector<float> thrust_doublePlusA(N, 2.0f);

    times.thrust_time = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrust_doublePlusA;
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                          DoublePlusA());
        thrust::copy(d_vec.begin(), d_vec.end(), thrust_doublePlusA.begin());
    });

    if(enable_prints){
        compareAndPrint("cuda_doublePlusA", cuda_doublePlusA, "thrust_doublePlusA", thrust_doublePlusA, "Map (1 Input - In-Place)", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct mysaxpy(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    const float a = 2.0f; 

    // CUDA version
    std::vector<float> cuda_saxpy_x(N, 2.0f), cuda_saxpy_y(N, 3.0f);

    times.cuda_time = timeFunction([&]() {
        map(cuda_saxpy_x, cuda_saxpy_y, saxpy(), a);
    });

    // Thrust version
    std::vector<float> thrust_saxpy_x(N, 2.0f), thrust_saxpy_y(N, 3.0f);
    thrust::device_vector<float> d_y = thrust_saxpy_y;
    thrust::device_vector<float> d_x = thrust_saxpy_x;
    times.thrust_time = timeFunction([&]() {
        thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_x.begin(),
                          [=] __device__ (float x, float y) { return saxpy()(x, y, a); });

        thrust::copy(d_x.begin(), d_x.end(), thrust_saxpy_x.begin());
    });


    if(enable_prints){
        compareAndPrint("cuda_saxpy_x", cuda_saxpy_x, "thrust_saxpy_x", thrust_saxpy_x, 
            "Map (2 Inputs - In-Place with Params)", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct mysaxpyReverse(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    const float a = 2.0f; 

    
    // Thrust version
    std::vector<float> thrust_saxpy_x(N, 2.0f), thrust_saxpy_y(N, 3.0f);
    thrust::device_vector<float> d_y = thrust_saxpy_y;
    thrust::device_vector<float> d_x = thrust_saxpy_x;
    times.thrust_time = timeFunction([&]() {
        thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_x.begin(),
                          [=] __device__ (float x, float y) { return saxpy()(x, y, a); });

        thrust::copy(d_x.begin(), d_x.end(), thrust_saxpy_x.begin());
    });
    d_x.clear();
    d_x.shrink_to_fit();
    d_y.clear();
    d_y.shrink_to_fit();



    // CUDA version
    std::vector<float> cuda_saxpy_x(N, 2.0f), cuda_saxpy_y(N, 3.0f);
    std::cout << "Running CUDA map with two inputs..." << std::endl;
    times.cuda_time = timeFunction([&]() {
        map(cuda_saxpy_x, cuda_saxpy_y, saxpy(), a);
    });


    if(enable_prints){
        compareAndPrint("cuda_saxpy_x", cuda_saxpy_x, "thrust_saxpy_x", thrust_saxpy_x, 
            "Map (2 Inputs - In-Place with Params)", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct ReduceSum(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> thrust_reduce(N, 1);

    // CUDA reduction
    std::vector<int> cuda_result_vector(1);
    times.cuda_time = timeFunction([&]() {
        cuda_result_vector[0] = reduce(cuda_reduce, 0,Sum());
    });

    // Thrust reduction
    thrust::device_vector<int> d_vec(cuda_reduce.begin(), cuda_reduce.end());
    std::vector<int> thrust_result_vector(1);
    times.thrust_time = timeFunction([&]() {
        thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 0, Sum());
    });

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
    }

    return times;
}

two_times_struct ReduceSumReverse(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> thrust_reduce(N, 1);

    // Thrust reduction
    thrust::device_vector<int> d_vec(cuda_reduce.begin(), cuda_reduce.end());
    std::vector<int> thrust_result_vector(1);
    times.thrust_time = timeFunction([&]() {
        thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 0, Sum());
    });

    // CUDA reduction
    std::vector<int> cuda_result_vector(1);
    times.cuda_time = timeFunction([&]() {
        cuda_result_vector[0] = reduce(cuda_reduce, 0,Sum());
    });

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
    }

    return times;
}

two_times_struct ReduceMult(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> thrust_reduce(N, 1);

    // CUDA reduction
    std::vector<int> cuda_result_vector(1);
    times.cuda_time = timeFunction([&]() {
        cuda_result_vector[0] = reduce(cuda_reduce, 1 ,Multiply());
    });

    // Thrust reduction
    thrust::device_vector<int> d_vec(cuda_reduce.begin(), cuda_reduce.end());
    std::vector<int> thrust_result_vector(1);
    times.thrust_time = timeFunction([&]() {
        thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 1, Multiply());
    });

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
    }

    return times;

}

two_times_struct ReduceMax(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> thrust_reduce(N, 1);

    // CUDA reduction
    std::vector<int> cuda_result_vector(1);
    times.cuda_time = timeFunction([&]() {
        cuda_result_vector[0] = reduce(cuda_reduce, 0,Max());
    });

    // Thrust reduction
    thrust::device_vector<int> d_vec(cuda_reduce.begin(), cuda_reduce.end());
    std::vector<int> thrust_result_vector(1);
    times.thrust_time = timeFunction([&]() {
        thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 0, Max());
    });

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
    }

    return times;
}

two_times_struct ReduceMaxReverse(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> thrust_reduce(N, 1);

    // Thrust reduction
    thrust::device_vector<int> d_vec(cuda_reduce.begin(), cuda_reduce.end());
    std::vector<int> thrust_result_vector(1);
    times.thrust_time = timeFunction([&]() {
        thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 0, Max());
    });

    // CUDA reduction
    std::vector<int> cuda_result_vector(1);
    times.cuda_time = timeFunction([&]() {
        cuda_result_vector[0] = reduce(cuda_reduce, 0,Max());
    });

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
    }

    return times;
}

two_times_struct IntensiveComputationCompare(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    std::vector<float> cuda_input(N, 2.0f), cuda_output(N);
    std::vector<float> thrust_input(N, 2.0f), thrust_output(N);

    // CUDA Non-In-Place
    times.cuda_time = timeFunction([&]() {
        map(cuda_input, IntensiveComputationParams(), cuda_output);
    });

    // Thrust Non-In-Place
    thrust::device_vector<float> d_input = thrust_input;
    thrust::device_vector<float> d_output(N);

    times.thrust_time = timeFunction([&]() {
        thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), IntensiveComputationParams());
        thrust::copy(d_output.begin(), d_output.end(), thrust_output.begin());
    });

    // Print results
    if (enable_prints) {
        compareAndPrint("cuda_1in_inplace", cuda_output, 
                        "thrust_1in_inplace", thrust_output, 
                        "Map (1 Input - In-Place)", 
                        times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}


two_times_struct IntensiveComputationCompareReverse(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    std::vector<float> cuda_input(N, 2.0f), cuda_output(N);
    std::vector<float> thrust_input(N, 2.0f), thrust_output(N);

    
    // Thrust Non-In-Place
    thrust::device_vector<float> d_input = thrust_input;
    thrust::device_vector<float> d_output(N);

    times.thrust_time = timeFunction([&]() {
        thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), IntensiveComputationParams());
        thrust::copy(d_output.begin(), d_output.end(), thrust_output.begin());
    });

    // CUDA Non-In-Place
    times.cuda_time = timeFunction([&]() {
        map(cuda_input, IntensiveComputationParams(), cuda_output);
    });

    // Print results
    if (enable_prints) {
        compareAndPrint("cuda_1in_inplace", cuda_output, 
                        "thrust_1in_inplace", thrust_output, 
                        "Map (1 Input - In-Place)", 
                        times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}