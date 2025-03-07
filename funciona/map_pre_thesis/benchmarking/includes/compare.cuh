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


void oneInputInPlace(const size_t N) {
    // CUDA map without parameters (single input, modifies in-place)

    std::vector<float> cuda_1in_inplace(N, 2.0f);

    auto cuda_time_1in_inplace = timeFunction([&]() {
        map(cuda_1in_inplace, IntensiveComputation() );
    });

    // Thrust map equivalent for single input, modifying in-place

    std::vector<float> thrust_1in_inplace(N, 2.0f);

    auto thrust_time_1in_inplace = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrust_1in_inplace;
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                          IntensiveComputation());
        thrust::copy(d_vec.begin(), d_vec.end(), thrust_1in_inplace.begin());
    });

    compareAndPrint("cuda_1in_inplace", cuda_1in_inplace, "thrust_1in_inplace", thrust_1in_inplace, "Map (1 Input - In-Place)", cuda_time_1in_inplace.count(), thrust_time_1in_inplace.count());
}

void oneInputInPlaceParameters(const size_t N){ 
    // CUDA map without parameters (single input, modifies in-place)

    std::vector<float> cuda_1in_inplace_params(N, 2.0f);

    auto cuda_time_1in_inplace_params = timeFunction([&]() {
        map(cuda_1in_inplace_params, IntensiveComputationParams(), 5, 2.3, true);
    });

    // Thrust map equivalent for single input, modifying in-place

    std::vector<float> thrust_1in_inplace_params(N, 2.0f);

    auto thrust_time_1in_inplace_params = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrust_1in_inplace_params;
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                          IntensiveComputationParams());
        thrust::copy(d_vec.begin(), d_vec.end(), thrust_1in_inplace_params.begin());
    });

    compareAndPrint("cuda_1in_inplace_params", cuda_1in_inplace_params, "thrust_1in_inplace_params", thrust_1in_inplace_params, "Map (1 Input - In-Place with Params)", cuda_time_1in_inplace_params.count(), thrust_time_1in_inplace_params.count());
}

void oneInputOutput(const size_t N) {
    // CUDA map without parameters (single input, single output)

    std::vector<float> cuda_1in_output(N, 2.0f), cudaOutput_1in_output(N);

    auto cuda_time_1in_output = timeFunction([&]() {
        map(cuda_1in_output, IntensiveComputation(), cudaOutput_1in_output);
    });

    // Thrust map equivalent for single input, single output

    std::vector<float> thrust_1in_output(N, 2.0f);
    std::vector<float> thrustOut_1in_output(N);

    auto thrust_time_1in_output = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrust_1in_output;
        thrust::device_vector<float> d_out(thrustOut_1in_output.size());
        thrust::transform(d_vec.begin(), d_vec.end(), d_out.begin(),
                          IntensiveComputation());
        thrust::copy(d_out.begin(), d_out.end(), thrustOut_1in_output.begin());
    });

    compareAndPrint("cudaOutput_1in_output", cudaOutput_1in_output, "thrust_1in_output", thrustOut_1in_output, "Map (1 Input - Output)", cuda_time_1in_output.count(), thrust_time_1in_output.count());
}

void oneInputOutputParameters(const size_t N) { 
    // CUDA map without parameters (single input, single output)

    std::vector<float> cuda_1in_output_params(N, 2.0f), cudaOutput_1in_output_params(N);

    auto cuda_time_1in_output_params = timeFunction([&]() {
        map(cuda_1in_output_params, IntensiveComputationParams(), cudaOutput_1in_output_params, 5, 2.3, true);
    });

    // Thrust map equivalent for single input, single output

    std::vector<float> thrust_1in_output_params(N, 2.0f);
    std::vector<float> thrustOut_1in_output_params(N);

    auto thrust_time_1in_output_params = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrust_1in_output_params;
        thrust::device_vector<float> d_out(thrustOut_1in_output_params.size());
        thrust::transform(d_vec.begin(), d_vec.end(), d_out.begin(),
                          IntensiveComputationParams());
        thrust::copy(d_out.begin(), d_out.end(), thrustOut_1in_output_params.begin());
    });

    compareAndPrint("cudaOutput_1in_output_params", cudaOutput_1in_output_params, "thrust_1in_output_params", thrustOut_1in_output_params, "Map (1 Input - Output with Params)", cuda_time_1in_output_params.count(), thrust_time_1in_output_params.count());
}
    
void twoInputsInPlace(const size_t N){
    // CUDA map without parameters (two inputs, modifies in-place)

    std::vector<float> cuda_2in_inplace1(N, 2.0f), cuda_2in_inplace2(N, 2.0f);

    auto cuda_time_2in_inplace = timeFunction([&]() {
        map(cuda_2in_inplace1, cuda_2in_inplace2, IntensiveComputation2Inputs());
    });

    // Thrust map equivalent for two inputs, modifying in-place

    std::vector<float> thrust_2in_inplace1(N, 2.0f), thrust_2in_inplace2(N, 2.0f);

    auto thrust_time_2in_inplace = timeFunction([&]() {
        thrust::device_vector<float> d_vec1 = thrust_2in_inplace1;
        thrust::device_vector<float> d_vec2 = thrust_2in_inplace2;
        thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_vec1.begin(),
                          IntensiveComputation2Inputs());
        thrust::copy(d_vec1.begin(), d_vec1.end(), thrust_2in_inplace1.begin());
    });

    compareAndPrint("cuda_2in_inplace1", cuda_2in_inplace1, "thrust_2in_inplace1", thrust_2in_inplace1, "Map (2 Inputs - In-Place)", cuda_time_2in_inplace.count(), thrust_time_2in_inplace.count());
}

void twoInputsInPlaceParameters(const size_t N){ 
    // CUDA map without parameters (two inputs, modifies in-place)

    std::vector<float> cuda_2in_inplace1(N, 2.0f), cuda_2in_inplace2(N, 2.0f);

    auto cuda_time_2in_inplace = timeFunction([&]() {
        map(cuda_2in_inplace1, cuda_2in_inplace2, IntensiveComputation2Inputs(), 5, 2.3, true);
    });

    // Thrust map equivalent for two inputs, modifying in-place

    std::vector<float> thrust_2in_inplace1(N, 2.0f), thrust_2in_inplace2(N, 2.0f);

    auto thrust_time_2in_inplace = timeFunction([&]() {
        thrust::device_vector<float> d_vec1 = thrust_2in_inplace1;
        thrust::device_vector<float> d_vec2 = thrust_2in_inplace2;
        thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_vec1.begin(),
                          IntensiveComputation2Inputs());
        thrust::copy(d_vec1.begin(), d_vec1.end(), thrust_2in_inplace1.begin());
    });

    compareAndPrint("cuda_2in_inplace1", cuda_2in_inplace1, "thrust_2in_inplace1", thrust_2in_inplace1, "Map (2 Inputs - In-Place)", cuda_time_2in_inplace.count(), thrust_time_2in_inplace.count());
}

void twoInputsOutput(const size_t N){
    // CUDA map without parameters (two inputs, single output)

    std::vector<float> cuda_2in_output1(N, 2.0f), cuda_2in_output2(N, 2.0f), cudaOutput_2in_output(N);

    auto cuda_time_2in_output = timeFunction([&]() {
        map(cuda_2in_output1, cuda_2in_output2, IntensiveComputation2Inputs(), cudaOutput_2in_output);
    });

    // Thrust map equivalent for two inputs, single output

    std::vector<float> thrust_2in_output1(N, 2.0f), thrust_2in_output2(N, 2.0f);
    std::vector<float> thrustOut_2in_output(N);

    auto thrust_time_2in_output = timeFunction([&]() {
        thrust::device_vector<float> d_vec1 = thrust_2in_output1;
        thrust::device_vector<float> d_vec2 = thrust_2in_output2;
        thrust::device_vector<float> d_out(thrustOut_2in_output.size());
        thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_out.begin(),
                          IntensiveComputation2Inputs());
        thrust::copy(d_out.begin(), d_out.end(), thrustOut_2in_output.begin());
    });

    compareAndPrint("cudaOutput_2in_output", cudaOutput_2in_output, "thrustOut_2in_output", thrustOut_2in_output, "Map (2 Inputs - Output)", cuda_time_2in_output.count(), thrust_time_2in_output.count());
}

void twoInputsOutputParameters(const size_t N){
    // CUDA map without parameters (two inputs, single output)

    std::vector<float> cuda_2in_output_params1(N, 2.0f), cuda_2in_output_params2(N, 2.0f), cudaOutput_2in_output_params(N);

    auto cuda_time_2in_output_params = timeFunction([&]() {
        map(cuda_2in_output_params1, cuda_2in_output_params2, IntensiveComputation2Inputs(), cudaOutput_2in_output_params, 5, 2.3, true);
    });

    // Thrust map equivalent for two inputs, single output

    std::vector<float> thrust_2in_output_params1(N, 2.0f), thrust_2in_output_params2(N, 2.0f);
    std::vector<float> thrustOut_2in_output_params(N);

    auto thrust_time_2in_output_params = timeFunction([&]() {
        thrust::device_vector<float> d_vec1 = thrust_2in_output_params1;
        thrust::device_vector<float> d_vec2 = thrust_2in_output_params2;
        thrust::device_vector<float> d_out(thrustOut_2in_output_params.size());
        thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_out.begin(),
                          IntensiveComputation2Inputs());
        thrust::copy(d_out.begin(), d_out.end(), thrustOut_2in_output_params.begin());
    });

    compareAndPrint("cudaOutput_2in_output_params", cudaOutput_2in_output_params, "thrustOut_2in_output_params", thrustOut_2in_output_params, "Map (2 Inputs - Output with Params)", cuda_time_2in_output_params.count(), thrust_time_2in_output_params.count());
}

void doublePlusA(const size_t N){
    // CUDA map without parameters (single input, modifies in-place)

    std::vector<float> cuda_doublePlusA(N, 2.0f);

    auto cuda_time_doublePlusA = timeFunction([&]() {
        map(cuda_doublePlusA, DoublePlusA(), 1);
    });

    // Thrust map equivalent for single input, modifying in-place

    std::vector<float> thrust_doublePlusA(N, 2.0f);

    auto thrust_time_doublePlusA = timeFunction([&]() {
        thrust::device_vector<float> d_vec = thrust_doublePlusA;
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                          DoublePlusA());
        thrust::copy(d_vec.begin(), d_vec.end(), thrust_doublePlusA.begin());
    });

    compareAndPrint("cuda_doublePlusA", cuda_doublePlusA, "thrust_doublePlusA", thrust_doublePlusA, "Map (1 Input - In-Place)", cuda_time_doublePlusA.count(), thrust_time_doublePlusA.count());
}

void mysaxpy(const size_t N) {
    const float a = 2.0f;  // Scalar parameter

    // CUDA version
    std::vector<float> cuda_saxpy_x(N, 2.0f), cuda_saxpy_y(N, 3.0f);

    auto cuda_time_saxpy = timeFunction([&]() {
        map(cuda_saxpy_x, cuda_saxpy_y, saxpy(), a);
    });

    // Thrust version
    std::vector<float> thrust_saxpy_x(N, 2.0f), thrust_saxpy_y(N, 3.0f);

    auto thrust_time_saxpy = timeFunction([&]() {
        thrust::device_vector<float> d_x = thrust_saxpy_x;
        thrust::device_vector<float> d_y = thrust_saxpy_y;

        // Capture `a` in a device lambda
        thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_x.begin(),
                          [=] __device__ (float x, float y) { return saxpy()(x, y, a); });

        thrust::copy(d_x.begin(), d_x.end(), thrust_saxpy_x.begin());
    });

    compareAndPrint("cuda_saxpy_x", cuda_saxpy_x, "thrust_saxpy_x", thrust_saxpy_x, 
                    "Map (2 Inputs - In-Place with Params)", cuda_time_saxpy.count(), thrust_time_saxpy.count());
}
