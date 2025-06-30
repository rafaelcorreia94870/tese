#pragma once
#include "computations.cuh"
#include "utils.cuh"
#include "skeletons/skeletons.cuh"
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <list>
#include "types/two_times_struct.h"
#include <numeric>
#include <stdlib.h>


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
    /* cudaEvent_t start, stop;
    std::cout << "1" << std::endl;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); */
    const float a = 2.0f;

    // CUDA version
    std::vector<float> cuda_saxpy_x(N, 2.0f), cuda_saxpy_y(N, 3.0f);
    //cudaEventRecord(start);
    {
        times.cuda_time = timeFunction([&]() {
            map(cuda_saxpy_x, cuda_saxpy_y, saxpy(), a);
        });
    }
    /* cudaEventRecord(stop);  
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    std::cout << "3" << std::endl; */

    /* float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "4" << std::endl;


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "5" << std::endl;
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    std::cout << "6" << std::endl; */
    // Thrust version
    std::vector<float> thrust_saxpy_x(N, 2.0f), thrust_saxpy_y(N, 3.0f);
    //cudaEventRecord(start2);
    {
        times.thrust_time = timeFunction([&]() {
            thrust::device_vector<float> d_y = thrust_saxpy_y;
            thrust::device_vector<float> d_x = thrust_saxpy_x;
            thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_x.begin(),
            [=] __device__ (float x, float y) { return saxpy()(x, y, a); });
            
            thrust::copy(d_x.begin(), d_x.end(), thrust_saxpy_x.begin());
        });
    }
    /* cudaEventRecord(stop2); 
    std::cout << "7" << std::endl;
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop2);
    std::cout << "8" << std::endl;
    float ms2 = 0;
    cudaEventElapsedTime(&ms2, start2, stop2);
    std::cout << "9" << std::endl;
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2); 

    std::cout << "CUDA time: " << ms << " ms" << "\nThrust time: " << ms2 << " ms" << std::endl;
    */
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
    {

        times.thrust_time = timeFunction([&]() {
            thrust::device_vector<float> d_y = thrust_saxpy_y;
            thrust::device_vector<float> d_x = thrust_saxpy_x;
            thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_x.begin(),
            [=] __device__ (float x, float y) { return saxpy()(x, y, a); });
            
            thrust::copy(d_x.begin(), d_x.end(), thrust_saxpy_x.begin());
        });
    }
    //d_x.clear();
    //d_x.shrink_to_fit();
    //d_y.clear();
    //d_y.shrink_to_fit();



    // CUDA version
    std::vector<float> cuda_saxpy_x(N, 2.0f), cuda_saxpy_y(N, 3.0f);
    {
        times.cuda_time = timeFunction([&]() {
            map(cuda_saxpy_x, cuda_saxpy_y, saxpy(), a);
        });
    }


    if(enable_prints){
        compareAndPrint("cuda_saxpy_x", cuda_saxpy_x, "thrust_saxpy_x", thrust_saxpy_x, 
            "Map (2 Inputs - In-Place with Params)", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct ReduceSum(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    
    // CUDA reduction
    std::vector<int> cuda_result_vector(1);
    {
        std::vector<int> cuda_reduce(N, 1);
        times.cuda_time = timeFunction([&]() {
            cuda_result_vector[0] = reduce(cuda_reduce, 0,Sum());
        });
    }
    
    // Thrust reduction
    std::vector<int> thrust_reduce(N, 1);
    std::vector<int> thrust_result_vector(1);
    {
        times.thrust_time = timeFunction([&]() {
            thrust::device_vector<int> d_vec(thrust_reduce.begin(), thrust_reduce.end());
            thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 0, Sum());
        });
    }

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
    }

    return times;
}

two_times_struct ReduceSumReverse(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    
    // Thrust reduction
    std::vector<int> thrust_reduce(N, 1);
    std::vector<int> thrust_result_vector(1);
    {
        times.thrust_time = timeFunction([&]() {
            thrust::device_vector<int> d_vec(thrust_reduce.begin(), thrust_reduce.end());
            thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 0, Sum());
        });
    }

    // CUDA reduction
    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> cuda_result_vector(1);
    {
        times.cuda_time = timeFunction([&]() {
            cuda_result_vector[0] = reduce(cuda_reduce, 0,Sum());
        });
    }

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
    }

    return times;
}

three_times_struct ReduceSum3Impl(const size_t N, const bool enable_prints = true) {
    three_times_struct times;

    
    // CUDA reduction
    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> cuda_result_vector(1);
    {
        times.cuda_time = timeFunction([&]() {
            cuda_result_vector[0] = reduce(cuda_reduce, 0,Sum());
        });
    }

    // Thrust reduction
    std::vector<int> thrust_reduce(N, 1);
    std::vector<int> thrust_result_vector(1);
    {
        times.thrust_time = timeFunction([&]() {
            thrust::device_vector<int> d_vec(thrust_reduce.begin(), thrust_reduce.end());
            thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 0, Sum());
        });
    }

    // New implementation
    std::vector<int> new_reduce(N, 1);
    std::vector<int> new_result_vector(1);
    {
        times.new_time = timeFunction([&]() {
            new_result_vector[0] = reduce_fast(new_reduce, 0, Sum());
        });
    }

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        compareAndPrint("new_reduce", new_result_vector, "thrust_reduce", thrust_result_vector, "Reduce (New)", times.new_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
        std::cout << "New Reduce Result: " << new_result_vector[0] << std::endl;
    }

    return times;
}

two_times_struct ReduceMult(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    
    // CUDA reduction
    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> cuda_result_vector(1);
    {
        times.cuda_time = timeFunction([&]() {
            cuda_result_vector[0] = reduce(cuda_reduce, 1 ,Multiply());
        });
    }

    // Thrust reduction
    std::vector<int> thrust_reduce(N, 1);
    std::vector<int> thrust_result_vector(1);
    {
        times.thrust_time = timeFunction([&]() {
            thrust::device_vector<int> d_vec(thrust_reduce.begin(), thrust_reduce.end());
            thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 1, Multiply());
        });
    }

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
    }

    return times;

}

three_times_struct ReduceMult3Impl(const size_t N, const bool enable_prints = true) {
    three_times_struct times;

    
    // CUDA reduction
    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> cuda_result_vector(1);
    {
        times.cuda_time = timeFunction([&]() {
            cuda_result_vector[0] = reduce(cuda_reduce, 1,Multiply());
        });
    }

    // Thrust reduction
    std::vector<int> thrust_reduce(N, 1);
    std::vector<int> thrust_result_vector(1);
    {
        times.thrust_time = timeFunction([&]() {
            thrust::device_vector<int> d_vec(thrust_reduce.begin(), thrust_reduce.end());
            thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 1, Multiply());
        });
    }

    // New implementation
    std::vector<int> new_reduce(N, 1);
    std::vector<int> new_result_vector(1);
    {
        times.new_time = timeFunction([&]() {
            new_result_vector[0] = reduce_fast(new_reduce, 1, Multiply());
        });
    }

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        compareAndPrint("new_reduce", new_result_vector, "thrust_reduce", thrust_result_vector, "Reduce (New)", times.new_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
        std::cout << "New Reduce Result: " << new_result_vector[0] << std::endl;
    }

    return times;
}

two_times_struct ReduceMax(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    
    // CUDA reduction
    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> cuda_result_vector(1);
    {
        times.cuda_time = timeFunction([&]() {
            cuda_result_vector[0] = reduce(cuda_reduce, 0,Max());
        });
    }
    
    // Thrust reduction
    std::vector<int> thrust_reduce(N, 1);
    std::vector<int> thrust_result_vector(1);
    {
        times.thrust_time = timeFunction([&]() {
            thrust::device_vector<int> d_vec(thrust_reduce.begin(), thrust_reduce.end());
            thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 0, Max());
        });
    }

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
    }

    return times;
}

two_times_struct ReduceMaxReverse(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    
    // Thrust reduction
    std::vector<int> thrust_reduce(N, 1);
    std::vector<int> thrust_result_vector(1);
    {
        times.thrust_time = timeFunction([&]() {
            thrust::device_vector<int> d_vec(thrust_reduce.begin(), thrust_reduce.end());
            thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 0, Max());
        });
    }
    
    // CUDA reduction
    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> cuda_result_vector(1);
    {
        times.cuda_time = timeFunction([&]() {
            cuda_result_vector[0] = reduce(cuda_reduce, 0,Max());
        });
    }

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
    }

    return times;
}

three_times_struct ReduceMax3Impl(const size_t N, const bool enable_prints = true) {
    three_times_struct times;

    
    // CUDA reduction
    std::vector<int> cuda_reduce(N, 1);
    std::vector<int> cuda_result_vector(1);
    {
        times.cuda_time = timeFunction([&]() {
            cuda_result_vector[0] = reduce(cuda_reduce, 0,Max());
        });
    }

    // Thrust reduction
    std::vector<int> thrust_reduce(N, 1);
    std::vector<int> thrust_result_vector(1);
    {
        times.thrust_time = timeFunction([&]() {
            thrust::device_vector<int> d_vec(thrust_reduce.begin(), thrust_reduce.end());
            thrust_result_vector[0] = thrust::reduce(d_vec.begin(), d_vec.end(), 0, Max());
        });
    }

    // New implementation
    std::vector<int> new_reduce(N, 1);
    std::vector<int> new_result_vector(1);
    {
        times.new_time = timeFunction([&]() {
            new_result_vector[0] = reduce_fast(new_reduce, 0, Max());
        });
    }

    if(enable_prints){
        compareAndPrint("cuda_reduce", cuda_result_vector, "thrust_reduce", thrust_result_vector, "Reduce", times.cuda_time.count(), times.thrust_time.count());
        compareAndPrint("new_reduce", new_result_vector, "thrust_reduce", thrust_result_vector, "Reduce (New)", times.new_time.count(), times.thrust_time.count());
        std::cout << "CUDA Reduce Result: " << cuda_result_vector[0] << std::endl;
        std::cout << "Thrust Reduce Result: " << thrust_result_vector[0] << std::endl;
        std::cout << "New Reduce Result: " << new_result_vector[0] << std::endl;
    }

    return times;
}

two_times_struct IntensiveComputationCompare(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    
    // CUDA Non-In-Place
    std::vector<float> cuda_input(N, 2.0f), cuda_output(N);

    {
        times.cuda_time = timeFunction([&]() {
            map(cuda_input, IntensiveComputationParams(), cuda_output);
        });
    }

    // Thrust Non-In-Place
    std::vector<float> thrust_input(N, 2.0f), thrust_output(N);

    {
        times.thrust_time = timeFunction([&]() {
            thrust::device_vector<float> d_input = thrust_input;
            thrust::device_vector<float> d_output(N);
            thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), IntensiveComputationParams());
            thrust::copy(d_output.begin(), d_output.end(), thrust_output.begin());
        });
    }
        
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

    std::vector<float> thrust_input(N, 2.0f), thrust_output(N);

    {
        // Thrust Non-In-Place
        thrust::device_vector<float> d_input = thrust_input;
        thrust::device_vector<float> d_output(N);
        
        times.thrust_time = timeFunction([&]() {
            thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), IntensiveComputationParams());
            thrust::copy(d_output.begin(), d_output.end(), thrust_output.begin());
        });
    }

    // CUDA Non-In-Place
    std::vector<float> cuda_input(N, 2.0f), cuda_output(N);


    {
        times.cuda_time = timeFunction([&]() {
            map(cuda_input, IntensiveComputationParams(), cuda_output);
        });
    }

    // Print results
    if (enable_prints) {
        compareAndPrint("cuda_1in_inplace", cuda_output, 
                        "thrust_1in_inplace", thrust_output, 
                        "Map (1 Input - In-Place)", 
                        times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct two_thrust(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    std::vector<float> thrust_input(N, 2.0f), thrust_output(N);

    {
        thrust::device_vector<float> d_input = thrust_input;
        thrust::device_vector<float> d_output(N);
        
        times.cuda_time = timeFunction([&]() {
            thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), IntensiveComputationParams());
            thrust::copy(d_output.begin(), d_output.end(), thrust_output.begin());
        });

        std::cout << "Thrust time: " << times.cuda_time.count() << std::endl;
    }  

    
    std::vector<float> thrust_input2(N, 2.0f), thrust_output2(N);

    {  
        thrust::device_vector<float> d_input2 = thrust_input2;
        thrust::device_vector<float> d_output2(N);

        times.thrust_time = timeFunction([&]() {
            thrust::transform(d_input2.begin(), d_input2.end(), d_output2.begin(), IntensiveComputationParams());
            thrust::copy(d_output2.begin(), d_output2.end(), thrust_output2.begin());
        });

        std::cout << "Thrust time: " << times.thrust_time.count() << std::endl;
    } 

    // Print results
    if (enable_prints) {
        compareAndPrint("thrust_first", thrust_output, 
                        "thrust_second", thrust_output2, 
                        "Map (1 Input - In-Place)", 
                        times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct two_cuda(const size_t N, const bool enable_prints = true) {
    two_times_struct times;

    std::vector<float> cuda_input(N, 2.0f), cuda_output(N);
    {
        // CUDA Non-In-Place
        
        times.cuda_time = timeFunction([&]() {
            map(cuda_input, IntensiveComputationParams(), cuda_output);
        });
    }

    std::vector<float> cuda_input2(N, 2.0f), cuda_output2(N);
    {
        // CUDA Non-In-Place
        
        times.thrust_time = timeFunction([&]() {
            map(cuda_input2, IntensiveComputationParams(), cuda_output2);
        });
    }
        
    // Print results
    if (enable_prints) {
        compareAndPrint("cuda_first", cuda_output, 
                        "cuda_second", cuda_output2, 
                        "Map (1 Input - In-Place)", 
                        times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

two_times_struct MandelbrotBenchmark(const size_t width, const size_t height, const int maxIter, const bool enable_prints = true) {
    two_times_struct times;
    size_t N = width * height;

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<int> cuda_result(N, 0);

    times.cuda_time = timeFunction([&]() {
        map(indices, MandelbrotFunctor(maxIter, -2.0f, 1.0f, -1.5f, 1.5f, width, height), cuda_result);
    });

    thrust::device_vector<int> thrust_result(N);
    std::vector<int> thrust_result_host(N);
    thrust::device_vector<int> indices2(N);
    thrust::sequence(indices2.begin(), indices2.end());

    
    times.thrust_time = timeFunction([&]() {
        thrust::transform(thrust::device, indices2.begin(), indices2.end(), thrust_result.begin(),
                            MandelbrotFunctor(maxIter, -2.0f, 1.0f, -1.5f, 1.5f, width, height));
        thrust::copy(thrust_result.begin(), thrust_result.end(), thrust_result_host.begin());
    });
    
    if (enable_prints) {
        compareAndPrint("cuda_result", cuda_result, "thrust_result", thrust_result_host, "Mandelbrot", times.cuda_time.count(), times.thrust_time.count());
    }

    return times;
}

void test_Reduces(const size_t N, const bool enable_prints = true) {
    std::vector<float> float_reduce(N, 2.0f);
    std::vector<int> int_reduce(N, 2);
    std::vector<double> double_reduce(N, 2.0);
    std::vector<uint8_t> bool_reduce(N, 1);
    std::vector<char> string_reduce(N, 'r');
    std::cout << "enable_prints: " << enable_prints << std::endl;
    std::cout << "------------------------------------------------\n";

    // Test Float Reduce
    std::vector<float> float_result(1);
    {
        float_result[0] = reduce_fast(float_reduce, 0.0f, Add<float>());
        if (enable_prints) {
            std::cout << "Float Reduce Result: " << float_result[0] << std::endl;
            std::cout << "Expected Result:     " << N * 2.0f << std::endl;
        }
    }
    std::cout << "------------------------------------------------\n";


    // Test Int Reduce
    std::vector<int> int_result(1);
    {
        int_result[0] = reduce_fast(int_reduce, 0, Add<int>());
        if (enable_prints) {
            std::cout << "Int Reduce Result: " << int_result[0] << std::endl;
            std::cout << "Expected Result:   " << N * 2 << std::endl;
        }
    }
    std::cout << "------------------------------------------------\n";


    // Test Double Reduce
    std::vector<double> double_result(1);
    {
        double_result[0] = reduce_fast(double_reduce, 0.0, Add<double>());
        if (enable_prints) {
            std::cout << "Double Reduce Result: " << double_result[0] << std::endl;
            std::cout << "Expected Result:      " << N * 2.0 << std::endl;
        }
    }

    std::cout << "------------------------------------------------\n";

    // Test Bool Reduce
    std::vector<bool> bool_result(1);
    {
        int xor_result = reduce_fast(bool_reduce, 0, Xor());
        bool final_bool = (xor_result != 0);
        if (enable_prints) {
            std::cout << "Bool Reduce Result: " << final_bool << "\n";
            std::cout << "Expected Result:    " << (N % 2 == 1) << "\n";
        }
    }


    std::cout << "------------------------------------------------\n";

    // Test Char Reduce
    std::vector<char> char_result(1);
    {
        char_result[0] = reduce_fast(string_reduce, ' ', Add<char>());
        if (enable_prints) {
            std::cout << "Char Reduce Result: " << char_result[0] << " (ASCII: " << static_cast<int>(char_result[0]) << ")" << std::endl;
            int expected_sum = 0;
            for (size_t i = 0; i < N; ++i) {
                expected_sum += static_cast<int>(string_reduce[i]); 
            }
            char expected_char = static_cast<char>(expected_sum % 128);
            std::cout << "Expected Result (as character): " << expected_char << " (ASCII: " << static_cast<int>(expected_char) << ")" << std::endl;
        }
    }



    
}