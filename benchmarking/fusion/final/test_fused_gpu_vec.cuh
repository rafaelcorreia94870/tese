#include "kernel_op.cuh"
#include "fused_gpu_vec.cuh"
#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

//Always passes
int testUnaryMap(size_t n) {
    std::cout << "Testing unary map operation..." << std::endl;

    

    lcuda::Vector<float> result_vec(n, 1.0f);
    result_vec = result_vec.map(SimpleComputation());

    std::vector<float> expected_result(n, 2.0f); // SimpleComputation doubles the value

    bool passed = true;
    for (size_t i = 0; i < n; ++i) {
        if (fabs(result_vec[i] - expected_result[i]) > 1e-5) {
            passed = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected_result[i] << ", got " << result_vec[i] << std::endl;
            break;
        }
    }

    std::cout << "Unary Map Result (first 10 elements): ";
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        std::cout << result_vec[i] << " ";
    }
    std::cout << "\n";

    int r = 0;
    if (passed) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}

//Always passes
int testBinaryMap(size_t n) {
    std::cout << "Testing binary map operation..." << std::endl;

    lcuda::Vector<float> device_vec1(n, 3.0f);
    lcuda::Vector<float> device_vec2(n, 2.0f);

    device_vec1 = device_vec1.map(device_vec2, Product());

    std::vector<float> expected_result(n, 6.0f); // Product of 3.0f and 2.0f

    bool passed = true;
    for (size_t i = 0; i < n; ++i) {
        if (fabs(device_vec1[i] - expected_result[i]) > 1e-5) {
            passed = false;
            break;
        }
    }

    std::cout << "Binary Map Result (first 10 elements): ";
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        std::cout << device_vec1[i] << " ";
    }
    std::cout << "\n";

    int r = 0;
    if (passed) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}

//Always passes
int testReduce(size_t n) {
    std::cout << "Testing reduce operation..." << std::endl;

    std::vector<float> host_vec(n,1);

    lcuda::Vector<float> device_vec(host_vec);

    float result = device_vec.reduce(Add<float>(), 0.0f);


    std::cout << "Reduce Result: " << result << std::endl;
    std::cout << "Expected Result: " << n << std::endl;

    int r = 0;
    if (fabs(result - (n)) < 1e-5) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}

int testMapReduce(size_t n) {
    std::cout << "Testing map-reduce operation..." << std::endl;

    std::vector<double> host_vec(n, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        host_vec[i] = static_cast<double>(i + 1);
    }

    lcuda::Vector<double> device_vec(host_vec);


    double result = device_vec.map(SimpleComputation()).reduce(Add<double>(), 0.0f);

    double expected_result;
    double expected_result_d = 0.0;
    SimpleComputation sc;
    for (size_t i = 0; i < n; ++i) {
        expected_result_d += 2.0 * static_cast<double>(host_vec[i]);
    }
    expected_result = static_cast<double>(expected_result_d);

    std::cout << "Map-Reduce Result: " << result << std::endl;
    std::cout << "Expected Result: " << expected_result << std::endl;

    int r = 0;
    if (fabs(result - expected_result) < 1e-9) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}

//Always passes
int testMapWithTwoInputs(size_t n) {
    std::cout << "Testing map with two inputs..." << std::endl;

    std::vector<float> host_vec1(n);
    std::vector<float> host_vec2(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec1[i] = static_cast<float>(i + 1);
        host_vec2[i] = static_cast<float>(i + 1);
    }

    lcuda::Vector<float> device_vec1(host_vec1);
    lcuda::Vector<float> device_vec2(host_vec2);

    lcuda::Vector<float> result_vec(n);
    result_vec = device_vec1.map(device_vec2, Product());

    std::vector<float> expected_result(n);
    Product prod;
    for (size_t i = 0; i < n; ++i) {
        expected_result[i] = prod(host_vec1[i], host_vec2[i]);
    }

    std::vector<float> actual_result;
    result_vec.copyToHost(actual_result);

    bool passed = true;
    for (size_t i = 0; i < n; ++i) {
        if (fabs(actual_result[i] - expected_result[i]) > 1e-5) {
            passed = false;
            break;
        }
    }

    std::cout << "Map with Two Inputs Result (first 10 elements): ";
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        std::cout << actual_result[i] << " ";
    }
    std::cout << "\n";

    int r = 0;
    if (passed) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}

int testMapReduceWithTwoInputs(size_t n) {
    std::cout << "Testing map-reduce with two inputs..." << std::endl;

    std::vector<float> host_vec1(n);
    std::vector<float> host_vec2(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec1[i] = static_cast<float>(i + 1);
        host_vec2[i] = static_cast<float>(i + 1);
    }

    lcuda::Vector<float> device_vec1(host_vec1);
    lcuda::Vector<float> device_vec2(host_vec2);

    float result = device_vec1.map(device_vec2, Product()).reduce(Add<float>(), 0.0f);

    double expected_result_d = 0.0;
    Product prod;
    for (size_t i = 0; i < n; ++i) {
        expected_result_d += prod(host_vec1[i], host_vec2[i]);
    }
    float expected_result = static_cast<float>(expected_result_d);


    std::cout << "Map-Reduce with Two Inputs Result: " << result << std::endl;
    std::cout << "Expected Result: " << expected_result << std::endl;

    int r = 0;
    if (fabs(result - expected_result) < 1e-4 * fabs(expected_result)) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}



int testDifferentReductions(size_t n) {
    std::cout << "Testing different reduction operations..." << std::endl;

    std::vector<float> host_vec(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec[i] = static_cast<float>(i + 1);
    }

    lcuda::Vector<float> device_vec(host_vec);

    // Test sum reduction
    float sum_result = device_vec.reduce(Add<float>(), 0.0f);
    float expected_sum = 0;
    double expected_sum_d = 0.0;
    for (size_t i = 0; i < n; ++i) {
        expected_sum_d += static_cast<double>(host_vec[i]);
    }
    expected_sum = static_cast<float>(expected_sum_d);
    std::cout << "Sum Reduction Result: " << sum_result << std::endl;
    std::cout << "Expected Sum: " << expected_sum << std::endl;

    // Test max reduction
    float max_result = device_vec.reduce(Max(), -std::numeric_limits<float>::max());
    float expected_max = *std::max_element(host_vec.begin(), host_vec.end());
    std::cout << "Max Reduction Result: " << max_result << std::endl;
    std::cout << "Expected Max: " << expected_max << std::endl;

    // Test min reduction
    float min_result = device_vec.reduce(Min(), std::numeric_limits<float>::max());
    float expected_min = *std::min_element(host_vec.begin(), host_vec.end());
    std::cout << "Min Reduction Result: " << min_result << std::endl;
    std::cout << "Expected Min: " << expected_min << std::endl;

    bool passed = true;
    int r = 0;
    if (fabs(sum_result - expected_sum) > 1e-5){
        passed = false;
        std::cout << "Sum reduction failed!" << std::endl;
    } else{
        std::cout << "Sum reduction passed!" << std::endl;
        r++;
    }
        
    if (fabs(max_result - expected_max) > 1e-5) {
        passed = false;
        std::cout << "Max reduction failed!" << std::endl;
    } else {
        std::cout << "Max reduction passed!" << std::endl;
        r++;
    }


    if (fabs(min_result - expected_min) > 1e-5) {
        passed = false;
        std::cout << "Min reduction failed!" << std::endl;
    } else {
        std::cout << "Min reduction passed!" << std::endl;
        r++;
    }

    if (passed) {
        std::cout << "All reductions passed!" << std::endl;
    } else {
        std::cout << "Some or all reductions failed!" << std::endl;
    }


    return r;
}


int testChainedMapReduce(size_t n) {
    std::cout << "Testing chained map-reduce operation..." << std::endl;

    std::vector<double> host_vec(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec[i] = static_cast<double>(i + 1);
    }

    lcuda::Vector<double> device_vec(host_vec);

    double final_result = device_vec.map(Square<double>())
                          .map(DoubleIt<double>())
                          .map(Square<double>())
                          .reduce(Add<double>(), 0.0f);

    double expected_result = 0;
    double expected_result_d = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double temp = static_cast<double>(host_vec[i]);
        temp = temp * temp;                  
        temp = DoubleIt<double>()(temp); 
        temp = temp * temp;                  
        expected_result_d += temp;
    }
    expected_result = static_cast<double>(expected_result_d);

    std::cout << "Chained Map-Reduce Result: " << final_result << std::endl;
    std::cout << "Expected Result: " << expected_result << std::endl;

    int r = 0;
    
    if (std::abs(final_result - expected_result) < 1e-5 * expected_result) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}


//Float overflows
int testParallelOperations(size_t n) {
    std::cout << "Testing parallel map and reduce operations..." << std::endl;

    std::vector<float> host_vec1(n);
    std::cout << "host_vec1 initialized." << std::endl;
    std::vector<float> host_vec2(n);
    std::cout << "host_vec2 initialized." << std::endl;
    for (size_t i = 0; i < n; ++i) {
        host_vec1[i] = static_cast<float>(i + 1.0);
        host_vec2[i] = static_cast<float>(i * 2.0 + 1.0);
    }

    lcuda::Vector<float> device_vec1(host_vec1);
    std::cout << "device_vec1 initialized." << std::endl;
    host_vec1.clear();
    host_vec1.shrink_to_fit();


    lcuda::Vector<float> device_vec2(host_vec2);
    std::cout << "device_vec2 initialized." << std::endl;

    host_vec2.clear();
    host_vec2.shrink_to_fit();

    std::cout << "Applying parallel operations..." << std::endl;
    device_vec1 = device_vec1.map(Square<float>());
    device_vec2 = device_vec2.map(SimpleComputation());
    std::cout << "Parallel operations applied." << std::endl;
    cudaDeviceSynchronize();

    float product_sum = device_vec1.map(device_vec2, Product()).reduce(Add<float>(), 0.0f);
    std::cout << "Product Sum Result: " << product_sum << std::endl;

    float expected_result = 0;
    Square<float> sq;
    SimpleComputation sc;
    Product prod;
    for (size_t i = 0; i < n; ++i) {
        float temp1 = sq(i + 1.0f);
        float temp2 = sc(i * 2.0f + 1.0f); 
        expected_result += (prod(temp1, temp2));
    }

    std::cout << "Parallel Operations Result: " << product_sum << std::endl;
    std::cout << "Expected Result: " << expected_result << std::endl;

    int r = 0;
    if (std::abs(product_sum - expected_result) < 1e-5 * expected_result) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;

    return 0; 
}


int testBinaryMapReduce(size_t n) {
    std::cout << "Testing binary map-reduce operation..." << std::endl;

    std::vector<float> host_vec1(n);
    std::vector<float> host_vec2(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec1[i] = static_cast<float>(i + 1);
        host_vec2[i] = static_cast<float>(i + 1);
    }

    lcuda::Vector<float> device_vec1(host_vec1);
    lcuda::Vector<float> device_vec2(host_vec2);

    float result = device_vec1.map(device_vec2, Product()).reduce(Add<float>(), 0.0f);

    float expected_result = 0;
    Product prod;
    for (size_t i = 0; i < n; ++i) {
        expected_result += prod(host_vec1[i], host_vec2[i]);
    }

    std::cout << "Binary Map-Reduce Result: " << result << std::endl;
    std::cout << "Expected Result: " << expected_result << std::endl;

    int r = 0;
    if (std::abs(result - expected_result) < 1e-5 * expected_result) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
    return 0; 
}


//Fails because of the ComplexAdd is not associative
int testComplexMapReduce(size_t n) {
    std::cout << "Testing complex map-reduce operation..." << std::endl;

    std::vector<float> host_vec(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec[i] = 1.0f + static_cast<float>(i) / (n + 1);
    }

    lcuda::Vector<float> device_vec(host_vec);

    float result = device_vec.map(IntensiveComputation<float>()).reduce(ComplexAdd<float>(), 0.0f);

    IntensiveComputation<float> ic;
    ComplexAdd<float> ca;

    float expected_result = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        expected_result = ca(expected_result, ic(host_vec[i]));
    }

    std::cout << "Complex Map-Reduce Result: " << result << std::endl;
    std::cout << "Expected Result: " << expected_result << std::endl;

    int r = 0;
    if (std::abs(result - expected_result) < 1e-5 * std::max(1.0f, std::abs(expected_result))) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}


int testMapReduceWithDifferentTypes(size_t n) {
    std::cout << "Testing map-reduce with different types..." << std::endl;
    std::cout << "Vector size: " << n << std::endl;

    // Use a Vector constructor that only allocates device memory
    lcuda::Vector<long long> device_vec(n);
    device_vec.fill_with_sequence();
    device_vec.synchronize(); // Ensure the fill is complete

    long long result = device_vec.map(Square<long long>()).reduce(Add<long long>(), 0LL);

    // Note: This expected_result calculation will overflow for large 'n'.
    // It is shown here for smaller test cases where overflow is not an issue.
    // For large 'n', a different verification strategy is needed.
    long long expected_result = 0;
    for (size_t i = 0; i < n; ++i) {
        long long val = static_cast<long long>(i + 1);
        expected_result += val * val;
    }

    std::cout << "Integer Map-Reduce Result: " << result << std::endl;
    std::cout << "Expected Result: " << expected_result << std::endl;

    int r = 0;
    if (result == expected_result) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}

int testMapTwoOutputs(size_t n) {
    std::cout << "Testing map with two outputs (unary and binary)..." << std::endl;
    bool passed = true;
    int r = 0;

    // --- Unary Test ---
    std::cout << "Running unary in-place test..." << std::endl;
    std::vector<float> host_in(n);
    for (size_t i = 0; i < n; ++i) {
        host_in[i] = static_cast<float>(i) * 0.1f;
    }

    lcuda::Vector<float> sin_vec(host_in); // This vector will be modified in-place
    lcuda::Vector<float> cos_vec(n);

    host_in.clear();
    host_in.shrink_to_fit();


    // Use the new in-place syntax. `sin_vec` is both input and output.
    sin_vec.map(SineCosine<float>(), cos_vec);


    std::cout << "Unary Map Result (first 10 elements): ";
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        std::cout << sin_vec[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Cosine Map Result (first 10 elements): ";
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        std::cout << cos_vec[i] << " ";
    }
    std::cout << "\n";

    // Verify results
    for (size_t i = 0; i < n; ++i) {
        float expected_sin = sinf(i*0.1f);
        float expected_cos = cosf(i*0.1f);
        if (fabs(sin_vec[i] - expected_sin) > 1e-5 || fabs(cos_vec[i] - expected_cos) > 1e-5) {
            passed = false;
            std::cerr << "Mismatch at index " << i << ": expected (" << expected_sin << ", " << expected_cos
                      << "), got (" << sin_vec[i] << ", " << cos_vec[i] << ")" << std::endl;
            break;
        }
    }

    if (passed) {
        std::cout << "Unary test passed!" << std::endl;
        r += 1;
    } else {
        std::cout << "Unary test failed!" << std::endl;
    }

    // Clean up
    sin_vec.~Vector();
    cos_vec.~Vector();
    host_in.clear();
    host_in.shrink_to_fit();



    // Fails at 16777217 because of the size of the vectors
    // --- Binary Test ---

    passed = true; 
    
    std::cout << "Running binary in-place test..." << std::endl;
    std::vector<float> host_in1(n), host_in2(n);
    for(size_t i = 0; i < n; ++i) {
        host_in1[i] = static_cast<float>(i + 1);
        host_in2[i] = static_cast<float>(n - i);
    }
    lcuda::Vector<float> sum_vec(host_in1);
    lcuda::Vector<float> dev_in2(host_in2);
    lcuda::Vector<float> prod_vec(n);

    host_in1.clear();
    host_in2.clear();
    host_in1.shrink_to_fit();
    host_in2.shrink_to_fit();

    sum_vec.map(dev_in2, SumAndProduct<float>(), prod_vec);


    // Print results
    

    std::cout << "Sum Map Result (first 10 elements): ";
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        std::cout << std::setprecision(20) << sum_vec[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Product Map Result (first 10 elements): ";
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        std::cout << std::setprecision(20) << prod_vec[i] << " ";
    }
    std::cout << "\n";

    // Verify results

    
    for (size_t i = 0; i < n; ++i) {
        float expected_sum = (i + 1.0f) + (n - i);
        float expected_prod = (i + 1.0f) * (n - i);
        if (fabs(sum_vec[i] - expected_sum) > 1.0e-9 || fabs(prod_vec[i] - expected_prod) > 1.0e-9) {
            passed = false;
            std::cerr << std::setprecision(20) << "Mismatch at index " << i << ": expected (" << expected_sum << ", " << expected_prod
                      << "), got (" << sum_vec[i] << ", " << prod_vec[i] << ")" << std::endl;
            break;
        }
    }

    if (passed) {
        std::cout << "Binary test passed!" << std::endl;
        r += 1;
    } else {
        std::cout << "Binary test failed!" << std::endl;
    }

    return r;
}

int testScalarOperations(size_t n) {
    std::cout << "Testing scalar operations..." << std::endl;

    lcuda::Vector<float> vec(n, 2.0f); // Vector of all 2.0s

    // Chain multiple scalar and map operations
    vec = (vec + 3.0f) * 2.0f; // (2.0 + 3.0) * 2.0 = 10.0
    vec = vec.map(Square<float>());   // 10.0 * 10.0 = 100.0
    vec = 200.0f - vec;               // 200.0 - 100.0 = 100.0
    vec = vec / 4.0f;                 // 100.0 / 4.0 = 25.0

    // Expected result is a vector of all 25.0s
    float expected_result = 25.0f;
    bool passed = true;
    for (size_t i = 0; i < n; ++i) {
        if (fabs(vec[i] - expected_result) > 1e-5) {
            passed = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected_result << ", got " << vec[i] << std::endl;
            break;
        }
    }

    int r = 0;
    if (passed) {
        std::cout << "Test passed! Final value: " << vec[0] << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}

int fusionBenchmark(size_t n = 1000) {
    cudaFree(0);
    std::cout << "Running fusion benchmark with vector size: " << n << "\n\n";
    // --- Warm-up Phase ---
    auto start = std::chrono::high_resolution_clock::now();
    lcuda::Vector<float> warm_up_vec(n);
    warm_up_vec = warm_up_vec.map(IntensiveComputation<float>())
                .map(IntensiveComputation<float>());
    cudaDeviceSynchronize(); 
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Warm-up Duration: " << duration.count() << " ms\n";


    warm_up_vec.~Vector(); 
    //benchmark our map map fusion with a manual fusion
    lcuda::Vector<float> vec1(n, 1.0f);
    cudaDeviceSynchronize();
    
    start = std::chrono::high_resolution_clock::now();
    lcuda::Vector<float> result(n);
    result = vec1.map(IntensiveComputation<float>())
                .map(IntensiveComputation<float>());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Composed Unary Fusion Duration: " << duration.count() << " ms\n";


    ComposeUnaryUnary op = ComposeUnaryUnary(IntensiveComputation<float>(), IntensiveComputation<float>());
    start = std::chrono::high_resolution_clock::now();
    lcuda::Vector<float> result2(n);
    result2 = vec1.map(op);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Manual Fusion Duration: " << duration.count() << " ms\n";

    std::vector<float> host_result2;
    result2.copyToHost(host_result2);
    result2.~Vector();

    start = std::chrono::high_resolution_clock::now();
    lcuda::Vector<float> result3(n);
    result3 = vec1.map(TwoIntensiveComputations<float>());
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Two Simple Computations Duration: " << duration.count() << " ms\n\n";
 
    std::vector<float> host_result3;
    result3.copyToHost(host_result3);
    result3.~Vector();

    
    
    std::vector<float> host_result;
    result.copyToHost(host_result);
    result.~Vector();

    bool passed = true;

    for (size_t i = 0; i < n; ++i) {
        if (fabs(host_result[i] - host_result2[i]) > 1e-5 || 
            fabs(host_result[i] - host_result3[i]) > 1e-5) {
            passed = false;
            break;
        }
    }

    //print the first 10 elements of the results
    std::cout << "Composed Unary Fusion Result (first 10 elements): ";
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        std::cout << host_result[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Manual Fusion Result (first 10 elements): ";
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        std::cout << host_result2[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Two Simple Computations Result (first 10 elements): ";
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        std::cout << host_result3[i] << " ";
    }
    std::cout << "\n";

    if (passed) {
        std::cout << "Fusion benchmark passed!\n\n";
    } else {
        std::cout << "Fusion benchmark failed!\n\n";
    }
    return passed ? 1 : 0;
}

void runAllTests(size_t n = 1000) {
    std::cout << "Running all tests with vector size: " << n << "\n\n";
    int counter = 0;

    counter += testUnaryMap(n);
    std::cout << "\n";

    counter += testBinaryMap(n);
    std::cout << "\n";

    counter += testReduce(n);
    std::cout << "\n";

    counter += testMapReduce(n);
    std::cout << "\n";

    counter += testMapWithTwoInputs(n);
    std::cout << "\n";

    counter += testMapReduceWithTwoInputs(n);
    std::cout << "\n";

    counter += testDifferentReductions(n); //returns 0-3
    std::cout << "\n";

    counter += testChainedMapReduce(n);
    std::cout << "\n";

    counter += testParallelOperations(n);
    std::cout << "\n";

    counter += testComplexMapReduce(n);
    std::cout << "\n";

    counter += testBinaryMapReduce(n);
    std::cout << "\n";

    //counter += testComplexMapReduce(n);
    //std::cout << "\n";
    //this test fails because of the ComplexAdd is not associative

    counter += testMapReduceWithDifferentTypes(n);
    std::cout << "\n";

    counter += testMapTwoOutputs(n); //returns 0-2
    std::cout << "\n";

    counter += testScalarOperations(n);
    std::cout << "\n";

    std::cout << "All tests completed.\n Passed: " << counter << " out of 17\n\n";
}

void runSimpleTests(size_t n = 1000) {
    std::cout << "Running simple tests with vector size: " << n << "\n";
    int counter = 0;
    counter += testUnaryMap(n);
    std::cout << "\n";
    counter += testBinaryMap(n);
    std::cout << "\n";
    counter += testReduce(n);
    std::cout << "\n";
    counter += testMapReduce(n);
    std::cout << "\n";

    std::cout << "Simple tests completed.\n Passed: " << counter << " out of 4\n";
}

void runReductionTests(size_t n = 1000) {
    std::cout << "Running reduction tests with vector size: " << n << "\n";
    int counter = 0;
    counter += testMapReduce(n);
    std::cout << "\n";
    counter += testDifferentReductions(n);
    std::cout << "\n";
    counter += testChainedMapReduce(n);
    std::cout << "\n";
    counter += testParallelOperations(n);
    std::cout << "\n";
    counter += testComplexMapReduce(n);
    std::cout << "\n";
    counter += testBinaryMapReduce(n);
    std::cout << "\n";

    std::cout << "Reduction tests completed.\n Passed: " << counter << " out of 6\n";
}

void runMapTests(size_t n = 1000) {
    std::cout << "Running map tests with vector size: " << n << "\n";
    int counter = 0;
    counter += testUnaryMap(n);
    std::cout << "\n";
    counter += testBinaryMap(n);
    std::cout << "\n";
    counter += testMapWithTwoInputs(n);
    std::cout << "\n";
    counter += testMapReduceWithTwoInputs(n);
    std::cout << "\n";
    counter += testMapReduceWithDifferentTypes(n);
    std::cout << "\n";
    counter += testMapTwoOutputs(n);
    std::cout << "\n";

    std::cout << "Map tests completed.\n Passed: " << counter << " out of 6\n";
}

std::chrono::duration<double> twointensivecomputations_gpu_vec(size_t n, int loop_count) {
    auto start = std::chrono::high_resolution_clock::now();

    lcuda::Vector<float> vec1(n, 1.0f);
    lcuda::Vector<float> result(n);
    result = vec1.map(BenchmarkingComputations(loop_count))
                .map(BenchmarkingComputations(loop_count));

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

std::chrono::duration<double> tensimplecomputations_gpu_vec(size_t n) {
    auto start = std::chrono::high_resolution_clock::now();

    lcuda::Vector<float> vec1(n, 1.0f);
    lcuda::Vector<float> result(n);
    result = vec1.map(SimpleComputation()).map(SimpleComputation())
                .map(SimpleComputation()).map(SimpleComputation())
                .map(SimpleComputation()).map(SimpleComputation())
                .map(SimpleComputation()).map(SimpleComputation())
                .map(SimpleComputation()).map(SimpleComputation());

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

std::chrono::duration<double> singlecomputation_gpu_vec(size_t n) {
    auto start = std::chrono::high_resolution_clock::now();

    lcuda::Vector<float> vec1(n, 1.0f);
    lcuda::Vector<float> result(n);
    result = vec1.map(SimpleComputation());

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

std::chrono::duration<double> reduce_gpu_vec(size_t n) {
    auto start = std::chrono::high_resolution_clock::now();

    lcuda::Vector<float> vec1(n, 1.0f);
    float result = vec1.reduce(Add<float>(), 0.0f);

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}
