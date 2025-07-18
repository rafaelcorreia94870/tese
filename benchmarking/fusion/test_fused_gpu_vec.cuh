#include "kernel_op.cuh"
#include "fused_gpu_vec.cuh"
#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

//Always passes
int testUnaryMap(size_t n){
    std::cout << "Testing unary map operation..." << std::endl;

    std::vector<float> host_vec(n, 1.0f);
    VectorExt<float> device_vec(host_vec);

    VectorExt<float> result_vec = device_vec.map(SimpleComputation());

    std::vector<float> expected_result(n, 2.0f); // SimpleComputation doubles the value
    std::vector<float> actual_result;
    result_vec.copyToHost(actual_result);

    bool passed = true;
    for (size_t i = 0; i < n; ++i) {
        if (fabs(actual_result[i] - expected_result[i]) > 1e-5) {
            passed = false;
            break;
        }
    }

    std::cout << "Unary Map Result (first 10 elements): ";
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

//Always passes
int testBinaryMap(size_t n) {
    std::cout << "Testing binary map operation..." << std::endl;

    std::vector<float> host_vec1(n, 3.0f);
    std::vector<float> host_vec2(n, 2.0f);

    VectorExt<float> device_vec1(host_vec1);
    VectorExt<float> device_vec2(host_vec2);

    VectorExt<float> result_vec(n);
    result_vec = device_vec1.map(device_vec2, Product());

    std::vector<float> expected_result(n, 6.0f); // Product of 3.0f and 2.0f
    std::vector<float> actual_result;
    result_vec.copyToHost(actual_result);

    bool passed = true;
    for (size_t i = 0; i < n; ++i) {
        if (fabs(actual_result[i] - expected_result[i]) > 1e-5) {
            passed = false;
            break;
        }
    }

    std::cout << "Binary Map Result (first 10 elements): ";
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

//Always passes
int testReduce(size_t n) {
    std::cout << "Testing reduce operation..." << std::endl;

    std::vector<float> host_vec(n,1);

    VectorExt<float> device_vec(host_vec);

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

//Fails if N 10 000 000
int testMapReduce(size_t n) {
    std::cout << "Testing map-reduce operation..." << std::endl;

    std::vector<float> host_vec(n, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        host_vec[i] = static_cast<float>(i + 1);
    }

    VectorExt<float> device_vec(host_vec);


    float result = device_vec.map(SimpleComputation()).reduce(Add<float>(), 0.0f);

    float expected_result;
    double expected_result_d = 0.0;
    SimpleComputation sc;
    for (size_t i = 0; i < n; ++i) {
        expected_result_d += 2.0 * static_cast<double>(host_vec[i]);
    }
    expected_result = static_cast<float>(expected_result_d);

    std::cout << "Map-Reduce Result: " << result << std::endl;
    std::cout << "Expected Result: " << expected_result << std::endl;

    int r = 0;
    if (fabs(result - expected_result) < 1e-5) {
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

    VectorExt<float> device_vec1(host_vec1);
    VectorExt<float> device_vec2(host_vec2);

    VectorExt<float> result_vec(n);
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


//Fails if N 10 000 000 (only fails sum)
int testDifferentReductions(size_t n) {
    std::cout << "Testing different reduction operations..." << std::endl;

    std::vector<float> host_vec(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec[i] = static_cast<float>(i + 1);
    }

    VectorExt<float> device_vec(host_vec);

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
    if (fabs(sum_result - expected_sum) > 1e-5) passed = false;
    if (fabs(max_result - expected_max) > 1e-5) passed = false;
    if (fabs(min_result - expected_min) > 1e-5) passed = false;

    int r = 0;
    if (passed) {
        r = 1;
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}


//Fails if N 10 000 000
int testChainedMapReduce(size_t n) {
    std::cout << "Testing chained map-reduce operation..." << std::endl;

    std::vector<float> host_vec(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec[i] = static_cast<float>(i + 1);
    }

    VectorExt<float> device_vec(host_vec);

    float final_result = device_vec.map(Square<float>())
                          .map(SimpleComputation())
                          .map(Square<float>())
                          .reduce(Add<float>(), 0.0f);

    float expected_result = 0;
    double expected_result_d = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double temp = static_cast<double>(host_vec[i]);
        temp = temp * temp;                  
        temp = SimpleComputation()(temp); 
        temp = temp * temp;                  
        expected_result_d += temp;
    }
    expected_result = static_cast<float>(expected_result_d);

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

//Fails if N 10 000 000
int testParallelOperations(size_t n) {
    std::cout << "Testing parallel map and reduce operations..." << std::endl;

    std::vector<float> host_vec1(n);
    std::vector<float> host_vec2(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec1[i] = static_cast<float>(i + 1);
        host_vec2[i] = static_cast<float>(i * 2 + 1);
    }

    VectorExt<float> device_vec1(host_vec1);
    VectorExt<float> device_vec2(host_vec2);

    VectorExt<float> mapped1(n), mapped2(n), result_vec(n);

    mapped1 = device_vec1.map(Square<float>());
    mapped2 = device_vec2.map(SimpleComputation());

    float product_sum = mapped1.map(mapped2, Product()).reduce(Add<float>(), 0.0f);
    

    float expected_result = 0;
    Square<float> sq;
    SimpleComputation sc;
    Product prod;
    for (size_t i = 0; i < n; ++i) {
        float temp1 = sq(host_vec1[i]);  
        float temp2 = sc(host_vec2[i]); 
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

//Fails if N 10 000 000
int testBinaryMapReduce(size_t n) {
    std::cout << "Testing binary map-reduce operation..." << std::endl;

    std::vector<float> host_vec1(n);
    std::vector<float> host_vec2(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec1[i] = static_cast<float>(i + 1);
        host_vec2[i] = static_cast<float>(i + 1);
    }

    VectorExt<float> device_vec1(host_vec1);
    VectorExt<float> device_vec2(host_vec2);

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

//Always fails
int testComplexMapReduce(size_t n) {
    std::cout << "Testing complex map-reduce operation..." << std::endl;

    std::vector<float> host_vec(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec[i] = 1.0f + static_cast<float>(i) / (n + 1);
    }

    VectorExt<float> device_vec(host_vec);

    float result = device_vec.map(IntensiveComputation<float>()).reduce(ComplexAdd<float>(), 0.0f);

    float expected_result = 0.0f;
    IntensiveComputation<float> ic;
    ComplexAdd<float> ca;

    std::vector<float> mapped_host_vec(n);
    for (size_t i = 0; i < n; ++i) {
        mapped_host_vec[i] = ic(host_vec[i]);
    }
    
    if (n > 0) {
        expected_result = mapped_host_vec[0];
        for (size_t i = 1; i < n; ++i) {
            expected_result = ca(expected_result, mapped_host_vec[i]);
        }
    } else {
        expected_result = 0.0f; 
    }

    std::cout << "Complex Map-Reduce Result: " << result << std::endl;
    std::cout << "Expected Result: " << expected_result << std::endl;
    int r = 0;
    if (std::abs(result - expected_result) < 1e-5 * expected_result) {
        std::cout << "Test passed!" << std::endl;
        r = 1;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return r;
}

//Always passes
int testMapReduceWithDifferentTypes(size_t n) {
    std::cout << "Testing map-reduce with different types..." << std::endl;

    std::vector<long long> host_vec(n);
    for (size_t i = 0; i < n; ++i) {
        host_vec[i] = static_cast<long long>(i + 1);
    }

    VectorExt<long long> device_vec(host_vec);

    long long result = device_vec.map(Square<long long>()).reduce(Add<long long>(), 0LL);

    long long expected_result = 0;
    for (size_t i = 0; i < n; ++i) {
        long long temp = (host_vec[i]) * host_vec[i];
        if (expected_result + temp < expected_result) {
            std::cerr << "Warning: Integer overflow detected in test calculation!" << std::endl;
        }
        expected_result += temp;
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

    counter += testMapReduceWithDifferentTypes(n);
    std::cout << "\n";

    std::cout << "All tests completed.\n Passed: " << counter << " out of 11\n";
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


std::chrono::duration<double> twointensivecomputations_gpu_vec(size_t n, int loop_count) {
    auto start = std::chrono::high_resolution_clock::now();

    VectorExt<float> vec1(n, 1.0f);
    VectorExt<float> result(n);
    result = vec1.map(BenchmarkingComputations(loop_count))
                .map(BenchmarkingComputations(loop_count));

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

std::chrono::duration<double> tensimplecomputations_gpu_vec(size_t n) {
    auto start = std::chrono::high_resolution_clock::now();

    VectorExt<float> vec1(n, 1.0f);
    VectorExt<float> result(n);
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

    VectorExt<float> vec1(n, 1.0f);
    VectorExt<float> result(n);
    result = vec1.map(SimpleComputation());

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

std::chrono::duration<double> reduce_gpu_vec(size_t n) {
    auto start = std::chrono::high_resolution_clock::now();

    VectorExt<float> vec1(n, 1.0f);
    float result = vec1.reduce(Add<float>(), 0.0f);

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}
