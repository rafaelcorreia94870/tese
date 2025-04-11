#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <list>
#include <fstream>
#include "skeletons/skeletons.cuh"

#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include <numeric>


template <typename Container, typename Func, typename... Args>
void cpuMap(Container& container, Func func, Args... args) {
    for (auto& elem : container) {
        elem = func(elem, args...);
    }
}

template <typename Container1, typename Container2>
void compareAndPrint(const std::string& name1, const Container1& container1,
                     const std::string& name2, const Container2& container2,
                     const std::string& operationName, float duration1, float duration2,
                     float tolerance = 1e-6) {

    if (container1.size() != container2.size()) {
        std::cout << "Error: Containers " << name1 << " and " << name2 << " have different sizes.\n";
        return;
    }

    bool resultsMatch = true;
    auto iter1 = container1.begin();
    auto iter2 = container2.begin();
    for (size_t i = 0; iter1 != container1.end() && iter2 != container2.end(); ++iter1, ++iter2, ++i) {
        if (std::abs(*iter1 - *iter2) > tolerance) {
            std::cout << name1 << "[" << i << "] = " << *iter1 << " "
                      << name2 << "[" << i << "] = " << *iter2 << "\n";
            resultsMatch = false;
            break;
        }
        if (i == 0) {
            std::cout << name1 << "[" << i << "] = " << *iter1 << " "
                      << name2 << "[" << i << "] = " << *iter2 << "\n";
        }
    }

    std::cout << operationName << " " << name1 << " Time: " << duration1 << " ms\n";
    std::cout << operationName << " " << name2 << " Time: " << duration2 << " ms\n";
    std::cout << operationName << " Results Match: " << (resultsMatch ? "Yes" : "No") << "\n\n";
}

template <typename Func>
auto timeFunction(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

void MandelbrotRender(const size_t width, const size_t height, const int maxIter) {
    size_t size = width * height;
    
    std::vector<int> mandelbrotSet(size);
    std::iota(mandelbrotSet.begin(), mandelbrotSet.end(), 0);
        
    map(mandelbrotSet, MandelbrotFunctor(maxIter, -2.0f, 1.0f, -1.5f, 1.5f, width, height));


    std::ofstream file("mandelbrot.pgm");
    if (!file) {
        std::cerr << "Error: Unable to open file for writing!" << std::endl;
        return;
    }    
    
    file << "P2\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < size; ++i) {
        file << (mandelbrotSet[i] * 255 / maxIter) << " ";
        if ((i + 1) % width == 0) file << "\n";
    }
    
    file.close();
    std::cout << "Mandelbrot set saved as 'mandelbrot.pgm'" << std::endl;
    //////////////////////////////////////////////////////////////

    thrust::device_vector<int> thrust_result(size);
    std::vector<int> thrust_result_host(size);
    thrust::device_vector<int> indices2(size);
    thrust::sequence(indices2.begin(), indices2.end());


    thrust::transform(thrust::device, indices2.begin(), indices2.end(), thrust_result.begin(),
                          MandelbrotFunctor(maxIter, -2.0f, 1.0f, -1.5f, 1.5f, width, height));
    thrust::copy(thrust_result.begin(), thrust_result.end(), thrust_result_host.begin());
    

    std::ofstream file2("mandelbrotthrust.pgm");
    if (!file2) {
        std::cerr << "Error: Unable to open file for writing!" << std::endl;
        return;
    }
    
    file2 << "P2\n" << width << " " << height << "\n255\n";
    
    for (size_t i = 0; i < size; ++i) {
        file2 << (thrust_result_host[i] * 255 / maxIter) << " ";
        if ((i + 1) % width == 0) file2 << "\n";
    }
    
    file2.close();
    std::cout << "Mandelbrot set saved as 'mandelbrotthrust.pgm'" << std::endl;

}