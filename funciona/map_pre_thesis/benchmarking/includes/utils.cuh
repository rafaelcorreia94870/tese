#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <list>


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
