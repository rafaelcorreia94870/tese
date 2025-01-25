//nvcc -std=c++20 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64" --extended-lambda --expt-relaxed-constexpr -Xcompiler /Zc:__cplusplus -o cxx20_test.exe cxx20_test.cu
#include <iostream>
#include <vector>
#include <concepts> // For concepts
#include <ranges>   // For ranges
#include <utility>  // For structured bindings

// Check if C++20 concepts are available
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

// Check a constexpr lambda
constexpr auto constexpr_lambda = [](int x) {
    return x * x;
};

// A kernel to test device-side features
__global__ void test_device_code() {
    // Test constexpr lambda
    constexpr int squared = constexpr_lambda(5);
    printf("Device: constexpr lambda works: 5^2 = %d\n", squared);

    // Test structured bindings
    int arr[2] = {42, 7};
    auto [a, b] = arr;
    printf("Device: Structured bindings work: a = %d, b = %d\n", a, b);

    // Test basic usage of concepts (host-only feature in practice)
#if defined(__cpp_concepts)
    printf("Device: Concepts are supported\n");
#else
    printf("Device: Concepts are NOT supported\n");
#endif
}

int main() {
    // Host-side check for C++20 features
    std::cout << "Host: Testing C++20 features..." << std::endl;

    // Test concepts
    if constexpr (Numeric<int>) {
        std::cout << "Host: Concepts work for int" << std::endl;
    }

    // Test constexpr lambda
    constexpr int squared = constexpr_lambda(6);
    std::cout << "Host: Constexpr lambda works: 6^2 = " << squared << std::endl;

    // Test ranges
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "Host: Ranges work: ";
    for (int n : numbers | std::views::transform([](int x) { return x * x; })) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    // Test structured bindings
    std::pair<int, int> pair = {10, 20};
    auto [x, y] = pair;
    std::cout << "Host: Structured bindings work: x = " << x << ", y = " << y << std::endl;

    // Launch device-side test
    test_device_code<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}
