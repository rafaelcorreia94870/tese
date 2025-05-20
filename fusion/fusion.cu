#include <iostream>
#include <vector>
#include <cuda_runtime.h>

struct First {
    __device__ __host__ float operator()(int x) const {
        return static_cast<float>(x) + 0.5f;
    }
};

struct Second {
    __device__ __host__ float operator()(float x) const {
        return x * 2.0f;
    }
};

struct SecondFail {
    __device__ __host__ int operator()(int x) const {
        return x * 2;
    }
};

template <typename Func, typename Tin, typename Tout>
__global__ void kernelWithFunction(Func func, Tin* input, Tout* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 8) {
        output[idx] = func(input[idx]);
    }
}

template <typename Func, typename Tin, typename Tout>
void launchKernel(Func func, Tin* input, Tout* output) {
    int blockSize = 256;
    int numBlocks = (8 + blockSize - 1) / blockSize;

    Tin* d_input;
    Tout* d_output;

    cudaMalloc((void**)&d_input, 8 * sizeof(Tin));
    cudaMalloc((void**)&d_output, 8 * sizeof(Tout));

    cudaMemcpy(d_input, input, 8 * sizeof(Tin), cudaMemcpyHostToDevice);

    kernelWithFunction<<<numBlocks, blockSize>>>(func, d_input, d_output);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, 8 * sizeof(Tout), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

template <typename Func1, typename Func2, typename Tin, typename Tout>
struct ComposedKernel {
    Func1 func1;
    Func2 func2;

    __device__ __host__ ComposedKernel(Func1 f1, Func2 f2) : func1(f1), func2(f2) {}

    __device__ __host__ Tout operator()(Tin x) const {
        return func2(func1(x));
    }
};

template <typename T>
void printVector(const std::vector<T>& vec) {
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> input(8);
    std::vector<float> output_float(8);
    std::vector<float> output_float2(8);

    for (int i = 0; i < 8; ++i) {
        input[i] = i;
    }
    std::cout << "Input vector: ";
    printVector(input);

    launchKernel(First(), input.data(), output_float.data());
    launchKernel(Second(), output_float.data(), output_float2.data());
    std::cout << "Output after first kernel: ";
    printVector(output_float); 
    std::cout << "Output after second kernel: ";
    printVector(output_float2);

    std::vector<int> input2(8);
    for (int i = 0; i < 8; ++i) {
        input2[i] = i;
    }
    std::vector<float> output_float3(8);
    auto composed = ComposedKernel<First, Second, int, float>(First(), Second());

    std::cout << "Input vector for composed kernel: ";
    printVector(input2);

    launchKernel(composed, input2.data(), output_float3.data());
    std::cout << "Output after composed kernel: ";
    printVector(output_float3);

    // Test with SecondFail - this should fail but it doesn't (compatible types?)
    std::vector<int> input3(8);
    for (int i = 0; i < 8; ++i) {
        input3[i] = i;
    }
    std::vector<int> output_int(8);
    auto composedFail = ComposedKernel<First, SecondFail, int, int>(First(), SecondFail());
    std::cout << "Input vector for composed kernel with SecondFail: ";
    printVector(input3);
    launchKernel(composedFail, input3.data(), output_int.data());
    std::cout << "Output after composed kernel with SecondFail: ";
    printVector(output_int);


    return 0;
}
