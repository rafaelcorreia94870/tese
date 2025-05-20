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

template <typename Func, typename Func2, typename Tin, typename Tout>
__global__ void fusedKernel(Func func, Func2 func2, Tin* input, Tout* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 8) {
        output[idx] = func2(func(input[idx]));
    }
}

template <typename Func, typename Func2, typename Tin, typename Tout>
void launchKernel(Func func, Func2 func2, Tin* input, Tout* output) {
    int blockSize = 256;
    int numBlocks = (8 + blockSize - 1) / blockSize;

    Tin* d_input;
    Tout* d_output;

    cudaMalloc((void**)&d_input, 8 * sizeof(Tin));
    cudaMalloc((void**)&d_output, 8 * sizeof(Tout));

    cudaMemcpy(d_input, input, 8 * sizeof(Tin), cudaMemcpyHostToDevice);

    fusedKernel<<<numBlocks, blockSize>>>(func, func2, d_input, d_output);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, 8 * sizeof(Tout), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

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

    for (int i = 0; i < 8; ++i) {
        input[i] = i;
    }
    std::cout << "Input vector: ";
    printVector(input);

    launchKernel(First(), Second(), input.data(), output_float.data());
    std::cout << "Output after fused kernel: ";
    printVector(output_float);


    // Test with SecondFail - this should fail but it doesn't (compatible types?)
    std::vector<int> input2(8);
    for (int i = 0; i < 8; ++i) {
        input2[i] = i;
    }
    std::vector<int> output_int(8);
    std::cout << "Input vector for composed kernel with SecondFail: ";
    printVector(input2);
    launchKernel(First(), SecondFail(), input2.data(), output_int.data());
    std::cout << "Output after composed kernel with SecondFail: ";
    printVector(output_int);


    return 0;
}
