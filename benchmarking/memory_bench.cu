#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <fstream>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

struct IntensiveComputation {
    __device__ float operator()(float x) const {
        for (int i = 0; i < 100; ++i) {
            x = sinf(x) * cosf(x) + logf(x + 1.0f);
        }
        return x;
    }
};

__global__ void computeKernel(float* data, size_t N) {
    IntensiveComputation op;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = op(data[idx]);
    }
}

void runWarmup(size_t N) {
    const size_t bytes = N * sizeof(float);
    float* hostPageable = (float*)malloc(bytes);
    float* hostPinned;
    cudaMallocHost((void**)&hostPinned, bytes);
    float* device;
    cudaMalloc((void**)&device, bytes);
    float* managed;
    cudaMallocManaged(&managed, bytes);

    for (size_t i = 0; i < N; ++i) {
        hostPageable[i] = i;
        hostPinned[i] = i;
        managed[i] = i;
    }

    for (int i = 0; i < 3; ++i) {
        cudaMemcpy(device, hostPageable, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(hostPageable, device, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(device, hostPinned, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(hostPinned, device, bytes, cudaMemcpyDeviceToHost);
        computeKernel<<<(N + 255) / 256, 256>>>(managed, N);
        cudaDeviceSynchronize();
    }

    free(hostPageable);
    cudaFreeHost(hostPinned);
    cudaFree(device);
    cudaFree(managed);
}

void benchmarkMemoryInitialization(const size_t MIN_N, const size_t MAX_N, const size_t NUMB_REPEAT) {
    std::cout << "N,Repetition,Pinned Init (ns),Pageable Init (ns),Managed Init (ns)\n";

    for (size_t N = MIN_N; N <= MAX_N; N *= 2) {
        const size_t bytes = N * sizeof(float);
        for (size_t repeat = 0; repeat < NUMB_REPEAT; ++repeat) {
            auto start = std::chrono::high_resolution_clock::now();
            float* pinned;
            cudaMallocHost((void**)&pinned, bytes);
            for (size_t i = 0; i < N; ++i) pinned[i] = i;
            auto end = std::chrono::high_resolution_clock::now();
            auto pinnedInit = end - start;
            cudaFreeHost(pinned);

            start = std::chrono::high_resolution_clock::now();
            float* pageable = (float*)malloc(bytes);
            for (size_t i = 0; i < N; ++i) pageable[i] = i;
            end = std::chrono::high_resolution_clock::now();
            auto pageableInit = end - start;
            free(pageable);

            start = std::chrono::high_resolution_clock::now();
            float* managed;
            cudaMallocManaged(&managed, bytes);
            for (size_t i = 0; i < N; ++i) managed[i] = i;
            cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            auto managedInit = end - start;
            cudaFree(managed);

            std::cout << N << "," << repeat << ","
                      << std::fixed << std::setprecision(0)
                      << pinnedInit.count() << ","
                      << pageableInit.count() << ","
                      << managedInit.count() << "\n";
        }
    }
}

void benchmarkMemoryTransfers(const size_t MIN_N, const size_t MAX_N, const size_t NUMB_REPEAT) {
    std::cout << "N,Repetition,Pinned Transfer (ns),Pageable Transfer (ns),Managed Kernel (ns),Managed Prefetch+Kernel (ns)\n";

    for (size_t N = MIN_N; N <= MAX_N; N *= 2) {
        const size_t bytes = N * sizeof(float);

        float* pinned;
        cudaMallocHost((void**)&pinned, bytes);
        float* pageable = (float*)malloc(bytes);
        float* managed;
        cudaMallocManaged(&managed, bytes);
        float* device;
        cudaMalloc((void**)&device, bytes);

        for (size_t i = 0; i < N; ++i) {
            pinned[i] = i;
            pageable[i] = i;
            managed[i] = i;
        }
        cudaDeviceSynchronize();

        for (size_t repeat = 0; repeat < NUMB_REPEAT; ++repeat) {
            auto start = std::chrono::high_resolution_clock::now();
            cudaMemcpy(device, pinned, bytes, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            computeKernel<<<(N + 255) / 256, 256>>>(device, N);
            cudaDeviceSynchronize();
            cudaMemcpy(pinned, device, bytes, cudaMemcpyDeviceToHost);
            auto end = std::chrono::high_resolution_clock::now();
            auto pinnedTransfer = end - start;

            start = std::chrono::high_resolution_clock::now();
            cudaMemcpy(device, pageable, bytes, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            computeKernel<<<(N + 255) / 256, 256>>>(device, N);
            cudaDeviceSynchronize();
            cudaMemcpy(pageable, device, bytes, cudaMemcpyDeviceToHost);
            end = std::chrono::high_resolution_clock::now();
            auto pageableTransfer = end - start;

            start = std::chrono::high_resolution_clock::now();
            computeKernel<<<(N + 255) / 256, 256>>>(managed, N);
            cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            auto managedKernel = end - start;

            start = std::chrono::high_resolution_clock::now();
            cudaMemPrefetchAsync(managed, bytes, 0);
            computeKernel<<<(N + 255) / 256, 256>>>(managed, N);
            cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            auto managedPrefetch = end - start;

            std::cout << N << "," << repeat << ","
                      << std::fixed << std::setprecision(0)
                      << pinnedTransfer.count() << ","
                      << pageableTransfer.count() << ","
                      << managedKernel.count() << ","
                      << managedPrefetch.count() << "\n";
        }

        cudaFreeHost(pinned);
        free(pageable);
        cudaFree(managed);
        cudaFree(device);
    }
}

void benchmarkMemoryInitAndTransfer(const size_t MIN_N, const size_t MAX_N, const size_t NUMB_REPEAT) {
    std::cout << "N,Repetition,Pinned Init+Transfer (ns),Pageable Init+Transfer (ns),Managed Init+Kernel+Prefetch (ns)\n";

    for (size_t N = MIN_N; N <= MAX_N; N *= 2) {
        const size_t bytes = N * sizeof(float);

        for (size_t repeat = 0; repeat < NUMB_REPEAT; ++repeat) {
            auto start = std::chrono::high_resolution_clock::now();
            float* pinned;
            cudaMallocHost((void**)&pinned, bytes);
            for (size_t i = 0; i < N; ++i) pinned[i] = i;
            float* device;
            cudaMalloc((void**)&device, bytes);
            cudaMemcpy(device, pinned, bytes, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            computeKernel<<<(N + 255) / 256, 256>>>(device, N);
            cudaDeviceSynchronize();
            cudaMemcpy(pinned, device, bytes, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            auto pinnedTime = end - start;
            cudaFreeHost(pinned);
            cudaFree(device);

            start = std::chrono::high_resolution_clock::now();
            float* pageable = (float*)malloc(bytes);
            for (size_t i = 0; i < N; ++i) pageable[i] = i;
            cudaMalloc((void**)&device, bytes);
            cudaMemcpy(device, pageable, bytes, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            computeKernel<<<(N + 255) / 256, 256>>>(device, N);
            cudaDeviceSynchronize();
            cudaMemcpy(pageable, device, bytes, cudaMemcpyDeviceToHost);
            end = std::chrono::high_resolution_clock::now();
            auto pageableTime = end - start;
            free(pageable);
            cudaFree(device);

            start = std::chrono::high_resolution_clock::now();
            float* managed;
            cudaMallocManaged(&managed, bytes);
            for (size_t i = 0; i < N; ++i) managed[i] = i;
            cudaMemPrefetchAsync(managed, bytes, 0);
            computeKernel<<<(N + 255) / 256, 256>>>(managed, N);
            cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            auto managedTime = end - start;
            cudaFree(managed);

            std::cout << N << "," << repeat << ","
                      << std::fixed << std::setprecision(0)
                      << pinnedTime.count() << ","
                      << pageableTime.count() << ","
                      << managedTime.count() << "\n";
        }
    }
}

size_t estimateMaxN(size_t buffers_required = 3, double safety_factor = 0.9) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t usable_bytes = static_cast<size_t>(free_mem * safety_factor);
    size_t max_floats = usable_bytes / (sizeof(float) * buffers_required);
    std::cout << "Estimated max N: " << max_floats << "\n";
    return max_floats;
}


int main() {
    const size_t MIN_N = 10'000;
    const size_t MAX_N = estimateMaxN(4);
    const size_t NUMB_REPEAT = 20;

    runWarmup(MAX_N);
    std::cout << "Running Init Benchmark\n";
    std::ofstream outFile("sheet/mem_init_fix.csv");
    auto cout_buf = std::cout.rdbuf();
    std::cout.rdbuf(outFile.rdbuf());
    benchmarkMemoryInitialization(MIN_N, MAX_N, NUMB_REPEAT);
    std::cout.rdbuf(cout_buf);
    outFile.close();
    std::cout << "Running Transfer Benchmark\n";
    std::ofstream outFile2("sheet/mem_transfer_fix.csv");
    cout_buf = std::cout.rdbuf();
    std::cout.rdbuf(outFile2.rdbuf());
    benchmarkMemoryTransfers(MIN_N, MAX_N, NUMB_REPEAT);
    std::cout.rdbuf(cout_buf);
    outFile2.close();
    std::cout << "Running Init and Transfer Benchmark\n";
    std::ofstream outFile3("sheet/mem_init_transfer_fix.csv");
    cout_buf = std::cout.rdbuf();
    std::cout.rdbuf(outFile3.rdbuf());
    benchmarkMemoryInitAndTransfer(MIN_N, MAX_N, NUMB_REPEAT);
    std::cout.rdbuf(cout_buf);
    outFile3.close();

    return 0;
}
