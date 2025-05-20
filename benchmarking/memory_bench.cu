#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void benchmarkMemoryTransfers(const size_t MIN_N, const size_t MAX_N, const size_t NUMB_REPEAT) {
    std::cout << "N,Repetition,Pinned Async (ns),Pinned Sync (ns),Pageable Async (ns),Pageable Sync (ns)" << std::endl;

    for (size_t N = MIN_N; N <= MAX_N; N *= 2) {
        const size_t bytes = N * sizeof(float);

        float* hostPageableMemory = (float*)malloc(bytes);
        float* hostPinnedMemory;
        cudaMallocHost((void**)&hostPinnedMemory, bytes);
        float* deviceMemory;
        cudaMalloc((void**)&deviceMemory, bytes);

        for (size_t i = 0; i < N; ++i) {
            hostPageableMemory[i] = static_cast<float>(i);
            hostPinnedMemory[i] = static_cast<float>(i);
        }

        for (int warmup = 0; warmup < 3; ++warmup) {
            cudaMemcpy(deviceMemory, hostPageableMemory, bytes, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            cudaMemcpy(hostPageableMemory, deviceMemory, bytes, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            cudaMemcpy(deviceMemory, hostPinnedMemory, bytes, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            cudaMemcpy(hostPinnedMemory, deviceMemory, bytes, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }

        for (size_t repeat = 0; repeat < NUMB_REPEAT; ++repeat) {
            cudaStream_t stream;
            cudaStreamCreate(&stream);

            auto start = std::chrono::high_resolution_clock::now();
            cudaMemcpyAsync(deviceMemory, hostPinnedMemory, bytes, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(hostPinnedMemory, deviceMemory, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::nano> pinnedAsyncTime = end - start;

            start = std::chrono::high_resolution_clock::now();
            cudaMemcpy(deviceMemory, hostPinnedMemory, bytes, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            cudaMemcpy(hostPinnedMemory, deviceMemory, bytes, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::nano> pinnedSyncTime = end - start;

            start = std::chrono::high_resolution_clock::now();
            cudaMemcpyAsync(deviceMemory, hostPageableMemory, bytes, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(hostPageableMemory, deviceMemory, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::nano> pageableAsyncTime = end - start;

            start = std::chrono::high_resolution_clock::now();
            cudaMemcpy(deviceMemory, hostPageableMemory, bytes, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            cudaMemcpy(hostPageableMemory, deviceMemory, bytes, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::nano> pageableSyncTime = end - start;

            std::cout << N << "," << repeat << ","
                      << std::fixed << std::setprecision(0)
                      << pinnedAsyncTime.count() << ","
                      << pinnedSyncTime.count() << ","
                      << pageableAsyncTime.count() << ","
                      << pageableSyncTime.count() << std::endl;

            cudaStreamDestroy(stream);
        }

        free(hostPageableMemory);
        cudaFreeHost(hostPinnedMemory);
        cudaFree(deviceMemory);
    }
}

void benchmarkMemoryInitialization(const size_t MIN_N, const size_t MAX_N, const size_t NUMB_REPEAT) {
    std::cout << "N,Repetition,Pinned Init (ns),Pageable Init (ns)" << std::endl;

    for (size_t N = MIN_N; N <= MAX_N; N *= 2) {
        const size_t bytes = N * sizeof(float);

        for (size_t repeat = 0; repeat < NUMB_REPEAT; ++repeat) {
            auto start = std::chrono::high_resolution_clock::now();
            float* hostPinnedMemory;
            cudaMallocHost((void**)&hostPinnedMemory, bytes);
            for (size_t i = 0; i < N; ++i) {
                hostPinnedMemory[i] = static_cast<float>(i);
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::nano> pinnedInitTime = end - start;
            cudaFreeHost(hostPinnedMemory);

            start = std::chrono::high_resolution_clock::now();
            float* hostPageableMemory = (float*)malloc(bytes);
            for (size_t i = 0; i < N; ++i) {
                hostPageableMemory[i] = static_cast<float>(i);
            }
            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::nano> pageableInitTime = end - start;
            free(hostPageableMemory);

            std::cout << N << "," << repeat << ","
                      << std::fixed << std::setprecision(0)
                      << pinnedInitTime.count() << ","
                      << pageableInitTime.count() << std::endl;
        }
    }
}

int main() {
    const size_t MIN_N = 10'000;
    const size_t MAX_N = 10'000'000'000;
    const size_t NUMB_REPEAT = 10;

    //benchmarkMemoryTransfers(MIN_N, MAX_N, NUMB_REPEAT);
    benchmarkMemoryInitialization(MIN_N, MAX_N, NUMB_REPEAT);

    return 0;
}
