#include "../types/types.cuh"
#include "../collections/collections.cuh"

#define CUDACHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

namespace rafa {
    namespace skeletons {

    template <typename T, typename Func, typename... Args>
    __global__ void mapKernel(T* d_array, int size, Func func, Args... args) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            printf("func: %p\n", func);
            printf("d_array[%d]: %i\n", idx, d_array[idx]);
            d_array[idx] = func(d_array[idx], args...);
            printf("d_array[%d] after: %i\n", idx, d_array[idx]);
        }
    }

    template <typename T>
    __global__ void device_print(T* d_array, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            printf("output[%d]: %i\n", idx, d_array[idx]);
        }
    }

    template <typename T, typename Func, typename... Args>
    __global__ void mapKernel2inputs(T* d_array, T* d_array2, int size, Func func, Args... args) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            d_array[idx] = func(d_array[idx], d_array2[idx], args...);
            printf("d_array[%d]: %i\n", idx, d_array[idx]);
        }
    }

    template <typename T, typename Func, typename... Args>
    __global__ void mapKernel2inputsOut(const T* input1, const T* input2, size_t size, Func func, T* output, Args... args) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = func(input1[idx], input2[idx], args...); 
            printf("output[%d]: %i\n", idx, output[idx]);
        }
    }

    template <VectorLike Container, typename Func, typename... Args>
    void map_impl(Container& container, Func func, Args... args) {
        using T = typename Container::value_type;
        size_t size = container.size();
        size_t bytes = size * sizeof(T);

        T* d_array;

        cudaStream_t stream;
        CUDACHECK(cudaStreamCreate(&stream));

        if constexpr (is_rafa_vector<Container>::value) {
            container.sync_host_to_device();
            d_array = container.device_data;
        } else {
            cudaHostAlloc(&d_array, bytes);
            cudaMemcpyAsync(d_array, container.data(), bytes, cudaMemcpyHostToDevice, stream);
        }

        int blockSize = 1024;
        int numBlocks = (size + blockSize - 1) / blockSize;

        mapKernel<<<numBlocks, blockSize, 0, stream>>>(d_array, size, func, args...);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        }

        CUDACHECK(cudaMemcpyAsync(container.data(), d_array, bytes, cudaMemcpyDeviceToHost, stream));

        CUDACHECK(cudaStreamSynchronize(stream));
        CUDACHECK(cudaStreamDestroy(stream));

        //CUDACHECK(cudaFree(d_array));
    }

    template <VectorLike Container, typename Func, typename... Args>
    void map_impl(Container& input, Func func, Container& output, Args... args) {
        /* using T = typename Container::value_type;
        size_t size = input.size();
        std::cout << "input size: " << size << std::endl;
        size_t bytes = size * sizeof(T);

        T* d_input; 

        cudaStream_t stream;
        CUDACHECK(cudaStreamCreate(&stream));

        if constexpr (is_rafa_vector<Container>::value) {
            std::cout << "rafa vector detected\n";
            input.sync_host_to_device();
            d_input = input.device_data;
        } else {
            cudaMalloc(&d_input, bytes);
            cudaMemcpyAsync(d_input, input.data(), bytes, cudaMemcpyHostToDevice, stream);
        }

        int blockSize = 1024;
        int numBlocks = (size + blockSize - 1) / blockSize;

        std::cout << "input size: " << size << " true size: " << input.size() << std::endl;
        std::cout << "output size: " << output.size() << " true size: " << output.size() << std::endl;
        mapKernel<<<numBlocks, blockSize, 0, stream>>>(d_input, size, func, args...);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        }
        
        CUDACHECK(cudaStreamSynchronize(stream));
        if constexpr (is_rafa_vector<Container>::value) {
            output = input;
        } else {
            cudaMemcpyAsync(output.data(), d_input, bytes, cudaMemcpyDeviceToHost, stream);
            cudaFree(d_input);
        }
    

        CUDACHECK(cudaStreamDestroy(stream));
        //CUDACHECK(cudaFree(d_array)); */

        using T = typename Container::value_type;
        size_t size = input.size();
        size_t bytes = size * sizeof(T);

        T* d_array = output.device_data;
        if (d_array == nullptr) {
            cudaHostAlloc(reinterpret_cast<void**>(&d_array), bytes, cudaHostAllocDefault);
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMemcpyAsync(d_array, input.data(), bytes, cudaMemcpyHostToDevice, stream);

        int blockSize = 1024;
        int numBlocks = (size + blockSize - 1) / blockSize;

        mapKernel<<<numBlocks, blockSize, 0, stream>>>(d_array, size, func, args...);
        
        std::cout << "cudaStreamSynchronize begins" << std::endl;
        cudaStreamSynchronize(stream);
        std::cout << "cudaStreamSynchronize ends" << std::endl;
        cudaMemcpy(output.data(), d_array, bytes, cudaMemcpyDeviceToHost);
        std::cout << "device data: " << d_array << std::endl;
        std::cout << "output data: " << output.device_data << std::endl;
        device_print<<<numBlocks, blockSize, 0>>>(output.device_data, size);

        cudaStreamDestroy(stream);
        //cudaFree(d_array);

    }

    

    template <VectorLike Container, typename Func, typename... Args>
    void map_impl(Container& input1, Container& input2, Func func, Args... args) {
        using T = typename Container::value_type;
        size_t size = input1.size();
        size_t bytes = size * sizeof(T);

        T* d_array;
        T* d_array2;
        cudaMalloc(&d_array, bytes);
        cudaMalloc(&d_array2, bytes);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMemcpyAsync(d_array, input1.data(), bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_array2, input2.data(), bytes, cudaMemcpyHostToDevice, stream);

        int blockSize = 1024;
        int numBlocks = (size + blockSize - 1) / blockSize;

        mapKernel2inputs<<<numBlocks, blockSize, 0, stream>>>(d_array, d_array2, size, func, args...);

        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(input1.data(), d_array, bytes, cudaMemcpyDeviceToHost, stream);

        cudaStreamDestroy(stream);

        cudaFree(d_array);
        cudaFree(d_array2);
    }

    template <VectorLike Container, typename Func, typename... Args>
    void map_impl(Container& input1, Container& input2, Func func, Container& output, Args... args) {
        using T = typename Container::value_type;
        size_t size = input1.size();
        size_t bytes = size * sizeof(T);
        std::cout << "2 inputs 1 output" << std::endl;
        cudaStream_t stream;
        CUDACHECK(cudaStreamCreate(&stream));
        // Ensure device data is allocated BEFORE accessing
        if (input1.device_data == nullptr) {
            std::cout << "Allocating device memory for input1 in map_impl" << std::endl;
            CUDACHECK(cudaMalloc(&input1.device_data, bytes));
            CUDACHECK(cudaMemcpyAsync(input1.device_data, input1.data(), bytes, cudaMemcpyHostToDevice, stream));
        }
        T* d_array = input1.device_data;
    
        if (input2.device_data == nullptr) {
            std::cout << "Allocating device memory for input2 in map_impl" << std::endl;
            CUDACHECK(cudaMalloc(&input2.device_data, bytes));
            CUDACHECK(cudaMemcpyAsync(input2.device_data, input2.data(), bytes, cudaMemcpyHostToDevice, stream));
        }
        T* d_array2 = input2.device_data;
    
        if (output.device_data == nullptr) {
            std::cout << "Allocating device memory for output in map_impl" << std::endl;
            CUDACHECK(cudaMalloc(&output.device_data, bytes));
        }
        T* d_output = output.device_data;
    
        int blockSize = 1024;
        int numBlocks = (size + blockSize - 1) / blockSize;
    
        
    
        // The initial copies might be redundant if sync_host_to_device was called
        // CUDACHECK(cudaMemcpyAsync(d_array, input1.data(), bytes, cudaMemcpyHostToDevice, stream));
        // CUDACHECK(cudaMemcpyAsync(d_array2, input2.data(), bytes, cudaMemcpyHostToDevice, stream));
    
        device_print<<<numBlocks, blockSize, 0>>>(d_array, size);
        device_print<<<numBlocks, blockSize, 0>>>(d_array2, size);
        device_print<<<numBlocks, blockSize, 0>>>(d_output, size);
    
        mapKernel2inputsOut<<<numBlocks, blockSize, 0, stream>>>(d_array, d_array2, size, func, d_output, args...);
    
        CUDACHECK(cudaStreamSynchronize(stream));
        CUDACHECK(cudaMemcpyAsync(output.data(), d_output, bytes, cudaMemcpyDeviceToHost, stream));
    
        CUDACHECK(cudaStreamDestroy(stream));
    
        // You might want to handle freeing device memory in the rafa::vector destructor or a specific cleanup function
        // cudaFree(d_array);
        // cudaFree(d_array2);
    }


    template <VectorLike Container, typename Func, typename... Args>
    auto map(Container& container, Func func, Args... args) {
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container.sync_host_to_device();
        }
        return map_impl(container, func, args...);
    }

    template <VectorLike Container, typename Func, typename... Args>
    auto map(Container& container, Func func, Container& output, Args... args) {
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container.sync_host_to_device();
        }
        return map_impl(container, func, output, args...); 
    }

    template <VectorLike Container, typename Func, typename... Args>
    auto map(Container& container1, Container& container2, Func func, Args... args) {
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container1.sync_host_to_device();
            container2.sync_host_to_device();
        }
        return map_impl(container1,container2, func, args...);
    }

    template <VectorLike Container, typename Func, typename... Args>
    auto map(Container& container1,Container& container2, Func func, Container& output, Args... args) {
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container1.sync_host_to_device();
            container2.sync_host_to_device();
        }
        return map_impl(container1,container2, func, output, args...); 
    }
    
}
    template <typename T>
    template <typename Func, typename... Args>
    auto rafa::vector<T>::map(Func kernel, Args&&... args) {
        sync_host_to_device();
        return rafa::skeletons::map(*this, kernel, std::forward<Args>(args)...);
    }

}