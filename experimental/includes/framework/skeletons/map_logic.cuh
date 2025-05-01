
#ifndef MAP_LOGIC_CUH
#define MAP_LOGIC_CUH
#include "../types/vector_like.cuh"
#include "../collections/vector.cuh"
#include "map_kernel.cuh"

namespace rafa {
   // namespace skeletons {


    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func>
    void map_impl(Container& container, Func func) {
        std::cout << "\nmap_impl with 1 input\n" << std::endl;
        using T = typename Container::value_type;
        size_t size = container.size();
        size_t bytes = size * sizeof(T);
   
        T* d_array;
        cudaStream_t stream;
   
        if constexpr (is_rafa_vector<Container>::value) {
            stream = container.stream;
        
            if constexpr (sync_device) {
                std::cout << "syncing host to device" << std::endl;
                container.sync_host_to_device();
            }
        
            d_array = container.device_data;
        
            if (d_array == nullptr) {
                std::cout << "Device data is null. Allocating..." << std::endl;
                cudaMalloc(&container.device_data, bytes);
                d_array = container.device_data;
            }
            else {
                std::cout << "Device pointer :" << d_array << std::endl;
            }
   
        } else {
                std::cout << "non-rafa vector, allocate device + pinned host copy" << std::endl;
                T* pinned_host = nullptr;
            
                cudaStreamCreate(&stream);
            
                cudaMalloc(&d_array, bytes);
            
                cudaHostAlloc(reinterpret_cast<void**>(&pinned_host), bytes, cudaHostAllocDefault);
            
                std::memcpy(pinned_host, container.data(), bytes);
            
                cudaMemcpyAsync(d_array, pinned_host, bytes, cudaMemcpyHostToDevice, stream);
            
                cudaStreamAddCallback(stream, [](cudaStream_t s, cudaError_t status, void* userData) {
                    cudaFreeHost(userData);
                }, pinned_host, 0);
       }
   
        // Kernel launch
        int blockSize = 1024;
        int numBlocks = (size + blockSize - 1) / blockSize;
        if (d_array == nullptr) {
            std::cerr << "ERROR: rafa::vector has no device_data allocated!" << std::endl;
        }

        
        mapKernel<<<numBlocks, blockSize, 0, stream>>>(d_array, size, func);
        CUDACHECK(cudaStreamSynchronize(stream));
        device_print<<<numBlocks, blockSize, 0, stream>>>(d_array, size);
        
   
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        }
   
        if constexpr (sync_host) {
            std::cout << "syncing device to host" << std::endl;
            if (container.host_pinned_data == nullptr || container.device_data == nullptr) {
                std::cerr << "ERROR: rafa::vector has no host_pinned_data or device_data allocated!" << std::endl;
            }
            container.sync_device_to_host();  // Only applies to rafa::vector
        } else {
            std::cout << "not syncing device to host" << std::endl;
        }
   
        if constexpr (!is_rafa_vector<Container>::value) {
            std::cout << "non-rafa cleanup" << std::endl;
            // Copy back to user memory
            cudaMemcpyAsync(container.data(), d_array, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cudaFree(d_array);
            cudaStreamDestroy(stream);
        }
        std::cout << "map_impl ends" << std::endl;
    }

    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_impl(Container& input, Func func, Container& output, Args... args) {
        std::cout << "\nmap_impl with 1 input + output\n" << std::endl;
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

    
    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_impl(Container& input1, Container& input2, Func func, Args... args) {
        std::cout << "\nmap_impl with 2 inputs\n" << std::endl;
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

    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_impl(Container& input1, Container& input2, Func func, Container& output, Args... args) {
        std::cout << "\nmap_impl with 2 inputs + output\n" << std::endl;
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
    
        //device_print<<<numBlocks, blockSize, 0>>>(d_array, size);
        //device_print<<<numBlocks, blockSize, 0>>>(d_array2, size);
        //device_print<<<numBlocks, blockSize, 0>>>(d_output, size);
    
        mapKernel2inputsOut<<<numBlocks, blockSize, 0, stream>>>(d_array, d_array2, size, func, d_output, args...);
    
        CUDACHECK(cudaStreamSynchronize(stream));
        CUDACHECK(cudaMemcpyAsync(output.data(), d_output, bytes, cudaMemcpyDeviceToHost, stream));
    
        CUDACHECK(cudaStreamDestroy(stream));
    
        // You might want to handle freeing device memory in the rafa::vector destructor or a specific cleanup function
        // cudaFree(d_array);
        // cudaFree(d_array2);
    }


    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_logic(Container& container, Func func, Args... args) {
        #pragma message("map with 1 input")
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container.sync_host_to_device();
        }
        return map_impl<sync_host,sync_device>(container, func, args...);
    }
 
    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_logic(Container& container, Func func, Container& output, Args... args) {
        #pragma message("map with output")
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container.sync_host_to_device();
        }
        return map_impl<sync_host,sync_device>(container, func, output, args...); 
    }

    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_logic(Container& container1, Container& container2, Func func, Args... args) {
        #pragma message("map with 2 inputs")
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container1.sync_host_to_device();
            container2.sync_host_to_device();
        }
        return map_impl<sync_host,sync_device>(container1,container2, func, args...);
    }

    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_logic(Container& container1,Container& container2, Func func, Container& output, Args... args) {
        #pragma message("map with 2 inputs and output")
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container1.sync_host_to_device();
            container2.sync_host_to_device();
        }
        return map_impl<sync_host,sync_device>(container1,container2, func, output, args...); 
    } 
    
    //}
} 

#endif // MAP_LOGIC_CUH