
#ifndef MAP_LOGIC_CUH
#define MAP_LOGIC_CUH
#include "../types/vector_like.cuh"
#include "../collections/vector.cuh"
#include "map_kernel.cuh"

namespace rafa {
   // namespace skeletons {


    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func>
    void map_impl(Container& container, Func func) {
        //std::cout << "\nmap_impl with 1 input\n" << std::endl;
        using T = typename Container::value_type;
        size_t size = container.size();
        size_t bytes = size * sizeof(T);
   
        T* d_array;
        cudaStream_t stream;
   
        if constexpr (is_rafa_vector<Container>::value) {
            stream = container.stream;
        
            if constexpr (sync_device) {
                container.sync_host_to_device();
            }
        
            
            if (container.device_data == nullptr) {
                std::cout << "d_array is null, allocating device memory" << std::endl;
                CUDACHECK(cudaMalloc(&container.device_data, bytes));
            }
            d_array = container.device_data;

   
        } else {
                //std::cout << "non-rafa vector, allocate device + pinned host copy" << std::endl;
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
        //CUDACHECK(cudaPeekAtLastError());
        //device_print<<<numBlocks, blockSize, 0, stream>>>(d_array, size);
        if constexpr (sync_host) {
            //std::cout << "syncing device to host" << std::endl;
            if (container.host_pinned_data == nullptr || container.device_data == nullptr) {
                std::cerr << "ERROR: rafa::vector has no host_pinned_data or device_data allocated!" << std::endl;
            }
            container.sync_device_to_host();  // Only applies to rafa::vector
        } else {
            //std::cout << "not syncing device to host" << std::endl;
        }
   
        if constexpr (!is_rafa_vector<Container>::value) {
            cudaMemcpyAsync(container.data(), d_array, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cudaFree(d_array);
            cudaStreamDestroy(stream);
        }else{
            cudaStreamSynchronize(stream);
        }
        //std::cout << "container.device_data: ";
        ////device_print<<<numBlocks, blockSize, 0, stream>>>(container.device_data, size);
    }

    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_impl(Container& input, Func func, Container& output, Args... args) {
        //std::cout << "\nmap_impl with 1 input + output\n" << std::endl;
        using T = typename Container::value_type;
        size_t size = input.size();
        size_t bytes = size * sizeof(T);

        T* d_array;

        cudaStream_t stream;

        if constexpr (is_rafa_vector<Container>::value) {
            stream = input.stream;

            if constexpr (sync_device) {
                input.sync_host_to_device();
            }

            d_array = input.device_data;
            if (d_array == nullptr) {
                CUDACHECK(cudaMalloc(&input.device_data, bytes));
                d_array = input.device_data;
            }
        } else {
            cudaStreamCreate(&stream);
            if (d_array == nullptr) {
                CUDACHECK(cudaMalloc(d_array, bytes));
            }
            
            CUDACHECK(cudaMemcpyAsync(d_array, input.data(), bytes, cudaMemcpyHostToDevice, stream));
        }

        int blockSize = 1024;
        int numBlocks = (size + blockSize - 1) / blockSize;
        //std::cout << "d_array before: " << d_array << std::endl;
        //device_print<<<numBlocks, blockSize, 0, stream>>>(d_array, size);
        mapKernel<<<numBlocks, blockSize, 0, stream>>>(d_array, size, func, args...);
        CUDACHECK(cudaStreamSynchronize(stream));
        //CUDACHECK(cudaPeekAtLastError());
        //std::cout << "d_array after: " << d_array << std::endl;
        //device_print<<<numBlocks, blockSize, 0, stream>>>(d_array, size);


        if constexpr (!is_rafa_vector<Container>::value){
            cudaMemcpyAsync(output.data(), d_array, bytes, cudaMemcpyDeviceToHost,stream);
            cudaFree(d_array);
            cudaStreamDestroy(stream);
        } else {
            output.device_data = d_array;
            if constexpr (sync_host) {
                //std::cout << "syncing device to host" << std::endl;
                output.sync_device_to_host();
            }
            //std::cout << "output.device_data: " << output.device_data << std::endl;
            ////device_print<<<numBlocks, blockSize, 0, stream>>>(output.device_data, size);
            cudaStreamSynchronize(stream);

            
        }
    }

    
    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_impl(Container& input1, Container& input2, Func func, Args... args) {
        std::cout << "\nmap_impl with 2 inputs\n" << std::endl;
        using T = typename Container::value_type;
        size_t size = input1.size();
        size_t bytes = size * sizeof(T);

        T* d_array;
        T* d_array2;
        cudaStream_t stream;


        if constexpr (is_rafa_vector<Container>::value) {
            d_array = input1.device_data;
            d_array2 = input2.device_data;
            stream = input1.stream;
            if constexpr (sync_device) {
                std::cout << "syncing device to host" << std::endl;
                input1.sync_host_to_device();
                input2.sync_host_to_device();
            }
            if (d_array == nullptr) {
                CUDACHECK(cudaMalloc(&input1.device_data, bytes));
                d_array = input1.device_data;
            }
            if (d_array2 == nullptr) {
                CUDACHECK(cudaMalloc(&input2.device_data, bytes));
                d_array2 = input2.device_data;
            }
        } else {
            cudaStreamCreate(&stream);
            CUDACHECK(cudaMalloc(&d_array, bytes));
            CUDACHECK(cudaMalloc(&d_array2, bytes));
            cudaMemcpyAsync(d_array, input1.data(), bytes, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_array2, input2.data(), bytes, cudaMemcpyHostToDevice, stream);   
        }

        int blockSize = 1024;
        int numBlocks = (size + blockSize - 1) / blockSize;

        mapKernel2inputs<<<numBlocks, blockSize, 0, stream>>>(d_array, d_array2, size, func, args...);
        CUDACHECK(cudaStreamSynchronize(stream));
        //CUDACHECK(cudaPeekAtLastError());

        if constexpr (sync_host) {
            input1.sync_device_to_host();
        }

        
        if constexpr (!is_rafa_vector<Container>::value) {
            cudaMemcpyAsync(input1.data(), d_array, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cudaFree(d_array);
            cudaFree(d_array2);
            cudaStreamDestroy(stream);
        } else {
            cudaStreamSynchronize(stream);
        }
    }

    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_impl(Container& input1, Container& input2, Func func, Container& output, Args... args) {
        std::cout << "\nmap_impl with 2 inputs + output\n" << std::endl;
        using T = typename Container::value_type;
        size_t size = input1.size();
        size_t bytes = size * sizeof(T);
        cudaStream_t stream;
        T* d_array, *d_array2, *d_output;

        std::cout << "input1.device_data: " << input1.device_data << std::endl;
        //device_print<<<1, 1>>>(input1.device_data, size);

        if constexpr (is_rafa_vector<Container>::value) {
            if constexpr (sync_device) {
                input1.sync_host_to_device();
                input2.sync_host_to_device();
            }
            stream = input1.stream;
            d_array = input1.device_data;
            d_array2 = input2.device_data;
            d_output = output.device_data;
            if (d_array == nullptr) {
                CUDACHECK(cudaMalloc(&input1.device_data, bytes));
            }
            if (d_array2 == nullptr) {
                CUDACHECK(cudaMalloc(&input2.device_data, bytes));
            }
            if (d_output == nullptr) {
                CUDACHECK(cudaMalloc(&output.device_data, bytes));
            }
        } else {
            cudaStreamCreate(&stream);
            cudaMalloc(&d_array, bytes);
            cudaMalloc(&d_array2, bytes);
            cudaMalloc(&d_output, bytes);
            cudaMemcpyAsync(d_array, input1.data(), bytes, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_array2, input2.data(), bytes, cudaMemcpyHostToDevice, stream);   
        }
    
        int blockSize = 1024;
        int numBlocks = (size + blockSize - 1) / blockSize;
        
        /* std::cout << "Device ptr (input): " << input1.device_data << " | Output: " << output.device_data << std::endl;
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, input1.device_data);
        std::cout << "Pointer type: " << attr.type << " | Device: " << attr.device << std::endl; */
        cudaStreamSynchronize(stream);
        std::cout << "d_array: " << d_array << std::endl;
        //device_print<<<numBlocks, blockSize, 0, stream>>>(d_array, size);
        cudaStreamSynchronize(stream);
        mapKernel2inputsOut<<<numBlocks, blockSize, 0, stream>>>(d_array, d_array2, size, func, d_output, args...);
        
        std::cout << "d_output: " << d_output << std::endl;
        //device_print<<<numBlocks, blockSize, 0, stream>>>(d_output, size);

        CUDACHECK(cudaStreamSynchronize(stream));
        //CUDACHECK(cudaPeekAtLastError());
        if constexpr (sync_host) {
            std::cout << "syncing device to host" << std::endl;
            output.sync_device_to_host();
            cudaStreamSynchronize(stream);
        }

        if constexpr (!is_rafa_vector<Container>::value) {
            CUDACHECK(cudaMemcpyAsync(output.data(), d_output, bytes, cudaMemcpyDeviceToHost, stream));
            CUDACHECK(cudaStreamSynchronize(stream));
            cudaFree(d_array);
            cudaFree(d_array2);
            cudaFree(d_output);
            CUDACHECK(cudaStreamDestroy(stream));
        } else {
            cudaStreamSynchronize(stream);
        }
    }


    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_logic(Container& container, Func func, Args... args) {
        /* #pragma message("map with 1 input")
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container.sync_host_to_device();
        } */
        return map_impl<sync_host,sync_device>(container, func, args...);
    }
 
    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_logic(Container& container, Func func, Container& output, Args... args) {
        /* #pragma message("map with output")
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container.sync_host_to_device();
        } */
        return map_impl<sync_host,sync_device>(container, func, output, args...); 
    }

    
    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_logic(Container& container1, Container& container2, Func func, Args... args) {
        /* #pragma message("map with 2 inputs")
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container1.sync_host_to_device();
            container2.sync_host_to_device();
        } */
        return map_impl<sync_host,sync_device>(container1,container2, func, args...);
    }

    template <bool sync_host, bool sync_device, VectorLike Container, typename Func, typename... Args>
    void map_logic(Container& container1,Container& container2, Func func, Container& output, Args... args) {
        /* #pragma message("map with 2 inputs and output")
        if constexpr(std::is_same_v<Container, rafa::vector<typename Container::value_type>>) {
            container1.sync_host_to_device();
            container2.sync_host_to_device();
        } */
        return map_impl<sync_host,sync_device>(container1,container2, func, output, args...); 
    } 
    
    //}
} 

#endif // MAP_LOGIC_CUH