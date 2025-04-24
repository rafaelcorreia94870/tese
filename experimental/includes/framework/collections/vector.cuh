#ifndef VECTOR_CUH
#define VECTOR_CUH
#pragma message("vector.cuh included")

#include <vector>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <iostream>
#include <cuda_runtime.h>
#include <type_traits>
#include <algorithm>
#include <memory>
#include "../types/vector_like.cuh"
#include "skel_queue.cuh"
#include "../types/skel_obj.cuh"

namespace rafa {

    template <typename T>
    class vector {
    public:
        std::vector<T> host_data;
        T* device_data;
        size_t vec_size;
        SkelQueue skel_queue;

        using value_type = T;
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        vector() : device_data(nullptr), vec_size(0) {}

        vector(size_t size, T* device_data) : vec_size(size), device_data(device_data) {
            host_data.resize(size);
            cudaHostAlloc(reinterpret_cast<void**>(&device_data), size * sizeof(T), cudaHostAllocDefault);
        }

        vector(size_t size) : vec_size(size), host_data(size), device_data(nullptr) {
            host_data.resize(size);
            cudaHostAlloc(reinterpret_cast<void**>(&device_data), size * sizeof(T), cudaHostAllocDefault);
        }

        vector(size_t size, const T& value, cudaStream_t stream = 0) : vec_size(size), host_data(size, value), device_data(nullptr) {
            cudaHostAlloc(reinterpret_cast<void**>(&device_data), size * sizeof(T), cudaHostAllocDefault);
            host_data.resize(size, value);
            cudaStreamCreate(&stream);
            cudaMemcpyAsync(device_data, host_data.data(), size * sizeof(T), cudaMemcpyHostToDevice, stream);
            cudaStreamDestroy(stream);
        }

        ~vector() {
            if (device_data) {
                cudaFree(device_data);
            }
        }

        size_t size() const { return vec_size; }

        iterator begin() { return host_data.begin(); }
        const_iterator begin() const { return host_data.begin(); }

        iterator end() { return host_data.end(); }
        const_iterator end() const { return host_data.end(); }

        T& operator[](size_t i) { return host_data[i]; }
        const T& operator[](size_t i) const { return host_data[i]; }

        T* data() { return host_data.data(); }
        const T* data() const { return host_data.data(); }

        void push_back(const T& value) {
            host_data.push_back(value);
            vec_size = host_data.size();
        }

        bool empty() const { return host_data.empty(); }

        void swap(vector& other) {
            host_data.swap(other.host_data);
            std::swap(device_data, other.device_data);
            std::swap(vec_size, other.vec_size);
        }

        bool valid_index(size_t i) const { return i < vec_size; }

        void resize(size_t new_size) {
            if (new_size != vec_size) {
                host_data.resize(new_size);
                vec_size = new_size;
                if (device_data) {
                    cudaFree(device_data);
                    device_data = nullptr;
                }
            }
        }

        void malloc_device(cudaStream_t stream) {
            if (device_data) {
                cudaFree(device_data);
            }
            cudaHostAlloc(&device_data, vec_size * sizeof(T));
        }

        void print() {
            for (size_t i = 0; i < host_data.size(); ++i) {
                std::cout << host_data[i] << " ";
            }
            std::cout << std::endl;
        }

        void print(const size_t index) {
            if (index < vec_size) {
                std::cout << "Element at index " << index << ": " << host_data[index] << std::endl;
            } else {
                std::cout << "Index out of bounds." << std::endl;
            }
        }

        void check_mem_bytes() {
            size_t size = 0;
            cudaPointerAttributes attributes;
            cudaPointerGetAttributes(&attributes, device_data);
            if (attributes.type == cudaMemoryTypeDevice) {
                cudaMemGetInfo(&size, nullptr);
                std::cout << "Device memory size: " << size << " bytes" << std::endl;
            } else {
                std::cout << "Pointer is not a device pointer." << std::endl;
            }
        }

        void sync_host_to_device(cudaStream_t stream = 0) {
            if (device_data == nullptr){
                cudaHostAlloc(reinterpret_cast<void**>(&device_data), vec_size * sizeof(T), cudaHostAllocDefault);
            }
            if (host_data.empty()) return;
            cudaMemcpyAsync(device_data, host_data.data(), vec_size * sizeof(T), cudaMemcpyHostToDevice, stream);
        }

        void sync_device_to_host(cudaStream_t stream = 0) {
            if (!device_data) return;
            cudaMemcpyAsync(host_data.data(), device_data, vec_size * sizeof(T), cudaMemcpyDeviceToHost, stream);
        }

        void simplify_skeletons() {
            auto& queue = skel_queue.execution_queue;
            auto it = queue.begin();
        
            
            std::cout << "Queue size: " << queue.size() << std::endl;
        
            while (it != queue.end()) {
                auto next_it = std::next(it);
        
                
                if (next_it == queue.end()) {
                    std::cerr << "Iterator out of range" << std::endl;
                    break;
                }
        
                // Check if the pointers are not null
                if (*it == nullptr || *next_it == nullptr) {
                    std::cerr << "Null pointer in queue" << std::endl;
                    ++it;
                    continue;
                }
        
            
        
                if ((*it)->getSkeletonType() == "Map" && (*next_it)->getSkeletonType() == "Map") {
                    //merge maps
                    /* auto merged_kernel = [this, it, next_it](const vector<T>& input, vector<T>& output) {
                        vector<T> temp(input.size());
                        (*it)->getKernel()(input, temp);
                        (*next_it)->getKernel()(temp, output);
                    }; */

                }
                else if ((*it)->getSkeletonType() == "Map" && (*next_it)->getSkeletonType() == "Reduce") {
                    //merge map and reduce
                }
                
        
                it = next_it;
            }
        }
        
        
        


        // Single input
        
        template <typename Func, typename... Args>
        vector<T> map_dispatch(Func kernel, Args... args) {
            auto obj = new rafa::SkeletonObject<Func, vector<T>>(
                "Map", std::vector<vector<T>*>{static_cast<vector<T>*>(this)}, kernel, args...
            );
            std::cout << "Created SkeletonObject with skeletonType: " << obj->skeletonType << std::endl;

            this->skel_queue.push_back(obj);

            std::cout << "Pushed SkeletonObject to skel_queue" << std::endl;
            std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;
            if (!this->skel_queue.empty()) {
                std::cout << "skel_queue skeletonType of first element: " << this->skel_queue.front()->getSkeletonType() << std::endl;
            } else {
                std::cout << "skel_queue is empty" << std::endl;
            }
            return *this;
        }

        // Single input + output

        template <typename Func, VectorLike Container, typename... Args>
        vector<T> map_dispatch(Func kernel, Container& output, Args... args) {
            auto obj = new rafa::SkeletonObject<Func, Container>(
                "Map", std::vector<Container*>{static_cast<Container*>(this)}, kernel, output, args...
            );
            std::cout << "Created SkeletonObject with skeletonType: " << obj->skeletonType << std::endl;
            this->skel_queue.push_back(obj);

            return *this;
        }

        // Two inputs

        template <typename Func, VectorLike Container, typename... Args>
        vector<T> map_dispatch(Container& input2, Func kernel, Args... args) {
            auto obj = new rafa::SkeletonObject<Func, Container>(
                "Map", std::vector<Container*>{static_cast<Container*>(this), &input2}, kernel, args...
            );
            std::cout << "Created SkeletonObject with skeletonType: " << obj->skeletonType << std::endl;
            this->skel_queue.push_back(obj);
            return *this;
        }

        // Two inputs + output

        template <typename Func, VectorLike Container, typename... Args>
        vector<T> map_dispatch(Container& input2, Func kernel, Container& output, Args... args) {
            auto obj = new rafa::SkeletonObject<Func, Container>(
                "Map", std::vector<Container*>{static_cast<Container*>(this), &input2}, kernel, output, args...
            );
            std::cout << "Created SkeletonObject with skeletonType: " << obj->skeletonType << std::endl;
            this->skel_queue.push_back(obj);
            return *this;
        }


        template <typename Func, typename... Args>
        vector<T> smart_map(Func kernel, Args... args) {
            return map_dispatch(kernel, args...);
        }


        template <typename Func, VectorLike Container, typename... Args>
        vector<T> smart_map(Func kernel,Container output ,Args... args) {
            return map_dispatch(kernel, output, args...);
        }


        template <typename Func, VectorLike Container, typename... Args>
        vector<T> smart_map(Container input2, Func kernel, Args... args) {
            return map_dispatch(input2, kernel, args...);
        }


        template <typename Func, VectorLike Container, typename... Args>
        vector<T> smart_map(Container input2, Func kernel, Container output, Args... args) {
            return map_dispatch(input2, kernel, output, args...);
        }

        template <typename Func, typename... Args>
        auto reduce(Func kernel, Args&&... args);

        vector<T> execute(){
            /* std::cout << "\nBefore simplification:\n";
            for (auto it = skel_queue.execution_queue.begin(); it != skel_queue.execution_queue.end(); ++it) {
                (*it)->print();
            }
            simplify_skeletons();
            std::cout << "\nAfter simplification:\n";
            for (auto it = skel_queue.execution_queue.begin(); it != skel_queue.execution_queue.end(); ++it) {
            (*it)->print();
            } */
            //simplify_skeletons();
            skel_queue.execute();

            return *this;
            
        }

      
    };

    template <typename T>
    struct is_rafa_vector : std::false_type {};

    template <typename T>
    struct is_rafa_vector<vector<T>> : std::true_type {};

} // namespace rafa

#endif // VECTOR_CUH
