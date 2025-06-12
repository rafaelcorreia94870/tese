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
#include "iterator.cuh"

namespace rafa {

    template <typename T>
    class vector {
    public:
        T* host_pinned_data = nullptr;
        T* device_data;
        size_t vec_size;
        SkelQueue skel_queue;
        cudaStream_t stream;

        using value_type = T;
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        vector() {
            ////std::cout << "\nDEFAULT CONSTRUCTOR CALLED\n" << std::endl;
            CUDACHECK(cudaStreamCreate(&stream));
        }

        vector(size_t size) : vec_size(size) {
            ////std::cout << "\nCONSTRUCTOR WITH SIZE CALLED\n" << std::endl;
            CUDACHECK(cudaStreamCreate(&stream));
            alloc_host();
            alloc_device();
            sync_host_to_device();
        }

        vector(size_t size, const T& value) : vec_size(size) {
            ////std::cout << "\nCONSTRUCTOR WITH SIZE AND VALUE CALLED\n" << std::endl;
            CUDACHECK(cudaStreamCreate(&stream));
            alloc_host();
            alloc_device();
            std::fill(host_pinned_data, host_pinned_data + vec_size, value);
            sync_host_to_device();
        }

        // construct that accpets {}
        vector(std::initializer_list<T> init) : vec_size(init.size()) {
            ////std::cout << "\nCONSTRUCTOR WITH INITIALIZER LIST CALLED\n" << std::endl;
            CUDACHECK(cudaStreamCreate(&stream));
            alloc_host();
            alloc_device();
            std::copy(init.begin(), init.end(), host_pinned_data);
            sync_host_to_device();
        }

        vector(const vector& other) : vec_size(other.vec_size) {
            //std::cout << "\nCOPY CONSTRUCTOR CALLED\n" << std::endl;
            host_pinned_data = other.host_pinned_data;
            device_data = other.device_data;
            skel_queue = other.skel_queue;
            vec_size = other.vec_size;
            if (!other.host_pinned_data) {
                CUDACHECK(cudaHostAlloc(&host_pinned_data, vec_size * sizeof(T), cudaHostAllocDefault));
            } 
            if (!other.device_data) {
                CUDACHECK(cudaMalloc(&device_data, vec_size * sizeof(T)));
            }
            stream = other.stream;
            if (stream == nullptr) {
                CUDACHECK(cudaStreamCreate(&stream));
            }
        }

        ~vector() {
            if (host_pinned_data) {
                (cudaFreeHost(host_pinned_data));
                host_pinned_data = nullptr;
            }
            if (device_data) {
                (cudaFree(device_data));
                device_data = nullptr;
            }
            if (stream) {
                (cudaStreamDestroy(stream));
                stream = nullptr;
            }
            
            ////std::cout << "\nDESTRUCTOR CALLED\n" << std::endl;
            //if (host_pinned_data) cudaFreeHost(host_pinned_data);
            //if (device_data) cudaFree(device_data);
            //cudaStreamDestroy(stream);
        }

        void clear() {
            //std::cout << "\nCLEAR CALLED\n" << std::endl;
            if (host_pinned_data) {
                cudaFreeHost(host_pinned_data);
                host_pinned_data = nullptr;
            }
            if (device_data) {
                cudaFree(device_data);
                device_data = nullptr;
            }
            skel_queue.clear();
            if (stream) {
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
            vec_size = 0;
        }

        //setters of device_data
        void set_device_data(T* data, size_t size) {
            //std::cout << "\nSET DEVICE DATA CALLED\n" << std::endl;
            vec_size = size;
            device_data = static_cast<T*>(data);
        }

        void alloc_host() {
            if (host_pinned_data) cudaFreeHost(host_pinned_data);
            CUDACHECK(cudaHostAlloc(&host_pinned_data, vec_size * sizeof(T), cudaHostAllocDefault));
        }
    
        void alloc_device() {
            if (device_data) cudaFree(device_data);
            CUDACHECK(cudaMalloc(&device_data, vec_size * sizeof(T)));
        }

        rafa::iterator<T> begin() { return rafa::iterator<T>(host_pinned_data); }
        rafa::iterator<T> end()   { return rafa::iterator<T>(host_pinned_data + vec_size); }


        // Element access
        T& operator[](size_t i) { return host_pinned_data[i]; }
        const T& operator[](size_t i) const { return host_pinned_data[i]; }

        T* data() { return host_pinned_data; }
        const T* data() const { return host_pinned_data; }

        size_t size() const { return vec_size; }
        bool empty() const { return vec_size == 0; }

        void print() const {
            std::cout << "Vector contents: ";
            for (size_t i = 0; i < vec_size; ++i)
                std::cout << host_pinned_data[i] << " ";
            std::cout << std::endl;
        }

        void print(const size_t index) {
            if (index < vec_size)
                std::cout << "Element at index " << index << ": " << host_pinned_data[index] << std::endl;
            else
                std::cout << "Index out of bounds." << std::endl;
        }

        bool valid_index(size_t i) const { return i < vec_size; }

        void resize(size_t new_size) {
            vec_size = new_size;
            alloc_host();
            alloc_device();
        }
    


        void check_mem_bytes() {
            size_t size = 0;
            cudaPointerAttributes attributes;
            cudaPointerGetAttributes(&attributes, device_data);
            if (attributes.type == cudaMemoryTypeDevice) {
                cudaMemGetInfo(&size, nullptr);
                //std::cout << "Device memory size: " << size << " bytes" << std::endl;
            } else {
                //std::cout << "Pointer is not a device pointer." << std::endl;
            }
        }

        void sync_host_to_device() {
            if (!host_pinned_data || !device_data) return;
            CUDACHECK(cudaMemcpyAsync(device_data, host_pinned_data, vec_size * sizeof(T), cudaMemcpyHostToDevice, stream));
            CUDACHECK(cudaStreamSynchronize(stream));
        }
    
        void sync_device_to_host() {
            if (!host_pinned_data || !device_data) return;
            ////std::cout << "Syncing device to host" << std::endl;
            CUDACHECK(cudaMemcpyAsync(host_pinned_data, device_data, vec_size * sizeof(T), cudaMemcpyDeviceToHost, stream));
            CUDACHECK(cudaStreamSynchronize(stream));
            //std::cout << "Sync complete" << std::endl;
        }

        void simplify_skeletons() {
            auto& queue = skel_queue.execution_queue;
            auto it = queue.begin();
        
            
            //std::cout << "Queue size: " << queue.size() << std::endl;
        
            while (it != queue.end()) {
                auto next_it = std::next(it);
        
                
                if (next_it == queue.end()) {
                    //std::cerr << "Iterator out of range" << std::endl;
                    break;
                }
        
                // Check if the pointers are not null
                if (*it == nullptr || *next_it == nullptr) {
                    //std::cerr << "Null pointer in queue" << std::endl;
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
        
        template <typename Func>
        vector<T>& map_dispatch(Func kernel) {
            //std::cout << "Created SkeletonObject 1 In" << std::endl;
            std::vector<vector<T>*> input_vec{static_cast<vector<T>*>(this)};

            auto obj = new rafa::SkeletonObject<Func, vector<T>>(
                "Map", input_vec, kernel
            );
        
            //std::cout << "Created SkeletonObject with skeletonType: " << obj->skeletonType << std::endl;

            this->skel_queue.push_back(obj);
            //std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;

            //std::cout << "Pushed SkeletonObject to skel_queue" << std::endl;
            //std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;
            /* if (!this->skel_queue.empty()) {
                //std::cout << "skel_queue skeletonType of first element: " << this->skel_queue.front()->getSkeletonType() << std::endl;
            } else {
                //std::cout << "skel_queue is empty" << std::endl;
            } */
            return *this;
        }


        template <typename Func, typename... Args>
        vector<T>& map_dispatch(Func kernel, Args... args) {
            //std::cout << "Created SkeletonObject 1 In + args" << std::endl;
            rafa::SkeletonObject<Func, vector<T>>* obj;
            std::vector<vector<T>*> input_vec{static_cast<vector<T>*>(this)};
            if constexpr (sizeof...(args) > 0) {
                obj = new rafa::SkeletonObject<Func, vector<T>, Args...>(
                    "Map", input_vec, kernel, std::make_tuple(args...)
                );
            } else {
                obj = new rafa::SkeletonObject<Func, vector<T>>(
                    "Map", input_vec, kernel
                );
            }
            //std::cout << "Created SkeletonObject with skeletonType: " << obj->skeletonType << std::endl;

            this->skel_queue.push_back(obj);
            //std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;

            //std::cout << "Pushed SkeletonObject to skel_queue" << std::endl;
            //std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;
            if (!this->skel_queue.empty()) {
                //std::cout << "skel_queue skeletonType of first element: " << this->skel_queue.front()->getSkeletonType() << std::endl;
            } else {
                //std::cout << "skel_queue is empty" << std::endl;
            }
            return *this;
        }

        // Single input + output

        template <typename Func, VectorLike Container>
        vector<T>& map_dispatch(Func kernel, Container& output) {
            //std::cout << "Created SkeletonObject 1 In + out" << std::endl;
            
            auto obj = new rafa::SkeletonObject<Func, Container>(
                "Map", std::vector<Container*>{static_cast<Container*>(this)}, kernel, output
            );
            this->skel_queue.push_back(obj);
            //std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;
           
            return *this;
        }

        template <typename Func, VectorLike Container, typename... Args>
        vector<T>& map_dispatch(Func kernel, Container& output, Args... args) {
            //std::cout << "Created SkeletonObject 1 In + out + args" << std::endl;
            if constexpr (sizeof...(args) > 0) {
                auto obj = new rafa::SkeletonObject<Func, Container, Args...>(
                    "Map", std::vector<Container*>{static_cast<Container*>(this)}, kernel, output, args...
                );
                this->skel_queue.push_back(obj);
                //std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;
            } else {
                auto obj = new rafa::SkeletonObject<Func, Container>(
                    "Map", std::vector<Container*>{static_cast<Container*>(this)}, kernel, output
                );
                this->skel_queue.push_back(obj);

                //std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;
            }

            return *this;
        }


        // Two inputs

        template <typename Func, VectorLike Container, typename... Args>
        vector<T>& map_dispatch(Container& input2, Func kernel, Args... args) {
            //std::cout << "Created SkeletonObject 2 In" << std::endl;
            auto obj = new rafa::SkeletonObject<Func, Container, Args...>(
                "Map", std::vector<Container*>{static_cast<Container*>(this), &input2}, kernel, std::make_tuple(args...)
            );
            //std::cout << "Created SkeletonObject with skeletonType: " << obj->skeletonType << std::endl;
            this->skel_queue.push_back(obj);
            //std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;
            return *this;
        }

        template <typename Func, VectorLike Container, typename... Args>
        vector<T>& map_dispatch(Container& input2, Func kernel) {
            //std::cout << "Created SkeletonObject 2 In" << std::endl;
            auto obj = new rafa::SkeletonObject<Func, Container>(
                "Map", std::vector<Container*>{static_cast<Container*>(this), &input2}, kernel
            );
            //std::cout << "Created SkeletonObject with skeletonType: " << obj->skeletonType << std::endl;
            this->skel_queue.push_back(obj);
            //std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;
            return *this;
        }

        // Two inputs + output

        template <typename Func, VectorLike Container, typename... Args>
        vector<T>& map_dispatch(Container& input2, Func kernel, Container& output, Args... args) {
            //std::cout << "Created SkeletonObject 2 In + out" << std::endl;
            if constexpr (sizeof...(args) > 0) {
                auto obj = new rafa::SkeletonObject<Func, Container, Args...>(
                    "Map", std::vector<Container*>{static_cast<Container*>(this), &input2}, kernel, output, std::make_tuple(args...)
                );
                this->skel_queue.push_back(obj);
                //std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;
                
            } else {
                auto obj = new rafa::SkeletonObject<Func, Container>(
                    "Map", std::vector<Container*>{static_cast<Container*>(this), &input2}, kernel, output
                );
                 this->skel_queue.push_back(obj);
                 //std::cout << "skel_queue size: " << this->skel_queue.size() << std::endl;

            }

            return *this;
        }


        template <typename Func, typename... Args>
        vector<T>& smart_map(Func kernel, Args... args) {
            return map_dispatch<Func,Args...>(kernel, args...);
        }


        template <typename Func, VectorLike Container, typename... Args>
        vector<T>& smart_map(Func kernel,Container& output ,Args... args) {
            return map_dispatch<Func,Container, Args...>(kernel, output, args...);
        }


        template <typename Func, VectorLike Container, typename... Args>
        vector<T>& smart_map(Container& input2, Func kernel, Args... args) {
            return map_dispatch<Func, Container, Args...>(input2, kernel, args...);
        }


        template <typename Func, VectorLike Container, typename... Args>
        vector<T>& smart_map(Container& input2, Func kernel, Container& output, Args... args) {
            return map_dispatch<Func, Container, Args...>(input2, kernel, output, args...);
        }

        template <typename Func, typename... Args>
        auto reduce(Func kernel, Args&&... args);

        vector<T>& execute(){
            /* //std::cout << "\nBefore simplification:\n";
            for (auto it = skel_queue.execution_queue.begin(); it != skel_queue.execution_queue.end(); ++it) {
                (*it)->print();
            }
            simplify_skeletons();
            //std::cout << "\nAfter simplification:\n";
            for (auto it = skel_queue.execution_queue.begin(); it != skel_queue.execution_queue.end(); ++it) {
            (*it)->print();
            } */
            //simplify_skeletons();
            skel_queue.execute();
            skel_queue.clear();
            ////std::cout << "Execution queue after execution: " << skel_queue.size() << std::endl;

            return *this;
            
        }

      
    };

    template <typename T>
    struct is_rafa_vector : std::false_type {};

    template <typename T>
    struct is_rafa_vector<vector<T>> : std::true_type {};

} // namespace rafa

#endif // VECTOR_CUH
