#ifndef MAP_DISPATCH_CUH
#define MAP_DISPATCH_CUH

#include "../types/skel_obj.cuh"

namespace rafa
{
    
/* 
    // Single input
    template <typename T>
    template <typename Func, typename... Args>
    rafa::vector<T> rafa::vector<T>::map_dispatch(Func kernel, Args... args) {
        auto obj = new rafa::SkeletonObject<Func, rafa::vector<T>>(
            "Map", std::vector<rafa::vector<T>*>{static_cast<rafa::vector<T>*>(this)}, kernel, args
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
    template <typename T>
    template <typename Func, VectorLike Container, typename... Args>
    rafa::vector<T> rafa::vector<T>::map_dispatch(Func kernel, Container& output, Args... args) {
        auto obj = new rafa::SkeletonObject<Func, Container>(
            "Map", std::vector<Container*>{static_cast<Container*>(this)}, kernel, output, args
        );
        std::cout << "Created SkeletonObject with skeletonType: " << obj->skeletonType << std::endl;
        this->skel_queue.push_back(obj);

        return *this;
    }

    // Two inputs
    template <typename T>
    template <typename Func, VectorLike Container, typename... Args>
    rafa::vector<T> rafa::vector<T>::map_dispatch(Container& input2, Func kernel, Args... args) {
        auto obj = new rafa::SkeletonObject<Func, Container>(
            "Map", std::vector<Container*>{static_cast<Container*>(this), &input2}, kernel, args
        );
        std::cout << "Created SkeletonObject with skeletonType: " << obj->skeletonType << std::endl;
        this->skel_queue.push_back(obj);
        return *this;
    }

    // Two inputs + output
    template <typename T>
    template <typename Func, VectorLike Container, typename... Args>
    rafa::vector<T> rafa::vector<T>::map_dispatch(Container& input2, Func kernel, Container& output, Args... args) {
        auto obj = new rafa::SkeletonObject<Func, Container>(
            "Map", std::vector<Container*>{static_cast<Container*>(this), &input2}, kernel, output, args
        );
        std::cout << "Created SkeletonObject with skeletonType: " << obj->skeletonType << std::endl;
        this->skel_queue.push_back(obj);
        return *this;
    }

    template <typename T>
    template <typename Func, typename... Args>
    rafa::vector<T> rafa::vector<T>::smart_map(Func kernel, Args... args) {
        return map_dispatch(kernel, args...);
    }

    template <typename T>
    template <typename Func, VectorLike Container, typename... Args>
    rafa::vector<T> rafa::vector<T>::smart_map(Func kernel,Container output ,Args... args) {
        return map_dispatch(kernel, output, args...);
    }

    template <typename T>
    template <typename Func, VectorLike Container, typename... Args>
    rafa::vector<T> rafa::vector<T>::smart_map(Container input2, Func kernel, Args... args) {
        return map_dispatch(input2, kernel, args...);
    }

    template <typename T>
    template <typename Func, VectorLike Container, typename... Args>
    rafa::vector<T> rafa::vector<T>::smart_map(Container input2, Func kernel, Container output, Args... args) {
        return map_dispatch(input2, kernel, output, args...);
    } */


}

#endif // MAP_DISPATCH_CUH