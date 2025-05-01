#ifndef SKEL_OBJ_CUH
#define SKEL_OBJ_CUH

#include <iostream>
#include <string>
#include <vector>
#include "../skeletons/map_logic.cuh"
#include "iskeleton_obj.cuh"

namespace rafa {

    template <typename Function, typename Container, typename... Args>
    class SkeletonObject : public ISkeletonObject {
    public:
        std::string skeletonType;
        std::vector<Container*> inputs;
        Container* output;
        Function kernel;
        std::tuple<Args...> extraArgs;

        //output, no args
        SkeletonObject(const std::string& skeletonType,
            std::vector<Container*> inputs,
            const Function& kernel,
            Container& output)
        : skeletonType(skeletonType),
        inputs(inputs),
        output(&output),
        kernel(kernel),
        extraArgs() {}


        //no output, no args
        SkeletonObject(const std::string& skeletonType,
            std::vector<Container*> inputs,
            const Function& kernel)
        : skeletonType(skeletonType),
        inputs(inputs),
        output(nullptr),
        kernel(kernel),
        extraArgs() {}
        
        
        //output, args
        SkeletonObject(const std::string& skeletonType,
            std::vector<Container*>& inputs,
            const Function& kernel,
            Container& output,
            const std::tuple<Args...>& args)
        : skeletonType(skeletonType),
        inputs(inputs),
        output(&output),
        kernel(kernel),
        extraArgs(args) {}
        
        
        //no output, args
        SkeletonObject(const std::string& skeletonType,
            std::vector<Container*>& inputs,
            const Function& kernel,
            const std::tuple<Args...>& args)
        : skeletonType(skeletonType),
        inputs(inputs),
        output(nullptr),
        kernel(kernel),
        extraArgs(args) {}

        void print() const {
            std::cout << "\nSkeleton Type: " << skeletonType << "\n";
            std::cout << "Inputs sizes: ";
            for (const auto& input : inputs) {
            std::cout << input->size() << " ";
            }
            std::cout << "\n";
            if (output) {
            std::cout << "Output sizes: " << output->size() << "\n";
            } else {
            std::cout << "Output: nullptr" << "\n";
            }
            std::cout << "Kernel: " << typeid(kernel).name() << "\n";
            std::cout << "Extra Args: ";
            std::apply([](const auto&... args) {
            ((std::cout << args << " "), ...);
            }, extraArgs);
            std::cout << "\n\n";
        }
        
        
        // Getters

        std::string getSkeletonType() const override { return skeletonType; }
        std::vector<Container*>& getInputs() const { return inputs; }
        /* std::string getSkeletonType() const override { return skeletonType;}        
        std::vector<Container*>& getInputs() const override { return inputs; }
        Container* getOutput() const override { return output; }
        Function& getKernel() const override { return kernel; }
        std::tuple<Args...>& getExtraArgs() const override { return extraArgs; } */

        /* std::any getAnyInputs() const override { return inputs; }
        std::any getAnyOutput() const override { return output; }
        std::any getAnyFunction() const override { return kernel; }
        std::any getAnyExtraArgs() const override { return extraArgs; } */
        

        // Setters
        void setType(const std::string& type) { skeletonType = type; }
        void setInputs(std::vector<Container*>& in) { inputs = in; }
        void setOutput(Container& out) { output = &out; }
        void setKernel(const Function& k) { kernel = k; }
        void setExtraArgs(const std::tuple<Args...>& args) { extraArgs = args; }
            

        void overrideDeviceInput(void* device_ptr) override {
            
            using T = typename Container::value_type;
    
            T* typed_device_ptr = static_cast<T*>(device_ptr);
    
            inputs[0]->set_device_data(typed_device_ptr);
            
            std::cout << "Overriding device input with pointer: " << device_ptr << std::endl;
        }

        void* getDeviceOutputPtr() const override {
            std::cout << "Calling getDeviceOutputPtr. Output is: " << output << std::endl;
            if (!output) {
                std::cerr << "Output is nullptr! Using first Input\n";
                return inputs[0]->device_data;
            }
            std::cout << "Device data pointer: " << output->device_data << std::endl;
            return static_cast<void*>(output->device_data);
        }

        // Modify to directly use Container's device pointer handling methods
        void setDevicePointer(void* ptr) {
            // Set device pointer on the container itself (if applicable)
            if (output) {
                output->setDevicePointer(ptr);
            }
        }

        void* getDevicePointer() const {
            return output ? output->getDevicePointer() : nullptr; // Return the device pointer of output
        }

        template <bool sync_host, bool sync_device>
        void dispatch_unary(std::vector<Container*>& inputs,
                            const Function& kernel,
                            Container* output,
                            const std::tuple<Args...>& extraArgs) {
            if constexpr (sizeof...(Args) == 0) {
                if (output)
                    map_logic<sync_host, sync_device>(*inputs[0], kernel, *output);
                else
                    map_logic<sync_host, sync_device>(*inputs[0], kernel);
            } else {
                if (output)
                    map_logic<sync_host, sync_device>(*inputs[0], kernel, *output, extraArgs);
                else
                    map_logic<sync_host, sync_device>(*inputs[0], kernel, extraArgs);
            }
        }

        template <bool sync_host, bool sync_device>
        void dispatch_binary(std::vector<Container*>& inputs,
                            const Function& kernel,
                            Container* output,
                            const std::tuple<Args...>& extraArgs) {
            if constexpr (sizeof...(Args) == 0) {
                if (output)
                    map_logic<sync_host, sync_device, Container, Function, Args...>(*inputs[0], *inputs[1], kernel, *output);
                else
                    map_logic<sync_host, sync_device, Container, Function, Args...>(*inputs[0], *inputs[1], kernel);
            } else {
                if (output)
                    map_logic<sync_host, sync_device, Container, Function, Args...>(*inputs[0], *inputs[1], kernel, *output, extraArgs);
                else
                    map_logic<sync_host, sync_device, Container, Function, Args...>(*inputs[0], *inputs[1], kernel, extraArgs);
            }
        }

        void executeSyncAll() {
            execute<true, true>();
        }

        void executeAsyncAll() {
            execute<false, false>();
        }

        void executeSyncHost() {
            execute<true, false>();
        }

        void executeSyncDevice() {
            execute<false, true>();
        }

        template <bool sync_host, bool sync_device>
        void execute() {
            using namespace rafa;
            int n_inputs = inputs.size();
            std::cout << "Number of inputs: " << n_inputs << std::endl;
            std::cout << "Output pointer is " << output << std::endl;
            std::cout << "size of args: " << sizeof...(Args) << std::endl;
            using T = typename Container::value_type;
        
            if (n_inputs == 1) {
                if constexpr (is_unary_kernel<Function, T>::value) {
                    dispatch_unary<sync_host, sync_device>(inputs, kernel, output, extraArgs);
                } else {
                    throw std::runtime_error("Kernel is not unary, but only one input provided");
                }
            } else if (n_inputs == 2) {
                if constexpr (is_binary_kernel<Function, T>::value) {
                    dispatch_binary<sync_host, sync_device>(inputs, kernel, output, extraArgs);
                } else {
                    throw std::runtime_error("Kernel is not binary, but two inputs provided");
                }
            } else {
                throw std::runtime_error("Invalid number of inputs");
            }
        }
        
            
        
        template<typename F, typename T>
        using is_unary_kernel = std::is_invocable<F, T>;
    
        template<typename F, typename T>
        using is_binary_kernel = std::is_invocable<F, T, T>;
        
    };



}
    

#endif