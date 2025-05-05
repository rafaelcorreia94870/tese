#ifndef SKEL_QUEUE_CUH
#define SKEL_QUEUE_CUH

#include <vector>
#include <iostream>
#include "../types/vector_like.cuh"
#include "../types/iskeleton_obj.cuh"


namespace rafa {

    class SkelQueue {
        public:
            std::vector<ISkeletonObject*> execution_queue;

            void push_back(ISkeletonObject* op) {
                execution_queue.push_back(op);
            }
        
            void print() const {
                for (const auto& op : execution_queue) {
                    op->print();
                }
            }

            //begin and end iterators for the queue
            auto begin() {
                return execution_queue.begin();
            }
            auto end() {
                return execution_queue.end();
            }

            auto front() {
                return execution_queue.front();
            }

            auto empty() const {
                return execution_queue.empty();
            }

            auto erase(auto it) {
                return execution_queue.erase(it);
            }
            auto size() const {
                return execution_queue.size();
            }
            
            void clear() {
                execution_queue.clear();
            }

            void execute() {
                //std::cout << "Executing operations in SkelQueue..." << std::endl;
                bool first = true;
                bool last = false;
                void* chained_output = nullptr;

                std::cout << "Executing queue of size: " << execution_queue.size() << std::endl;
                std::cout << "Execution queue : \n";
                for (size_t i = 0; i < execution_queue.size(); ++i) {
                    auto& op = execution_queue[i];

                    std::cout << "Executing operation: ";
                    op->print();
                }
                

                for (size_t i = 0; i < execution_queue.size(); ++i) {
                    auto& op = execution_queue[i];
                    last = (i == execution_queue.size() - 1);

                    std::cout << "Executing operation: ";
                    op->print();
                    std::cout << "input before: ";
                    op->printInput();
                
                    if (chained_output != nullptr) {
                        std::cout << "OVERRIDING DEVICE INPUT " << i << std::endl;
                        std::cout << "chained_output: " << chained_output << std::endl;

                        op->overrideDeviceInput(chained_output);
                    }
                
                    if (first && last) {
                        op->executeSyncAll();
                    } else if (first) {
                        std::cout << "Executing first operation: " << op->getSkeletonType() << std::endl;
                        op->executeSyncDevice();
                        first = false;
                    } else if (last) {
                        std::cout << "Executing last operation: " << op->getSkeletonType() << std::endl;
                        op->executeSyncHost();
                    } else {
                        std::cout << "Executing operation: " << op->getSkeletonType() << std::endl;
                        op->executeAsyncAll();
                    }
                    std::cout << "input after: ";
                    op->printInput();

                
                    // Grab output from this op for chaining
                    //std::cout << "getting new input" << std::endl;
                    //cast chained_output to the correct type
                    
                    chained_output = op->getDeviceOutputPtr();
                    //std::cout << "chained_output: " << chained_output << std::endl;
                }
                //std::cout << "Finished executing operations in SkelQueue." << std::endl;

            }
            
        };

} 

#endif