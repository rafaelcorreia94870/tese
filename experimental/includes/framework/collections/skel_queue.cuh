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
                for (auto& op : execution_queue) {
                    op->execute();
                }
            }
            
        };

} 

#endif