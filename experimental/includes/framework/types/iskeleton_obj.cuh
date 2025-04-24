#ifndef ISKEL_OBJ_CUH
#define ISKEL_OBJ_CUH

#include <string>

namespace rafa {
    class ISkeletonObject {
        public:
            virtual void print() const = 0;
            virtual std::string getSkeletonType() const = 0;
            virtual void execute() = 0;

            virtual ~ISkeletonObject() = default;
        };
        

}

#endif
