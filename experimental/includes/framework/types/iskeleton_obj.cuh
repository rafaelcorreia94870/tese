#ifndef ISKEL_OBJ_CUH
#define ISKEL_OBJ_CUH

#include <string>

namespace rafa {
    class ISkeletonObject {
        public:
            virtual void print() const = 0;
            virtual std::string getSkeletonType() const = 0;
            //virtual void execute(const bool = true) = 0;
            virtual void executeSyncAll() = 0;
            virtual void executeAsyncAll() = 0;
            virtual void executeSyncHost() = 0;
            virtual void executeSyncDevice() = 0;
            virtual void overrideDeviceInput(void* device_ptr) = 0;
            virtual void* getDeviceOutputPtr() const = 0;
            virtual void printInput() const = 0;


            virtual ~ISkeletonObject() = default;
        };
        

}

#endif
