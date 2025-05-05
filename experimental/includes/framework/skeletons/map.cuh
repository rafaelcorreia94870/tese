#ifndef MAP_CUH
#define MAP_CUH

#include <iostream>
#define CUDACHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        //exit(code);
    }
}
#include "map_logic.cuh"
//#include "map_dispatch.cuh"

#endif 