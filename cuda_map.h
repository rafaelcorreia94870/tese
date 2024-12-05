

#ifndef CUDA_MAP_H
#define CUDA_MAP_H

#include <iostream>
#include <vector>
#include <list>
#include <array>
#include <map>
#include <deque>
#include <iterator> 
#include <cuda_runtime.h>
#include <type_traits>

#include <vector>

// Function declaration for CUDA map
template <typename Iterator, typename Func>
void map(Iterator& container, Func& func);

#endif // CUDA_MAP_H
