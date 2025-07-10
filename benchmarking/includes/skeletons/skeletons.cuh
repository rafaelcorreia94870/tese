#pragma once
#include <cuda_runtime.h>
#define M_HARRIS_REDUCE
#define NEW_FAST_REDUCE
#define FAST_REDUCE
#define OG_REDUCE
#include "map.cuh"
#include "reduce.cuh"
#include "reduce_fast.cuh"
#include "new_reduce.cuh"
#include "reduce_mharris.cuh"

