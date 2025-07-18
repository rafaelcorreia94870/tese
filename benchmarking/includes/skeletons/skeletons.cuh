#pragma once
#include <cuda_runtime.h>
#define M_HARRIS_REDUCE

#define REDUCE_V2
#define REDUCE_V3
#define REDUCE_V4

#define OG_REDUCE
#include "map.cuh"
#include "reduce.cuh"
#include "reduce_v2.cuh"
#include "reduce_v3.cuh"
#include "reduce_v4.cuh"
#include "reduce_mharris.cuh"
