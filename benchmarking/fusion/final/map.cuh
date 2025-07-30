template<typename T, typename Op>
__global__ void kernelUnary(const T* input, size_t n, Op op, T* output) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = op(input[idx]);
}

template<typename T, typename Op>
__global__ void kernelBinary(const T* input1, const T* input2, size_t n, Op op, T* output) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = op(input1[idx], input2[idx]);
}

template<typename T_in, typename T_out1, typename T_out2, typename Op>
__global__ void kernelUnaryTwoOutputs(const T_in* input, size_t n, Op op, T_out1* output1, T_out2* output2) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        auto result_pair = op(input[idx]);
        output1[idx] = result_pair.first;
        output2[idx] = result_pair.second;
    }
}

template<typename T_in1, typename T_in2, typename T_out1, typename T_out2, typename Op>
__global__ void kernelBinaryTwoOutputs(const T_in1* input1, const T_in2* input2, size_t n, Op op, T_out1* output1, T_out2* output2) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        auto result_pair = op(input1[idx], input2[idx]);
        output1[idx] = result_pair.first;
        output2[idx] = result_pair.second;
    }
}