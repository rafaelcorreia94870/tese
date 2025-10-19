#ifndef EXPRESSION_CUH
#define EXPRESSION_CUH

#include <cuda_runtime.h>
#include <iostream>

namespace lcuda {
    template<typename T_raw> class Vector;
}

template<typename T, typename Op>
struct ReduceExpr {
    const T* d_in;
    size_t size;
    Op op;
    T identity;
    cudaStream_t stream;

    ReduceExpr(const T* d, size_t s, Op o, T id, cudaStream_t st) : d_in(d), size(s), op(o), identity(id), stream(st) {}
    
    operator T() const {
        return result();
    }

    T result() const {
        if (size == 0) return identity;

        T* d_temp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_temp, sizeof(T) * size));

        recursiveReduce(stream, d_in, d_temp, size, identity, op);

        T result_promoted;
        CUDA_CHECK(cudaMemcpyAsync(&result_promoted, d_temp, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_temp));
        return result_promoted;
    }
};

template<typename T, typename Op_map, typename Op_reduce>
struct MapReduceExpr {
    const T* d_in;
    size_t size;
    Op_map op_map;
    Op_reduce op_reduce;
    T identity;
    cudaStream_t stream;

    MapReduceExpr(const T* d, size_t s, Op_map o1, Op_reduce o2, T id, cudaStream_t st) : d_in(d), size(s), op_map(o1), op_reduce(o2), identity(id), stream(st) {}

    operator T() const {
        return result();
    }

    T result() const {
        T* d_reduce_temp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_reduce_temp, sizeof(T) * ((size + 1024 * 8 - 1) / (1024 * 8))));

        const int threads = 1024;
        const int itemsPerThread = 8;
        int blocks = (size + threads * itemsPerThread - 1) / (threads * itemsPerThread);
        
        mapReduceFusedKernel<<<blocks, threads, threads / 32 * sizeof(T), stream>>>(
            d_in, d_reduce_temp, size, identity, op_map, op_reduce
        );
        
        if (blocks > 1) {
            recursiveReduce(stream, d_reduce_temp, d_reduce_temp, blocks, identity, op_reduce);
        }

        T result_promoted;
        CUDA_CHECK(cudaMemcpyAsync(&result_promoted, d_reduce_temp, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_reduce_temp));
        return result_promoted;
    } 
};

template<typename T, typename Op_map, typename Op_reduce>
struct MapReduceExprBinary {
    const T* d_in1;
    const T* d_in2;
    size_t size;
    Op_map op_map;
    Op_reduce op_reduce;
    T identity;
    cudaStream_t stream;

    MapReduceExprBinary(const T* d1, const T* d2, size_t s, Op_map o1, Op_reduce o2, T id, cudaStream_t st)
        : d_in1(d1), d_in2(d2), size(s), op_map(o1), op_reduce(o2), identity(id), stream(st) {}

    operator T() const {
        return result();
    }

    T result() const {
        T* d_reduce_temp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_reduce_temp, sizeof(T) * ((size + 1024 * 8 - 1) / (1024 * 8))));

        const int threads = 1024;
        const int itemsPerThread = 8;
        int blocks = (size + threads * itemsPerThread - 1) / (threads * itemsPerThread);

        mapReduceFusedKernelBinary<<<blocks, threads, threads / 32 * sizeof(T), stream>>>(
            d_in1, d_in2, d_reduce_temp, size, identity, op_map, op_reduce
        );

        if (blocks > 1) {
            recursiveReduce(stream, d_reduce_temp, d_reduce_temp, blocks, identity, op_reduce);
        }

        T result_promoted;
        CUDA_CHECK(cudaMemcpyAsync(&result_promoted, d_reduce_temp, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_reduce_temp));
        return result_promoted;
    }
};

template<typename T, typename Op>
struct MapExprBinary {
    const T* d_in1;
    const T* d_in2;
    size_t size;
    Op op;
    cudaStream_t stream;

    MapExprBinary(const T* a, const T* b, size_t s, Op o, cudaStream_t st) : d_in1(a), d_in2(b), size(s), op(o), stream(st) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprBinary<T, ComposeBinaryUnary<Op, Op2>>(d_in1, d_in2, size, ComposeBinaryUnary<Op, Op2>(op, op2), stream);
    }

    template<typename Op2, typename T2>
    auto map(const lcuda::Vector<T2>& other, Op2 op2) const {
        return MapExprBinary<T, ComposeBinaryBinary<Op, Op2>>(d_in1, other.d_data, size, ComposeBinaryBinary<Op, Op2>(op, op2), stream);
    }

    template<typename Op2>
    auto reduce(Op2 op2, T identity) const {
        return MapReduceExprBinary<T, Op, Op2>(d_in1, d_in2, size, op, op2, identity, stream);
    }
};

template<typename T, typename Op, typename T_out1, typename T_out2>
struct MapExprBinaryOutput {
    const T* d_in1;
    const T* d_in2;
    T_out1* out1;
    T_out2* out2;
    size_t size;
    Op op;
    cudaStream_t stream;

    MapExprBinaryOutput(const T* a, const T* b, size_t s, Op o, T_out1* out1, T_out2* out2, cudaStream_t st)
        : d_in1(a), d_in2(b), size(s), op(o), out1(out1), out2(out2), stream(st) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprBinaryOutput<T, ComposeBinaryUnary<Op, Op2>, T_out1, T_out2>(d_in1, d_in2, size, ComposeBinaryUnary<Op, Op2>(op, op2),stream);
    }

    template<typename Op2, typename T2>
    auto map(const lcuda::Vector<T2>& other, Op2 op2) const {
        return MapExprBinaryOutput<T, ComposeBinaryBinary<Op, Op2>, T_out1, T_out2>(d_in1, other.d_data, size, ComposeBinaryBinary<Op, Op2>(op, op2), stream);
    }
};

template<typename T, typename Op>
struct MapExprUnary {
    const T* d_in;
    size_t size;
    Op op;
    cudaStream_t stream;

    MapExprUnary(const T* d, size_t s, Op o, cudaStream_t st): d_in(d), size(s), op(o), stream(st) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprUnary<T, ComposeUnaryUnary<Op, Op2>>(d_in, size, ComposeUnaryUnary<Op, Op2>(op, op2), stream);
    }

    template<typename Op2>
    auto map(const lcuda::Vector<T>& other, Op2 op2) const {
        return MapExprBinary<T, ComposeUnaryBinary<Op2, Op>>(d_in, other.d_data, size, ComposeUnaryBinary<Op2, Op>(op,op2), stream);
    }

    template<typename Op2>
    auto reduce(Op2 op2, T identity) const {
        return MapReduceExpr<T, Op, Op2>(d_in, size, op, op2, identity, stream);
    }
};

template<typename T, typename Op, typename T_out1, typename T_out2>
struct MapExprUnaryOutput {
    const T* d_in;
    T_out1* out1;
    T_out2* out2;
    size_t size;
    Op op;
    cudaStream_t stream;

    MapExprUnaryOutput(const T* d, size_t s, Op o, T_out1* out1, T_out2* out2, cudaStream_t st)
        : d_in(d), size(s), op(o), stream(st) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprUnaryOutput<T, ComposeUnaryUnary<Op, Op2>, T_out1, T_out2>(d_in, size, ComposeUnaryUnary<Op, Op2>(op, op2), stream);
    }

    template<typename Op2>
    auto map(const lcuda::Vector<T>& other, Op2 op2) const {
        return MapExprBinaryOutput<T, ComposeUnaryBinary<Op2, Op>, T_out1, T_out2>(d_in, other.d_data, size, ComposeUnaryBinary<Op2, Op>(op, op2), stream);
    }
};

#endif // EXPRESSION_CUH