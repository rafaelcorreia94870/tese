template<typename T>
struct VectorExt;

template<typename T, typename Op>
struct ReduceExpr {
    const T* d_in;
    size_t size;
    Op op;
    T identity;

    ReduceExpr(const T* d, size_t s, Op o, T id) : d_in(d), size(s), op(o), identity(id) {}

    T result() const {
        if (size == 0) return identity;

        T* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, sizeof(T)));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        recursiveReduce(stream, (d_in), d_temp, size, static_cast<T>(identity), op);

        T result_promoted;
        CUDA_CHECK(cudaMemcpyAsync(&result_promoted, d_temp, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_temp));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return (result_promoted);
    }
};

template<typename T, typename Op_map, typename Op_reduce>
struct MapReduceExpr {
    const T* d_in;
    size_t size;
    Op_map op_map;
    Op_reduce op_reduce;
    T identity;

    MapReduceExpr(const T* d, size_t s, Op_map o1, Op_reduce o2, T id) : d_in(d), size(s), op_map(o1), op_reduce(o2), identity(id) {}


    T result() const {
        T* d_reduce_temp;
        CUDA_CHECK(cudaMalloc(&d_reduce_temp, sizeof(T)));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        const int threads = 1024;
        const int itemsPerThread = 8;
        int blocks = (size + threads * itemsPerThread - 1) / (threads * itemsPerThread);

        
        mapReduceFusedKernel<<<blocks, threads, threads / 32 * sizeof(T), stream>>>(
            reinterpret_cast<const T*>(d_in), d_reduce_temp, size, identity, op_map, op_reduce
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (blocks > 1) {
            recursiveReduce(stream, d_reduce_temp, d_reduce_temp, blocks, identity, op_reduce);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        T result_promoted;
        CUDA_CHECK(cudaMemcpyAsync(&result_promoted, d_reduce_temp, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_reduce_temp));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return (result_promoted);
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

    MapReduceExprBinary(const T* d1, const T* d2, size_t s, Op_map o1, Op_reduce o2, T id)
        : d_in1(d1), d_in2(d2), size(s), op_map(o1), op_reduce(o2), identity(id) {}

    T result() const {
        T* d_reduce_temp;
        CUDA_CHECK(cudaMalloc(&d_reduce_temp, sizeof(T)));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        const int threads = 1024;
        const int itemsPerThread = 8;
        int blocks = (size + threads * itemsPerThread - 1) / (threads * itemsPerThread);

        mapReduceFusedKernelBinary<<<blocks, threads, threads / 32 * sizeof(T), stream>>>(
            d_in1, d_in2, d_reduce_temp, size, identity, op_map, op_reduce
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (blocks > 1) {
            recursiveReduce(stream, d_reduce_temp, d_reduce_temp, blocks, identity, op_reduce);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        T result_promoted;
        CUDA_CHECK(cudaMemcpyAsync(&result_promoted, d_reduce_temp, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_reduce_temp));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return (result_promoted);
    }
};

template<typename T, typename Op>
struct MapExprBinary {
    const T* d_in1;
    const T* d_in2;
    size_t size;
    Op op;

    MapExprBinary(const T* a, const T* b, size_t s, Op o) : d_in1(a), d_in2(b), size(s), op(o) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprBinary<T, ComposeBinaryUnary<Op, Op2>>(d_in1, d_in2, size, ComposeBinaryUnary<Op, Op2>(op, op2));
    }

    template<typename Op2, typename T2>
    auto map(const VectorExt<T2>& other, Op2 op2) const {
        return MapExprBinary<T, ComposeBinaryBinary<Op, Op2>>(d_in1, other.d_data, size, ComposeBinaryBinary<Op, Op2>(op, op2));
    }

    template<typename Op2>
    auto reduce(Op2 op2, T identity) const {
        return MapReduceExprBinary<T, Op, Op2>(d_in1, d_in2, size, op, op2, identity).result();
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

    MapExprBinaryOutput(const T* a, const T* b, size_t s, Op o, T_out1* out1, T_out2* out2)
        : d_in1(a), d_in2(b), size(s), op(o), out1(out1), out2(out2) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprBinaryOutput<T, ComposeBinaryUnary<Op, Op2>, T_out1, T_out2>(d_in1, d_in2, size, ComposeBinaryUnary<Op, Op2>(op, op2));
    }

    template<typename Op2, typename T2>
    auto map(const VectorExt<T2>& other, Op2 op2) const {
        return MapExprBinaryOutput<T, ComposeBinaryBinary<Op, Op2>, T_out1, T_out2>(d_in1, other.d_data, size, ComposeBinaryBinary<Op, Op2>(op, op2));
    }
};

template<typename T, typename Op>
struct MapExprUnary {
    const T* d_in;
    size_t size;
    Op op;

    MapExprUnary(const T* d, size_t s, Op o) : d_in(d), size(s), op(o) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprUnary<T, ComposeUnaryUnary<Op, Op2>>(d_in, size, ComposeUnaryUnary<Op, Op2>(op, op2));
    }

    template<typename Op2>
    auto map(const VectorExt<T>& other, Op2 op2) const {
        return MapExprBinary<T, ComposeUnaryBinary<Op2, Op>>(d_in, other.d_data, size, ComposeUnaryBinary<Op2, Op>(op,op2));
    }

    template<typename Op2>
    auto reduce(Op2 op2, T identity) const {
        return MapReduceExpr<T, Op, Op2>(d_in, size, op, op2, identity).result();
    }

};

template<typename T, typename Op, typename T_out1, typename T_out2>
struct MapExprUnaryOutput {
    const T* d_in;
    T_out1* out1;
    T_out2* out2;
    size_t size;
    Op op;

    MapExprUnaryOutput(const T* d, size_t s, Op o, T_out1* out1, T_out2* out2)
        : d_in(d), size(s), op(o) {}

    template<typename Op2>
    auto map(Op2 op2) const {
        return MapExprUnaryOutput<T, ComposeUnaryUnary<Op, Op2>, T_out1, T_out2>(d_in, size, ComposeUnaryUnary<Op, Op2>(op, op2));
    }

    template<typename Op2>
    auto map(const VectorExt<T>& other, Op2 op2) const {
        return MapExprBinaryOutput<T, ComposeUnaryBinary<Op2, Op>, T_out1, T_out2>(d_in, other.d_data, size, ComposeUnaryBinary<Op2, Op>(op, op2));
    }
};