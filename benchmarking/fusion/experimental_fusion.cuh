#include <cstdio>
#include <tuple>
#include <utility>
#include <type_traits>
#include <cuda_runtime.h>

template <typename... Args>
__host__ __device__
constexpr auto my_make_tuple(Args&&... args) {
    return std::tuple<Args...>(std::forward<Args>(args)...);
}

template <typename F, typename Tuple, std::size_t... I>
__device__ auto device_apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
    return f(std::get<I>(t)...);
}

template <typename F, typename Tuple>
__device__ auto device_apply(F&& f, Tuple&& t) {
    constexpr std::size_t N = std::tuple_size<std::decay_t<Tuple>>::value;
    return device_apply_impl(std::forward<F>(f), std::forward<Tuple>(t), std::make_index_sequence<N>{});
}

template <typename F, typename = void>
struct FunctorArity : std::integral_constant<int, 1> {};

template <typename F>
struct FunctorArity<F, std::void_t<decltype(&F::operator())>> {
private:
    template <typename R, typename C, typename T>
    static std::integral_constant<int, 1> test(R(C::*)(T) const);

    template <typename R, typename C, typename... T>
    static std::integral_constant<int, sizeof...(T)> test(R(C::*)(T...) const);

    static std::integral_constant<int, 1> test(...);

public:
    static constexpr int value = decltype(test(&F::operator()))::value;
};


template <std::size_t... I, typename Tuple>
__host__ __device__ auto tuple_head_impl(Tuple&& tpl, std::index_sequence<I...>) {
    return my_make_tuple(std::get<I>(std::forward<Tuple>(tpl))...);
}

template <std::size_t N, typename Tuple>
__host__ __device__ auto tuple_head(Tuple&& tpl) {
    return tuple_head_impl(std::forward<Tuple>(tpl), std::make_index_sequence<N>{});
}

template <std::size_t N, typename Tuple, std::size_t... I>
__host__ __device__ auto tuple_tail_impl(Tuple&& tpl, std::index_sequence<I...>) {
    return my_make_tuple(std::get<N + I>(std::forward<Tuple>(tpl))...);
}

template <std::size_t N, typename Tuple>
__host__ __device__ auto tuple_tail(Tuple&& tpl) {
    constexpr auto size = std::tuple_size<std::decay_t<Tuple>>::value;
    static_assert(N <= size, "N exceeds tuple size");
    return tuple_tail_impl<N>(std::forward<Tuple>(tpl), std::make_index_sequence<size - N>{});
}

template <typename... Fs>
struct Pipeline;

template <typename F, typename... Rest>
struct Pipeline<F, Rest...> {
    F f;
    Pipeline<Rest...> rest;

    __host__ __device__ Pipeline(F f, Rest... rest) : f(f), rest(rest...) {}

    template <typename... Args>
    __device__ auto operator()(Args&&... args) const {
        auto all_args = my_make_tuple(std::forward<Args>(args)...);
        constexpr int arity = FunctorArity<F>::value;
        static_assert(sizeof...(Args) >= arity, "Not enough arguments");

        auto head = tuple_head<arity>(all_args);
        auto tail = tuple_tail<arity>(all_args);

        auto result = device_apply(f, head);
        auto next_args = std::tuple_cat(my_make_tuple(result), tail);
        return device_apply(rest, next_args);
    }
};

template <typename F>
struct Pipeline<F> {
    F f;
    __host__ __device__ Pipeline(F f) : f(f) {}

    template <typename... Args>
    __device__ auto operator()(Args&&... args) const {
        auto all_args = my_make_tuple(std::forward<Args>(args)...);
        constexpr int arity = FunctorArity<F>::value;
        static_assert(sizeof...(Args) >= arity, "Not enough arguments");

        auto head = tuple_head<arity>(all_args);
        return device_apply(f, head);
    }
};

template <typename... Fs>
__host__ __device__ auto make_pipeline(Fs... fs) {
    return Pipeline<Fs...>(fs...);
}


template <typename PipelineT, typename... Args>
__global__ void pipeline_kernel(PipelineT pipeline, int N, Args... arrays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        auto result = pipeline((arrays[idx])...);
        std::get<sizeof...(Args) - 1>(std::forward_as_tuple(arrays...))[idx] = result;
    }
}

template <typename PipelineT, typename... Arrays>
void launch_kernel(PipelineT pipeline, int N, Arrays... arrays) {
    constexpr int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    pipeline_kernel<<<gridSize, blockSize>>>(pipeline, N, arrays...);
    cudaDeviceSynchronize();
}

std::chrono::duration<double> twointensivecomputations_expr(size_t n, size_t loop_count) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> vec1(n, 1.0f);
    std::vector<float> result(n, 0);
    float *d_vec1, *d_result;
    cudaMalloc(&d_vec1, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));
    cudaMemcpyAsync(d_vec1, vec1.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_result, result.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);

    auto pipeline = make_pipeline(BenchmarkingComputations(loop_count), BenchmarkingComputations(loop_count));
    launch_kernel(pipeline, n, d_vec1, d_vec1, d_result);
    cudaMemcpyAsync(result.data(), d_result, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    //std::cout << "Result[0]: " << result[0] << std::endl;
    cudaFree(d_vec1);
    cudaFree(d_result);
    cudaStreamDestroy(stream);

    return end - start;
}


std::chrono::duration<double> tensimplecomputations_expr(size_t n) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> vec1(n, 1.0f);
    std::vector<float> result(n, 0);
    float *d_vec1, *d_result;
    cudaMalloc(&d_vec1, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));
    cudaMemcpyAsync(d_vec1, vec1.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_result, result.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);

    auto pipeline = make_pipeline(SimpleComputation(),SimpleComputation(),
                                  SimpleComputation(), SimpleComputation(),
                                  SimpleComputation(), SimpleComputation(),
                                  SimpleComputation(), SimpleComputation(),
                                  SimpleComputation(), SimpleComputation());
    launch_kernel(pipeline, n, d_vec1, d_vec1, d_vec1, d_vec1, d_vec1, d_vec1, d_vec1, d_vec1, d_vec1, d_vec1, d_result);
    cudaMemcpyAsync(result.data(), d_result, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    //std::cout << "Result[0]: " << result[0] << std::endl;
    cudaFree(d_vec1);
    cudaFree(d_result);
    cudaStreamDestroy(stream);

    return end - start;
}