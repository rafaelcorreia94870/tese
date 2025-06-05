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

struct First {
    __device__ int operator()(int x) const { return x + 1; }
};

struct Second {
    __device__ float operator()(int x) const { return x + 2.5f; }
};

struct DoubleIt {
    __device__ float operator()(float x) const { return 2 * x; }
};

struct Add2Inputs {
    __device__ float operator()(float x, float y) const { return x + y; }
};

struct CustomObject {
    int n_loops;
    bool flag;
    float value;
    __host__ __device__ CustomObject(int n, bool f, float v) : n_loops(n), flag(f), value(v) {}
    __host__ __device__ float customfunction(float x) const {
        float result = x;
        for (int i = 0; i < n_loops; ++i) {
            result += value;
        }
        return flag ? result : -result;
    }
};

struct ManyInputs {
    __device__ float operator()(float x, float y, float z, int w, CustomObject v) const {
        return v.customfunction(x) + v.customfunction(y) + v.customfunction(z) + w;
    }
};

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

int main() {
    constexpr int N = 2'500'000;
    std::vector<float> h_x(N, 1);
    std::vector<float> h_y(N, 2);
    std::vector<float> h_z(N, 3);

    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    auto pipeline = make_pipeline(DoubleIt(), Add2Inputs(), Add2Inputs(), DoubleIt(), DoubleIt());
    launch_kernel(pipeline, N, d_x, d_y, d_z, d_x);

    cudaMemcpy(h_x.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; ++i) {
        printf("Result[%d] = %f\n", i, h_x[i]);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    std::vector<int> first_input(N, 1);
    std::vector<float> result(N);

    int *d_first_input;
    float *d_result;
    cudaMalloc(&d_first_input, N * sizeof(int));
    cudaMalloc(&d_result, N * sizeof(float));
    cudaMemcpy(d_first_input, first_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    auto first_pipeline = make_pipeline(First(), Second());
    launch_kernel(first_pipeline, N, d_first_input, d_result);
    cudaMemcpy(result.data(), d_result, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; ++i) {
        printf("First Pipeline Result[%d] = %f\n", i, result[i]);
    }

    cudaFree(d_first_input);
    cudaFree(d_result);

    std::vector<float> many_inputs_input_x(N, 1.0f);
    std::vector<float> many_inputs_input_y(N, 2.0f);
    std::vector<float> many_inputs_input_z(N, 3.0f);
    std::vector<int> many_inputs_input_w(N, 4);
    std::vector<CustomObject> many_inputs_custom_objects(N, CustomObject(10, true, 1.0f));

    std::vector<float> many_inputs_result(N);
    float *d_many_inputs_result;
    float *d_many_inputs_input_x, *d_many_inputs_input_y, *d_many_inputs_input_z;
    int *d_many_inputs_input_w;
    CustomObject *d_many_inputs_custom_objects;
    cudaMalloc(&d_many_inputs_custom_objects, N * sizeof(CustomObject));
    cudaMalloc(&d_many_inputs_input_x, N * sizeof(float));
    cudaMalloc(&d_many_inputs_input_y, N * sizeof(float));
    cudaMalloc(&d_many_inputs_input_z, N * sizeof(float));
    cudaMalloc(&d_many_inputs_input_w, N * sizeof(int));
    cudaMalloc(&d_many_inputs_result, N * sizeof(float));

    cudaMemcpy(d_many_inputs_input_x, many_inputs_input_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_many_inputs_input_y, many_inputs_input_y.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_many_inputs_input_z, many_inputs_input_z.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_many_inputs_input_w, many_inputs_input_w.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_many_inputs_custom_objects, many_inputs_custom_objects.data(), N * sizeof(CustomObject), cudaMemcpyHostToDevice);

    auto many_inputs_pipeline = make_pipeline(ManyInputs());
    launch_kernel(many_inputs_pipeline, N,
                  d_many_inputs_input_x,
                  d_many_inputs_input_y,
                  d_many_inputs_input_z,
                  d_many_inputs_input_w,
                  d_many_inputs_custom_objects,
                  d_many_inputs_result);

    cudaMemcpy(many_inputs_result.data(), d_many_inputs_result, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; ++i) {
        printf("Many Inputs Pipeline Result[%d] = %f\n", i, many_inputs_result[i]);
    }

    cudaFree(d_many_inputs_input_x);
    cudaFree(d_many_inputs_input_y);
    cudaFree(d_many_inputs_input_z);
    cudaFree(d_many_inputs_input_w);
    cudaFree(d_many_inputs_custom_objects);
    cudaFree(d_many_inputs_result);

    return 0;
}
