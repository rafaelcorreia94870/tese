#ifndef VECTOR_CUH
#define VECTOR_CUH

template <typename T>
concept VectorLike = requires(T a, size_t i) {
    { a.size() } -> std::convertible_to<size_t>;
    { a.begin() };
    { a.end() };
    { a.data() };
    { a[i] };
    //typename T::value_type;
} && std::ranges::range<T>;

#endif