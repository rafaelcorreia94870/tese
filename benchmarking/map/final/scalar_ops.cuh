#pragma once

#include "fused_gpu_vec.cuh"
#include "compose.cuh"

namespace lcuda {

//==================================================================================
//==   SCALAR OPERATION FUNCTORS
//==================================================================================

template<typename T>
struct AddScalar {
    T val;
    __host__ __device__ AddScalar(T v) : val(v) {}
    __host__ __device__ T operator()(T x) const { return x + val; }
};

template<typename T>
struct MultiplyScalar {
    T val;
    __host__ __device__ MultiplyScalar(T v) : val(v) {}
    __host__ __device__ T operator()(T x) const { return x * val; }
};

template<typename T>
struct SubtractScalar {
    T val;
    __host__ __device__ SubtractScalar(T v) : val(v) {}
    __host__ __device__ T operator()(T x) const { return x - val; }
};

template<typename T>
struct SubtractFromScalar {
    T val;
    __host__ __device__ SubtractFromScalar(T v) : val(v) {}
    __host__ __device__ T operator()(T x) const { return val - x; }
};

template<typename T>
struct DivideScalar {
    T val;
    __host__ __device__ DivideScalar(T v) : val(v) {}
    __host__ __device__ T operator()(T x) const { return x / val; }
};

template<typename T>
struct DivideByScalar {
    T val;
    __host__ __device__ DivideByScalar(T v) : val(v) {}
    __host__ __device__ T operator()(T x) const { return val / x; }
};


//==================================================================================
//==   OPERATOR OVERLOADS FOR lcuda::Vector (ENTRY POINTS)
//==================================================================================

template<typename T>
auto operator+(const Vector<T>& vec, T scalar) {
    return vec.map(AddScalar<T>(scalar));
}

template<typename T>
auto operator+(T scalar, const Vector<T>& vec) {
    return vec.map(AddScalar<T>(scalar));
}

template<typename T>
auto operator*(const Vector<T>& vec, T scalar) {
    return vec.map(MultiplyScalar<T>(scalar));
}

template<typename T>
auto operator*(T scalar, const Vector<T>& vec) {
    return vec.map(MultiplyScalar<T>(scalar));
}

template<typename T>
auto operator-(const Vector<T>& vec, T scalar) {
    return vec.map(SubtractScalar<T>(scalar));
}

template<typename T>
auto operator-(T scalar, const Vector<T>& vec) {
    return vec.map(SubtractFromScalar<T>(scalar));
}

template<typename T>
auto operator/(const Vector<T>& vec, T scalar) {
    return vec.map(DivideScalar<T>(scalar));
}

template<typename T>
auto operator/(T scalar, const Vector<T>& vec) {
    return vec.map(DivideByScalar<T>(scalar));
}

//==================================================================================
//==   OPERATOR OVERLOADS FOR EXPRESSIONS (CHAINING)
//==================================================================================

// --- For Unary Expressions ---

template<typename T, typename Op>
auto operator+(const MapExprUnary<T, Op>& expr, T scalar) {
    return expr.map(AddScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator+(T scalar, const MapExprUnary<T, Op>& expr) {
    return expr.map(AddScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator*(const MapExprUnary<T, Op>& expr, T scalar) {
    return expr.map(MultiplyScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator*(T scalar, const MapExprUnary<T, Op>& expr) {
    return expr.map(MultiplyScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator-(const MapExprUnary<T, Op>& expr, T scalar) {
    return expr.map(SubtractScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator-(T scalar, const MapExprUnary<T, Op>& expr) {
    return expr.map(SubtractFromScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator/(const MapExprUnary<T, Op>& expr, T scalar) {
    return expr.map(DivideScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator/(T scalar, const MapExprUnary<T, Op>& expr) {
    return expr.map(DivideByScalar<T>(scalar));
}


// --- For Binary Expressions ---

template<typename T, typename Op>
auto operator+(const MapExprBinary<T, Op>& expr, T scalar) {
    return expr.map(AddScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator+(T scalar, const MapExprBinary<T, Op>& expr) {
    return expr.map(AddScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator*(const MapExprBinary<T, Op>& expr, T scalar) {
    return expr.map(MultiplyScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator*(T scalar, const MapExprBinary<T, Op>& expr) {
    return expr.map(MultiplyScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator-(const MapExprBinary<T, Op>& expr, T scalar) {
    return expr.map(SubtractScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator-(T scalar, const MapExprBinary<T, Op>& expr) {
    return expr.map(SubtractFromScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator/(const MapExprBinary<T, Op>& expr, T scalar) {
    return expr.map(DivideScalar<T>(scalar));
}

template<typename T, typename Op>
auto operator/(T scalar, const MapExprBinary<T, Op>& expr) {
    return expr.map(DivideByScalar<T>(scalar));
}

} // namespace lcuda