#ifndef OP_CUH
#define OP_CUH

template <typename T>
concept Operation = requires(T a) {
    { a.identity() };
};

#endif