#ifndef OP_CUH
#define OP_CUH

namespace rafa {

template <typename T>
concept Operation = requires(T a) {
    { a.identity() };
};
}

#endif