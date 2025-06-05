#include <iostream>
#include <vector>
#include <string>
#include <numeric>

struct DoubleIt {
    int operator()(int x) const { return 2 * x; }
};

struct AddTen {
    int operator()(int x) const { return x + 10; }
};

struct SquareIt {
    int operator()(int x) const { return x * x; }
};

template <typename T> class Vector;

template <typename BaseExpr, typename Op>
struct MapExpr {
    BaseExpr base_expr;
    Op op;

    MapExpr(BaseExpr b_expr, Op operation)
        : base_expr(b_expr), op(operation) {}

    void print() const {
        std::cout << "MapExpr: ";
        std::cout << "Base size: " << base_expr.size() << ", Operation: " << typeid(Op).name() << std::endl;
        for (size_t i = 0; i < base_expr.size(); ++i) {
            std::cout << "Index " << i << ": " << base_expr[i] << " -> " << op(base_expr[i]) << std::endl;
        }
        std::cout << "End of MapExpr" << std::endl;
    }

    auto operator[](size_t i) const {
        return op(base_expr[i]);
    }

    size_t size() const {
        return base_expr.size();
    }

    template <typename NextOp>
    auto map(NextOp next_op) const {
        return MapExpr<const MapExpr&, NextOp>(*this, next_op);
    }
};

template <typename T>
class Vector {
private:
    std::vector<T> data;

public:
    explicit Vector(size_t n) : data(n) {}
    Vector(const std::vector<T>& initial_data) : data(initial_data) {}

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
    size_t size() const { return data.size(); }

    std::vector<T>& get_data() { return data; }
    const std::vector<T>& get_data() const { return data; }

    void print(const std::string& label = "") const {
        if (!label.empty()) {
            std::cout << label << ": ";
        }
        for (const T& val : data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    template <typename Op>
    auto map(Op op) const {
        return MapExpr<const Vector&, Op>(*this, op);
    }

    template <typename BaseExpr, typename Op>
    Vector& operator=(const MapExpr<BaseExpr, Op>& expr) {
        if (data.size() != expr.size()) {
            data.resize(expr.size());
        }

        expr.print();

        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = expr[i];
        }
        return *this;
    }

    Vector& operator=(const Vector& other) {
        if (this != &other) {
            data = other.data;
        }
        return *this;
    }
};

int main() {

    Vector<int> my_vec(5);
    std::iota(my_vec.get_data().begin(), my_vec.get_data().end(), 1);
    my_vec.print("Original Vector");

    Vector<int> result_vec(5);

    std::cout << "\n--- Example 1: my_vec.map(DoubleIt()).map(AddTen()).map(SquareIt()) ---\n";
    result_vec = my_vec.map(DoubleIt()).map(AddTen()).map(SquareIt());
    result_vec.print("Result 1");
    std::cout << "Expected: 144 196 256 324 400\n";


    std::cout << "\n--- Example 2: my_vec.map(SquareIt()).map(DoubleIt()) ---\n";
    result_vec = my_vec.map(SquareIt()).map(DoubleIt());
    result_vec.print("Result 2");
    std::cout << "Expected: 2 8 18 32 50\n";

    return 0;
}