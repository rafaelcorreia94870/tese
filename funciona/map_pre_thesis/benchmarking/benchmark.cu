#include "includes/compare.cuh"


void benchmark(const size_t N) {
    /*
    oneInputInPlace(N);
    oneInputOutput(N);
    twoInputsInPlace(N);
    twoInputsOutput(N);
    
    oneInputInPlaceParameters(N);
    oneInputOutputParameters(N);
    twoInputsInPlaceParameters(N);
    twoInputsOutputParameters(N);
    
    doublePlusA(N);
    mysaxpy(N);
    
    */
    std::cout << "Simple reduce" << std::endl;
    simpleReduce(N);
    ReduceMult(N);
    ReduceMax(N);
}

int main() {
    const size_t N = 1'000'000'000;
    benchmark(N);
    return 0;
}
