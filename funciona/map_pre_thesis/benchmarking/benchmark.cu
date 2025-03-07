#include "includes/compare.cuh"


void benchmark(const size_t N) {
    
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
}

int main() {
    const size_t N = 1000000;
    benchmark(N);
    return 0;
}
