#include "includes/compare.cuh"

using BenchmarkFunction = std::function<two_times_struct(size_t, bool)>;

std::string formatNumber(long long number) {
    std::stringstream ss;
    ss.imbue(std::locale("en_US.UTF-8"));
    ss << number;
    return ss.str();
}

void test_capabilities(const size_t N){

    std::cout << "Testing Maps" << std::endl;
    oneInputInPlace(N);
    std::cout << "------------------------------------------------" << std::endl;
    oneInputOutput(N);
    std::cout << "------------------------------------------------" << std::endl;
    twoInputsInPlace(N);
    std::cout << "------------------------------------------------" << std::endl;
    twoInputsOutput(N);
    std::cout << "------------------------------------------------" << std::endl;
    
    oneInputInPlaceParameters(N);
    std::cout << "------------------------------------------------" << std::endl;
    oneInputOutputParameters(N);
    std::cout << "------------------------------------------------" << std::endl;
    twoInputsInPlaceParameters(N);
    std::cout << "------------------------------------------------" << std::endl;
    twoInputsOutputParameters(N);
    std::cout << "------------------------------------------------" << std::endl;
    
    doublePlusA(N);
    std::cout << "------------------------------------------------" << std::endl;
    mysaxpy(N);
    std::cout << "------------------------------------------------" << std::endl;
    
    std::cout << "Testing reduces" << std::endl;
    simpleReduce(N);
    std::cout << "------------------------------------------------" << std::endl;
    ReduceMult(N);
    std::cout << "------------------------------------------------" << std::endl;
    ReduceMax(N);
    std::cout << "------------------------------------------------" << std::endl;
}

void benchmark(const size_t MIN_N,const size_t MAX_N, std::vector<BenchmarkFunction>& functions, bool verbose = false){
    two_times_struct two_times;
    //for each function on a vector do the following
    for (auto& function : functions){
        std::string functionName = typeid(function).name();  
        std::cout << "Function: " << functionName << std::endl;
        for(size_t N = MAX_N; N >= MIN_N; N /= 10){
            std::cout << "N = " << formatNumber(N) << std::endl;
            two_times = function(N, verbose);
            std::cout << "CUDA time: " << two_times.cuda_time.count() << " ms" << "\nThrust time: " << two_times.thrust_time.count() << " ms" << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
        }
        std::cout << "#########################################################" << std::endl;
    }
    
}

int main() {
    const size_t MAX_N = 1'000'000'000;
    const size_t MIN_N = 10'000;

    std::vector<BenchmarkFunction> functions = { mysaxpy };

    benchmark(MIN_N, MAX_N, functions, false);
    return 0;
}
