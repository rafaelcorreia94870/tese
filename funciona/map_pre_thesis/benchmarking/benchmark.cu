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
    ReduceSum(N);
    std::cout << "------------------------------------------------" << std::endl;
    ReduceMult(N);
    std::cout << "------------------------------------------------" << std::endl;
    ReduceMax(N);
    std::cout << "------------------------------------------------" << std::endl;
}


void mandel_brot(const size_t width, const size_t height, const int maxIter, const bool enable_prints = true, const bool enable_image = false){
    if (enable_prints){
        MandelbrotBenchmark(width, height, maxIter, enable_prints);
    }
    if(enable_image){
        MandelbrotRender(width, height, maxIter);
    }
}

void benchmark(const size_t MIN_N,const size_t MAX_N, const size_t NUMB_REPEAT, std::vector<BenchmarkFunction>& functions, bool verbose = false){
    two_times_struct two_times;
    int function_index = 0;
    std::vector<std::tuple<int, unsigned __int64, unsigned __int64, double, double>> results;

    for(size_t i = 0; i < NUMB_REPEAT; i++){
        for (auto& function : functions){
            std::cout << "Function index: " << function_index << std::endl;
            std::cout << "#################### LOOP " << i+1 << " ####################" << std::endl;
            function_index++;
            for(size_t N = MAX_N; N >= MIN_N; N /= 10){
                std::cout << "N = " << formatNumber(N) << std::endl;
                two_times = function(N, verbose);
                double cudaTime = static_cast<double>(two_times.cuda_time.count());
                double thrustTime = static_cast<double>(two_times.thrust_time.count());
                std::cout << "CUDA time: " << cudaTime  << " ms" << "\nThrust time: " << thrustTime << " ms" << std::endl;
                std::cout << "------------------------------------------------" << std::endl;
                results.emplace_back(function_index, i + 1, N, cudaTime, thrustTime);
            }
            std::cout << "#########################################################" << std::endl;
        }
        function_index = 0;
    } 

    std::cout << "Function,Loop, N, CUDA Time (ms),Thrust Time (ms)\n";
    for (const auto& [func, loop, N, cuda_time, thrust_time] : results) {
        std::cout << func << ";" << loop << ";" << N << ";" << cuda_time << ";" << thrust_time << "\n";
    }
}


void benchmark(const size_t NUMB_REPEAT, const size_t width, const size_t height, const int maxIter, const int minIter){
    two_times_struct two_times;
    std::vector<std::tuple<int, unsigned __int64, unsigned __int64, double, double>> results;
    for (size_t i = 0; i < NUMB_REPEAT; i++){
        std::cout << "#################### LOOP " << i+1 << " ####################" << std::endl;
        for (int iter = maxIter; iter >= minIter; iter /= 10){
            std::cout << "Iter = " << formatNumber(iter) << std::endl;
            two_times = MandelbrotBenchmark(width, height, iter, true);
            std::cout << "------------------------------------------------" << std::endl;
            results.emplace_back(9, i+1, iter, two_times.cuda_time.count(), two_times.thrust_time.count());
        }
        std::cout << "#########################################################" << std::endl;
    }

    std::cout << "Function,Loop, N, CUDA Time (ms),Thrust Time (ms)\n";
    for (const auto& [func, loop, N, cuda_time, thrust_time] : results) {
        std::cout << func << ";" << loop << ";" << N << ";" << cuda_time << ";" << thrust_time << "\n";
    }

}

int main() {
    const size_t MAX_N = 1'000'000'000;
    const size_t MIN_N = 10'000;
    size_t width = 1024, height = 1024;
    int maxIter = 100'000;

    std::vector<BenchmarkFunction> functions = {mysaxpy, mysaxpyReverse,
        IntensiveComputationCompare, IntensiveComputationCompareReverse,
        ReduceMax, ReduceMaxReverse,
        ReduceSum, ReduceSumReverse 
        };
    

    //benchmark(MIN_N, MAX_N, 10, functions, false);
    //mandel_brot(width, height, maxIter, true, true);
    benchmark(10, width, height, 10'000, 100);
    //benchmark(10, width, height, MAX_N, MIN_N);
    return 0;
}
