
struct IntensiveComputation {
    __device__ float operator()(float x) const {
        for (int i = 0; i < 100; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
        }
        return x;
    }
};

struct DoublePlusA {
    __device__
    float operator()(float x, int a=1) const {
        return 2.0f * x + a;
    }
};

struct saxpy {
    __device__
    float operator()(float x, float y, float a) {

        return a*x + y;
    }
};

struct IntensiveComputationParams {
    __device__ float operator()(float x, int a=5, double b=2.3, bool flag=true) const {
        for (int i = 0; i < 100; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
        }
        return flag ? (x * a + b) : (x / a - b);
    }
};

struct IntensiveComputation2Inputs {
    __device__ float operator()(float x, float y, int a=5, double b=2.3, bool flag=true) const {
        for (int i = 0; i < 100; ++i) { 
            x = sin(x) * cos(x) + log(x + 1.0f);
            y = sin(y) * cos(y) + log(y + 1.0f);
        }
        return flag ? (x * y * a + b) : ((x + y) / a - b);
    }
};

struct Sum {
    __device__ float operator()(float x, float y) const {
        return x + y;
    }
};

struct Max {
    __device__ float operator()(float x, float y) const {
        return x > y ? x : y;
    }
};

struct Multiply {
    __device__ float operator()(float x, float y) const {
        return x * y;
    }
};