
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

struct MandelbrotFunctor {
    int maxIter;
    float xmin, xmax, ymin, ymax;
    int width, height;

    __host__ __device__ MandelbrotFunctor() 
        : maxIter(100000), xmin(-2.0f), xmax(1.0f), ymin(-1.5f), ymax(1.5f), width(1024), height(1024) {}

    __host__ __device__ MandelbrotFunctor(int maxIter_, float xmin_, float xmax_, 
                                          float ymin_, float ymax_, int width_, int height_) 
        : maxIter(maxIter_), xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), width(width_), height(height_) {}

    __device__ int operator()(int idx) const {
        int i = idx % width;
        int j = idx / width;
        float x0 = xmin + i * (xmax - xmin) / width;
        float y0 = ymin + j * (ymax - ymin) / height;

        float x = 0.0f, y = 0.0f;
        int iter = 0;
        while (x * x + y * y <= 4.0f && iter < maxIter) {
            float xtemp = x * x - y * y + x0;
            y = 2.0f * x * y + y0;
            x = xtemp;
            iter++;
        }
        return iter;
    }
};