#ifndef TIME_STRUCT
#define TIME_STRUCT

struct two_times_struct
{
    std::chrono::milliseconds cuda_time;
    std::chrono::milliseconds thrust_time;
};
#endif